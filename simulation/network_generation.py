"""Resolve reusable format-three configurations with per-site returns and workstation schedules."""
import copy
import json
import random
from .models import Record
from .dynamic_config import validate_dynamic_scenario
from .return_generation import derive_stream_seed, generate_return_batches, generate_products, validate_generation_config
from .schedule_generation import generate_windows, station_schedule_settings
from .coordinate_generation import COORDINATE_SITE_GROUPS, generate_site_coordinates, validate_coordinate_generation


def network_stream_name(kind: str, *identifiers: str) -> str:
    """Encode stream identities without separator collisions or dependence on input list order."""
    return json.dumps(['network-v2', kind, *identifiers], separators=(',', ':'), ensure_ascii=True)


def collection_generation_proxy(configuration: Record, point: Record, plants: list[Record]) -> Record:
    """Adapt one collection point to existing single-site generation without changing network inputs."""
    proxy = copy.deepcopy(configuration)
    proxy.update(format_version=2, policy='static_quality_unaware', plant=copy.deepcopy(plants[0]),
        collection_point=copy.deepcopy(point), generation=copy.deepcopy(point['generation']), trips=[])
    for vehicle in proxy['vehicles']:
        vehicle['initial_site'] = plants[0]['id']
    return proxy


def validate_network_generation(configuration: Record) -> None:
    """Check unresolved network, per-point quality routes and fixed/generated calendar declarations."""
    if configuration['format_version'] != 3 or any(key in configuration for key in ('products', 'arrival_batches', 'seed_provenance')):
        raise ValueError('Use an unresolved format-three configuration without products, batches or seed provenance')
    if not configuration['plants'] or not configuration['collection_points']:
        raise ValueError('Seeded networks require plants and collection points')
    physical = copy.deepcopy(configuration)
    if 'coordinate_generation' in configuration:
        settings = configuration['coordinate_generation']
        validate_coordinate_generation(settings)
        # Geometry is replaced before simulation, so source coordinates may be omitted.
        for _, group in COORDINATE_SITE_GROUPS:
            for site in physical[group]:
                site['coordinates'] = [settings['x_range'][0], settings['y_range'][0]]
    for plant in physical['plants']:
        for station in plant['workstations']:
            if station_schedule_settings(configuration, station) is not None:
                station['windows'] = []
    physical.update(products=[], seed_provenance={'source': 'network_generation_validation'})
    validate_dynamic_scenario(physical)
    for point in physical['collection_points']:
        # Every eligible class must be processable at every remanufacturing destination,
        # preserving the existing quality-unaware destination boundary.
        for plant in physical['plants']:
            proxy = collection_generation_proxy(configuration, point, [plant])
            proxy['recycling_facilities'] = copy.deepcopy(physical['recycling_facilities'])
            validate_generation_config(proxy)


def prepare_network_scenario(configuration: Record, seed: int) -> tuple[Record, bytes]:
    """Freeze independently sampled site returns, qualities and workstation calendars as canonical LF bytes."""
    if type(seed) is not int:
        raise ValueError('Generation requires an explicit integer seed')
    validate_network_generation(configuration)
    scenario = copy.deepcopy(configuration)
    seeds: Record = {'returns': {}, 'qualities': {}, 'workstation_schedules': {}}
    if 'coordinate_generation' in scenario:
        seeds['coordinates'] = generate_site_coordinates(scenario, seed)
    point_seeds = {}
    for plant in scenario['plants']:
        station_seeds = {}
        for station in plant['workstations']:
            settings = station_schedule_settings(configuration, station)
            if settings is not None:
                station_seed = derive_stream_seed(seed, network_stream_name('schedule', plant['id'], station['id']))
                station_seeds[station['id']] = station_seed
                station['window_generation'] = copy.deepcopy(settings)
                station['windows'] = generate_windows(settings, random.Random(station_seed))
        if station_seeds:
            seeds['workstation_schedules'][plant['id']] = station_seeds
    products, batches = [], []
    for point in sorted(scenario['collection_points'], key=lambda item: item['id']):
        identifier = point['id']
        point_seed = derive_stream_seed(seed, network_stream_name('collection', identifier))
        point_seeds[identifier] = point_seed
        returns_seed = derive_stream_seed(point_seed, 'arrivals_batches')
        quality_seed = derive_stream_seed(point_seed, 'qualities')
        seeds['returns'][identifier], seeds['qualities'][identifier] = returns_seed, quality_seed
        proxy = collection_generation_proxy(configuration, point, scenario['plants'])
        local_batches = generate_return_batches(proxy, random.Random(returns_seed))
        local_products = generate_products(proxy, local_batches, random.Random(quality_seed))
        for batch in local_batches:
            batch.update(id=f"{identifier}:{batch['id']}", origin=identifier,
                         product_ids=[f'{identifier}:{product_id}' for product_id in batch['product_ids']])
        for product in local_products:
            product.update(id=f"{identifier}:{product['id']}", batch_id=f"{identifier}:{product['batch_id']}")
        batches.extend(local_batches)
        products.extend(local_products)
    scenario.update(products=sorted(products, key=lambda item: (item['release_time'], item['origin'], item['id'])),
        arrival_batches=sorted(batches, key=lambda item: (item['release_time'], item['origin'], item['id'])),
        seed_provenance={'source': 'seeded_network_configuration', 'master_seed': seed, 'generation_version': 2,
            'algorithm': 'site-streams-v2/exponential-inverse-cdf/weighted-cdf/alternating-uniform-windows-v1',
            'point_seeds': point_seeds, 'stream_seeds': seeds})
    if 'coordinate_generation' in scenario:
        scenario['seed_provenance']['coordinate_algorithm'] = 'site-type-id/uniform-xy-v1'
    validate_dynamic_scenario(scenario)
    scenario_bytes = (json.dumps(scenario, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')
    return json.loads(scenario_bytes), scenario_bytes
