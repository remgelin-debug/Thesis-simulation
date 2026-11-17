"""Resolved dynamic-network validation and backward-compatible input conversion."""
import copy
from .models import Record
from .inventory_config import validate_inventory_scenario
from .plant_network import configured_plants


def as_dynamic_scenario(scenario: Record) -> Record:
    """Copy static resolved physical inputs into the multi-collection dynamic execution shape."""
    dynamic = copy.deepcopy(scenario)
    if dynamic['format_version'] == 2:
        dynamic['collection_points'] = [dynamic.pop('collection_point')]
    policy = dynamic.get('policy') if dynamic.get('policy') in ('dynamic_quality_unaware', 'dynamic_quality_aware') else 'dynamic_quality_unaware'
    if 'plants' in dynamic:
        dynamic['plant'] = copy.deepcopy(dynamic['plants'][0])
    dynamic.update(format_version=3, policy=policy, trips=[])
    return dynamic


def validate_dynamic_scenario(scenario: Record) -> None:
    """Validate dynamic sites/origins and reuse only physical limits from the inventory schema."""
    if scenario['format_version'] != 3 or scenario['policy'] not in ('dynamic_quality_unaware', 'dynamic_quality_aware') or scenario.get('trips', []):
        raise ValueError('Dynamic format 3 requires a dynamic policy and no fixed trips')
    plants = configured_plants(scenario)
    if not plants:
        raise ValueError('At least one remanufacturing plant is required')
    points = scenario['collection_points']
    point_ids = {point['id'] for point in points}
    identifiers = [site['id'] for site in [*plants, *points, *scenario['recycling_facilities']]]
    identifiers += [vehicle['id'] for vehicle in scenario['vehicles']] + [product['id'] for product in scenario['products']]
    if len(set(identifiers)) != len(identifiers) or not all(isinstance(identifier, str) and identifier for identifier in identifiers):
        raise ValueError('Dynamic entity IDs must be unique nonempty strings')
    import math
    if any(len(point['coordinates']) != 2 or any(type(value) not in (int, float) or not math.isfinite(value)
           for value in point['coordinates']) for point in points):
        raise ValueError('Collection coordinates must be finite two-dimensional values')
    if any(product['origin'] not in point_ids for product in scenario['products']):
        raise ValueError('Product origin must reference a configured collection point')
    proxy = copy.deepcopy(scenario)
    dummy = '__empty_collection__'
    while dummy in identifiers:
        dummy += '_'
    proxy['plant'] = plants[0]
    proxy.update(format_version=2, policy='static_quality_unaware', trips=[],
                 collection_point=points[0] if points else {'id': dummy, 'coordinates': [0, 0]})
    for product in proxy['products']:
        product['origin'] = proxy['collection_point']['id']
    facilities = {plant['id'] for plant in plants} | {site['id'] for site in scenario['recycling_facilities']}
    if any(vehicle['initial_site'] not in facilities for vehicle in scenario['vehicles']):
        raise ValueError('Vehicles must start at a configured facility')
    for plant in plants:
        proxy['plant'] = plant
        for vehicle in proxy['vehicles']:
            vehicle['initial_site'] = plant['id']
        validate_inventory_scenario(proxy)
