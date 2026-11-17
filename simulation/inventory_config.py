"""Validation of resolved fixed trips with shared inventory and optional recycling."""
import math
from .models import Record


def validate_inventory_scenario(scenario: Record) -> None:
    """Validate version two, inventory/range limits and fixed trips ending at facilities."""
    if scenario['format_version'] != 2 or scenario['policy'] not in ('static_quality_unaware', 'static_quality_aware'):
        raise ValueError('Inventory runs require format 2 and a supported static policy')
    plant, point = scenario['plant'], scenario['collection_point']
    capacity = plant['inventory_capacity']
    if capacity is not None and (type(capacity) is not int or capacity <= 0):
        raise ValueError('Inventory capacity must be a positive integer or null')
    identifiers = [plant['id'], point['id']]
    numbers = [scenario['horizon']]
    identifiers.extend(recycler['id'] for recycler in scenario['recycling_facilities'])
    for site in (plant, point, *scenario['recycling_facilities']):
        if len(site['coordinates']) != 2:
            raise ValueError('Coordinates must have two components')
        numbers.extend(site['coordinates'])
    stations = plant['workstations']
    if len({s['id'] for s in stations}) != len(stations) or len({s['operation_id'] for s in stations}) != len(stations):
        raise ValueError('Workstation IDs and operation IDs must be unique within the plant')
    for station in stations:
        if not isinstance(station['id'], str) or not station['id'] or station['duration'] <= 0:
            raise ValueError('Workstation IDs must be strings and durations positive')
        numbers.append(station['duration'])
        previous_end = -math.inf
        for opening, closing in station['windows']:
            if not 0 <= opening < closing or opening < previous_end:
                raise ValueError('Calendar windows must be ordered and not overlap')
            numbers.extend((opening, closing))
            previous_end = closing
    vehicles = {v['id']: v for v in scenario['vehicles']}
    identifiers.extend(v['id'] for v in scenario['vehicles'])
    for vehicle in vehicles.values():
        if vehicle['initial_site'] not in {plant['id'], *(site['id'] for site in scenario['recycling_facilities'])}:
            raise ValueError('Vehicles must start at a configured facility')
        maximum = vehicle['maximum_range']
        if maximum is not None:
            if type(maximum) not in (int, float) or not math.isfinite(maximum) or maximum <= 0:
                raise ValueError('Maximum range must be positive and finite, or null')
            numbers.append(maximum)
        if vehicle['speed'] <= 0 or type(vehicle['payload_capacity']) is not int or vehicle['payload_capacity'] <= 0:
            raise ValueError('Speed and integer payload capacity must be positive')
        numbers.append(vehicle['speed'])
    for product in scenario['products']:
        identifiers.append(product['id'])
        numbers.append(product['release_time'])
        if not 0 <= product['release_time'] <= scenario['horizon']:
            raise ValueError('Product releases must fall within the horizon')
        if product['origin'] != point['id']:
            raise ValueError('Products must originate at the configured point')
        if product['disposition'] == 'remanufacture':
            if not product['operations'] or any(op not in {s['operation_id'] for s in stations} for op in product['operations']):
                raise ValueError('Every operation must have a compatible workstation')
        elif product['disposition'] != 'recycle' or product['operations'] or not scenario['recycling_facilities']:
            raise ValueError('Recycling products require a recycler and an empty operation list')
    if not all(isinstance(i, str) and i for i in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError('Entity IDs must be unique nonempty strings')
    used = set()
    for trip in scenario['trips']:
        if trip['vehicle_id'] not in vehicles or trip['vehicle_id'] in used:
            raise ValueError('Allow at most one fixed trip per configured vehicle')
        used.add(trip['vehicle_id'])
        facilities = {plant['id'], *(r['id'] for r in scenario['recycling_facilities'])}
        if not trip['stops'] or trip['stops'][-1] not in facilities or trip['stops'].count(point['id']) > 1 or any(stop not in facilities | {point['id']} for stop in trip['stops']) or not 0 <= trip['departure_time'] <= scenario['horizon']:
            raise ValueError('Fixed trips may pick up once, may visit facilities, and must end at a facility')
        numbers.append(trip['departure_time'])
    if not all(type(value) in (int, float) and math.isfinite(value) for value in numbers) or scenario['horizon'] < 0:
        raise ValueError('Times, coordinates, durations and speeds must be finite')
    scenario['seed_provenance']
