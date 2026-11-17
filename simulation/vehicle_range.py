"""Distance allowance checks and actual travel accounting for fixed vehicle plans."""
from math import dist
from .models import Record, InventoryRunState, record_event

RANGE_TOLERANCE = 1e-9


def network_site(scenario: Record, site_id: str) -> Record:
    """Look up coordinates for the collection point or a configured facility."""
    return next(site for site in [scenario['collection_point'], *scenario.get('plants', [scenario['plant']]), *scenario['recycling_facilities']]
                if site['id'] == site_id)


def evaluate_fixed_trip_range(scenario: Record, state: InventoryRunState, trip: Record,
                              start_index: int = 0) -> Record:
    """Check every remaining facility-to-facility segment without changing live state."""
    vehicle_id = trip['vehicle_id']
    definition = next(vehicle for vehicle in scenario['vehicles'] if vehicle['id'] == vehicle_id)
    maximum = definition['maximum_range']
    if maximum is None:
        return {'feasible': True, 'segments': []}
    vehicle = state.vehicles[vehicle_id]
    position = vehicle['position']
    allowance = vehicle['remaining_range']
    segment_start = vehicle['location']
    facilities = {*(plant['id'] for plant in scenario.get('plants', [scenario['plant']])), *(site['id'] for site in scenario['recycling_facilities'])}
    required = 0.0
    segments = []
    for site_id in trip['stops'][start_index:]:
        site = network_site(scenario, site_id)
        required += dist(position, site['coordinates'])
        position = site['coordinates']
        if site_id in facilities:
            segments.append({'start': segment_start, 'end': site_id,
                             'required_distance': required, 'available_range': allowance})
            required = 0.0
            allowance = maximum
            segment_start = site_id
    failing = next((segment for segment in segments
                    if segment['required_distance'] > segment['available_range'] + RANGE_TOLERANCE), None)
    result = {'feasible': failing is None, 'segments': segments}
    if failing is not None:
        result['rejection'] = {'reason': 'range_infeasible', **failing}
    return result


def advance_vehicle_travel(scenario: Record, state: InventoryRunState, time: float) -> None:
    """Accumulate actual leg prefixes for position-tracked AVs, consuming finite allowance when configured."""
    for definition in scenario['vehicles']:
        vehicle = state.vehicles[definition['id']]
        if definition['maximum_range'] is None and 'position' not in vehicle:
            continue
        leg = vehicle['active_leg']
        if leg is None:
            continue
        travelled = min(leg['distance'], max(0.0, time - leg['departure_time']) * definition['speed'])
        increment = travelled - vehicle['leg_distance_accounted']
        if increment < -RANGE_TOLERANCE or (vehicle['remaining_range'] is not None and increment > vehicle['remaining_range'] + RANGE_TOLERANCE):
            raise ValueError(f"Vehicle {definition['id']} cannot travel beyond its remaining allowance")
        vehicle['distance'] += increment
        if vehicle['remaining_range'] is not None:
            vehicle['remaining_range'] = max(0.0, vehicle['remaining_range'] - increment)
        vehicle['leg_distance_accounted'] = travelled
        fraction = travelled / leg['distance'] if leg['distance'] else 1.0
        vehicle['position'] = [origin + fraction * (target - origin)
                               for origin, target in zip(leg['origin_coordinates'], leg['target_coordinates'])]


def reset_vehicle_range(scenario: Record, state: InventoryRunState, vehicle_id: str) -> None:
    """Reset a finite allowance on facility arrival, before delivery or waiting to unload."""
    definition = next(vehicle for vehicle in scenario['vehicles'] if vehicle['id'] == vehicle_id)
    if definition['maximum_range'] is None:
        return
    vehicle = state.vehicles[vehicle_id]
    facilities = {*(plant['id'] for plant in scenario.get('plants', [scenario['plant']])), *(site['id'] for site in scenario['recycling_facilities'])}
    if vehicle['location'] not in facilities:
        return
    before = vehicle['remaining_range']
    vehicle['remaining_range'] = definition['maximum_range']
    record_event(state, 'range_reset', vehicle_id=vehicle_id, facility_id=vehicle['location'],
                 range_before=before, range_after=vehicle['remaining_range'], load=list(vehicle['load']))
