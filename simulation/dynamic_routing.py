"""Quality-free pickup/delivery insertion over complete remaining vehicle routes."""
import copy
import json
from math import dist
from .models import Record, InventoryRunState
from .plant_network import configured_plants
from .dynamic_forecasting import build_dynamic_forecast_context, forecast_dynamic_insertion, dynamic_quality_score


def merge_route_stops(stops: list[Record]) -> list[Record]:
    """Copy and coalesce adjacent visits to one site, preserving all committed service IDs."""
    merged: list[Record] = []
    for stop in stops:
        if merged and merged[-1]['site_id'] == stop['site_id']:
            for field in ('pickups', 'deliveries'):
                merged[-1][field] = sorted(set(merged[-1][field]) | set(stop[field]))
        else:
            merged.append(copy.deepcopy(stop))
    return merged


def remaining_route_distance(position: list[float], stops: list[Record], sites: dict[str, Record]) -> float:
    """Calculate full remaining distance from the actual position without changing the plan."""
    total = 0.0
    for stop in stops:
        target = sites[stop['site_id']]['coordinates']
        total += dist(position, target)
        position = target
    return total


def check_insertion_route(vehicle: Record, stops: list[Record], requests: dict[str, Record],
                          sites: dict[str, Record], candidate_id: str, destination: str) -> tuple[bool, str]:
    """Check ownership, precedence, payload, commitments, every range leg and facility escape."""
    if vehicle['status'] == 'waiting_to_unload':
        return False, 'vehicle_waiting_to_unload'
    load = set(vehicle['load'])
    picked = set(load)
    delivered: set[str] = set()
    expected = {product_id for product_id, request in requests.items()
                if request['assigned_vehicle'] == vehicle['id']} | {candidate_id}
    position, allowance = vehicle['position'], vehicle['remaining_range']
    for stop in stops:
        site = sites[stop['site_id']]
        distance = dist(position, site['coordinates'])
        if allowance is not None:
            if distance > allowance + 1e-9:
                return False, 'range_exceeded_before_facility'
            allowance -= distance
        position = site['coordinates']
        if site['kind'] != 'collection' and allowance is not None:
            allowance = vehicle['maximum_range']
        for product_id in stop['pickups']:
            request = requests[product_id]
            if request['origin'] != site['id'] or site['kind'] != 'collection' or product_id in picked:
                return False, 'invalid_or_duplicate_pickup'
            if request['assigned_vehicle'] not in (None, vehicle['id']):
                return False, 'pickup_owned_by_another_vehicle'
            load.add(product_id)
            picked.add(product_id)
            if len(load) > vehicle['payload_capacity']:
                return False, 'payload_exceeded'
        for product_id in stop['deliveries']:
            request = requests[product_id]
            required = destination if product_id == candidate_id else request['planned_destination']
            kind = 'plant' if request['disposition'] == 'remanufacture' else 'recycler'
            if product_id not in load or product_id in delivered or required != site['id'] or site['kind'] != kind:
                return False, 'precedence_or_destination_violation'
            load.remove(product_id)
            delivered.add(product_id)
    if load or delivered != expected:
        return False, 'unserved_commitment'
    if not stops or sites[stops[-1]['site_id']]['kind'] == 'collection':
        return False, 'missing_facility_escape'
    return True, 'feasible'


def enumerate_dynamic_candidates(policy_input: Record, request: Record) -> list[Record]:
    """Enumerate insertion positions and up to two explicit reset visits, deduplicating complete routes."""
    sites = {site['id']: site for site in policy_input['sites']}
    requests = {item['product_id']: item for item in policy_input['requests']}
    destinations = sorted(site_id for site_id, site in sites.items()
        if site['kind'] == ('plant' if request['disposition'] == 'remanufacture' else 'recycler'))
    facilities = sorted(site_id for site_id, site in sites.items() if site['kind'] != 'collection')
    candidates: list[Record] = []
    seen = set()
    for vehicle in sorted(policy_input['vehicles'], key=lambda vehicle: vehicle['id']):
        if vehicle['status'] == 'waiting_to_unload':
            continue
        old = vehicle['remaining_stops']
        baseline = remaining_route_distance(vehicle['position'], old, sites)
        for destination in destinations:
            for pickup_index in range(len(old) + 1):
                pickup = {'site_id': request['origin'], 'pickups': [request['product_id']], 'deliveries': []}
                with_pickup = [*copy.deepcopy(old[:pickup_index]), pickup, *copy.deepcopy(old[pickup_index:])]
                for delivery_index in range(pickup_index + 1, len(with_pickup) + 1):
                    delivery = {'site_id': destination, 'pickups': [], 'deliveries': [request['product_id']]}
                    base = [*copy.deepcopy(with_pickup[:delivery_index]), delivery, *copy.deepcopy(with_pickup[delivery_index:])]
                    variants = [(base, 0)]
                    if vehicle['maximum_range'] is not None:
                        frontier = [base]
                        for visit_count in (1, 2):
                            following = []
                            for route in frontier:
                                for index in range(len(route) + 1):
                                    for facility in facilities:
                                        visit = {'site_id': facility, 'pickups': [], 'deliveries': []}
                                        variant = [*copy.deepcopy(route[:index]), visit, *copy.deepcopy(route[index:])]
                                        following.append(variant)
                                        variants.append((variant, visit_count))
                            frontier = following
                    for variant, visit_count in variants:
                        route = merge_route_stops(variant)
                        key = json.dumps(route, sort_keys=True, separators=(',', ':'))
                        identity = vehicle['id'], destination, key
                        if identity in seen:
                            continue
                        seen.add(identity)
                        feasible, reason = check_insertion_route(vehicle, route, requests, sites, request['product_id'], destination)
                        distance = remaining_route_distance(vehicle['position'], route, sites)
                        added = max(0.0, distance - baseline)
                        candidates.append({'candidate_id': f'C{len(candidates) + 1:05d}',
                            'product_id': request['product_id'], 'vehicle_id': vehicle['id'], 'destination': destination,
                            'pickup_index': pickup_index, 'delivery_index': delivery_index, 'extra_visits': visit_count,
                            'old_remaining_stops': copy.deepcopy(old), 'new_remaining_stops': route,
                            'position': list(vehicle['position']), 'remaining_range': vehicle['remaining_range'],
                            'feasible': feasible, 'reason': reason, 'remaining_distance': distance, 'added_distance': added,
                            'score': [round(added, 12), vehicle['id'], destination, pickup_index, delivery_index, visit_count, key]})
    return candidates


def choose_dynamic_insertions(policy_input: Record) -> tuple[list[Record], list[Record]]:
    """Process outstanding requests by release/ID and rank feasible complete insertions using only boundary data."""
    working = copy.deepcopy(policy_input)
    proposals, audits = [], []
    outstanding = sorted((request for request in working['requests'] if request['state'] == 'at_collection_point'
                          and request['assigned_vehicle'] is None), key=lambda request: (request['release_time'], request['product_id']))
    for request in outstanding:
        candidates = enumerate_dynamic_candidates(working, request)
        if working.get('policy') == 'dynamic_quality_aware':
            for candidate in candidates:
                if candidate['feasible']:
                    candidate.update(forecast_dynamic_insertion(working, candidate))
                    candidate['score'] = dynamic_quality_score(candidate)
        feasible = [candidate for candidate in candidates if candidate['feasible']]
        winner = min(feasible, key=lambda candidate: candidate['score']) if feasible else None
        audits.append({'product_id': request['product_id'], 'candidates': candidates,
            'selected_candidate_id': winner['candidate_id'] if winner else None,
            'reason': 'minimum_added_distance_then_stable_ids_and_positions' if winner else 'no_feasible_insertion'})
        if working.get('policy') == 'dynamic_quality_aware' and winner is not None:
            audits[-1]['reason'] = 'start_within_horizon_then_admission_wait_then_added_distance_then_stable_ties'
        if winner is not None:
            proposals.append(copy.deepcopy(winner))
            vehicle = next(vehicle for vehicle in working['vehicles'] if vehicle['id'] == winner['vehicle_id'])
            vehicle['remaining_stops'] = copy.deepcopy(winner['new_remaining_stops'])
            request['assigned_vehicle'], request['planned_destination'] = vehicle['id'], winner['destination']
    return proposals, audits


def build_dynamic_policy_input(scenario: Record, state: InventoryRunState) -> Record:
    """Expose released transport facts, adding copied physical forecasts only for the aware policy."""
    sites = [{**copy.deepcopy(site), 'kind': kind} for kind, definitions in
        (('plant', configured_plants(scenario)), ('recycler', scenario['recycling_facilities']), ('collection', scenario['collection_points']))
        for site in definitions]
    sites = [{'id': site['id'], 'coordinates': site['coordinates'], 'kind': site['kind']} for site in sites]
    definitions = {product['id']: product for product in scenario['products']}
    requests = [{'product_id': product_id, 'origin': definitions[product_id]['origin'],
        'release_time': definitions[product_id]['release_time'], 'disposition': definitions[product_id]['disposition'],
        'state': product['state'], 'assigned_vehicle': product['assigned_vehicle'],
        'planned_destination': product['planned_destination']} for product_id, product in state.products.items()
        if product['state'] in ('at_collection_point', 'onboard_av')]
    vehicles = [{**copy.deepcopy(state.vehicles[definition['id']]), 'id': definition['id'],
        'speed': definition['speed'], 'payload_capacity': definition['payload_capacity'], 'maximum_range': definition['maximum_range']}
        for definition in scenario['vehicles']]
    boundary = {'time': state.time, 'sites': sites, 'requests': requests, 'vehicles': vehicles,
                'insertion_rules': {'max_extra_facility_visits': 2, 'distance_score_decimals': 12}}
    if scenario['policy'] == 'dynamic_quality_aware':
        boundary['policy'] = scenario['policy']
        boundary['forecast_context'] = build_dynamic_forecast_context(scenario, state)
        for request in requests:
            request['operations'] = list(definitions[request['product_id']]['operations'])
    return boundary
