"""Read-only plant projection and transparent selection for the fixed quality-aware policy."""
import copy
from dataclasses import asdict
import heapq
from math import dist
from .models import Record, InventoryRunState
from .plant import finish_inventory_operation, start_inventory_operation
from .transport import retry_inventory_admissions
from .vehicle_range import network_site


def build_forecast_context(scenario: Record, state: InventoryRunState) -> Record:
    """Copy only released physical inputs and current state across the aware policy boundary."""
    released = {product_id for product_id, product in state.products.items() if product['state'] != 'unreleased'}
    snapshot = asdict(state)
    del snapshot['events'], snapshot['decisions']
    if not snapshot.get('plant_states'):
        snapshot.pop('plant_states', None)
    snapshot['products'] = {product_id: product for product_id, product in snapshot['products'].items() if product_id in released}
    physical = {key: copy.deepcopy(scenario[key]) for key in
                ('horizon', 'plant', 'collection_point', 'recycling_facilities', 'vehicles', 'trips')}
    physical['products'] = [{key: copy.deepcopy(product[key]) for key in
        ('id', 'origin', 'release_time', 'disposition', 'operations')} for product in scenario['products'] if product['id'] in released]
    progress = {vehicle['id']: sum(event['event_type'] == 'leg_departure' and event.get('vehicle_id') == vehicle['id']
                                 for event in state.events) for vehicle in scenario['vehicles']}
    return {'scenario': physical, 'state': snapshot, 'departed_legs': progress}


def predicted_delivery_time(scenario: Record, state: InventoryRunState, trip: Record, departed_legs: int = 0) -> float:
    """Calculate fixed-stop final arrival from an active leg or the vehicle's current site."""
    vehicle = state.vehicles[trip['vehicle_id']]
    speed = next(vehicle['speed'] for vehicle in scenario['vehicles'] if vehicle['id'] == trip['vehicle_id'])
    if vehicle['active_leg'] is not None:
        time = vehicle['active_leg']['arrival_time']
        position = vehicle['active_leg']['target_coordinates']
    else:
        time = state.time
        position = network_site(scenario, vehicle['location'])['coordinates']
    for site_id in trip['stops'][departed_legs:]:
        target = network_site(scenario, site_id)['coordinates']
        time += dist(position, target) / speed
        position = target
    return time


def forecast_candidate(context: Record, trip: Record, product_id: str, selected: list[str]) -> Record:
    """Project admission/first start with prior selected items and committed loads, without live changes."""
    scenario = context['scenario']
    state = InventoryRunState(**copy.deepcopy(context['state']))
    arrival = predicted_delivery_time(scenario, state, trip)
    result = {'predicted_arrival': arrival, 'predicted_admission': None, 'predicted_first_start': None,
              'predicted_inventory_wait': None}
    if trip['stops'][-1] != scenario['plant']['id']:
        if arrival <= scenario['horizon']:
            result.update(predicted_admission=arrival)
        return result
    deliveries: dict[str, tuple[float, list[str]]] = {}
    for committed in scenario['trips']:
        vehicle_id = committed['vehicle_id']
        if vehicle_id == trip['vehicle_id'] or committed['stops'][-1] != scenario['plant']['id'] or vehicle_id in state.waiting_vehicles:
            continue
        carried = sorted(candidate for candidate, product in state.products.items()
            if product['assigned_vehicle'] == vehicle_id and product['state'] in ('at_collection_point', 'onboard_av'))
        if carried:
            time = predicted_delivery_time(scenario, state, committed, context['departed_legs'][vehicle_id])
            deliveries[vehicle_id] = time, carried
    deliveries[trip['vehicle_id']] = arrival, sorted([*selected, product_id])
    pending = [(operation['end_time'], 0, station_id, 'complete') for station_id, operation in state.workstations.items() if operation is not None]
    pending.extend((opening, 1, station['id'], 'window') for station in scenario['plant']['workstations']
                   for opening, _ in station['windows'] if opening >= state.time)
    pending.extend((time, 2, vehicle_id, 'delivery') for vehicle_id, (time, _) in deliveries.items())
    pending.append((state.time, 1, '', 'settle'))
    heapq.heapify(pending)
    while pending and pending[0][0] <= scenario['horizon']:
        state.time = pending[0][0]
        batch = []
        while pending and pending[0][0] == state.time:
            batch.append(heapq.heappop(pending))
        for _, _, identifier, action in batch:
            if action == 'complete':
                finish_inventory_operation(scenario, state, identifier)
            elif action == 'delivery':
                load = deliveries[identifier][1]
                state.vehicles[identifier].update(location=scenario['plant']['id'], status='idle', load=load,
                    active_leg=None, waiting_since=None, wait_logged=False)
                state.waiting_vehicles[identifier] = state.time
                for carried in load:
                    state.products[carried].update(state='onboard_av', location=identifier,
                        assigned_vehicle=identifier, planned_destination=scenario['plant']['id'])
        while True:
            admitted = retry_inventory_admissions(scenario, state)
            started = False
            for station in sorted(scenario['plant']['workstations'], key=lambda station: station['id']):
                end = start_inventory_operation(scenario, state, station)
                if end is not None:
                    heapq.heappush(pending, (end, 0, station['id'], 'complete'))
                    started = True
                    operation = state.workstations[station['id']]
                    if operation['product_id'] == product_id:
                        admission = state.products[product_id]['milestones']['admission']
                        result.update(predicted_admission=admission, predicted_first_start=state.time,
                                      predicted_inventory_wait=state.time - admission)
                        return result
                    break
            if not admitted and not started:
                break
    result['predicted_admission'] = state.products[product_id]['milestones'].get('admission')
    return result


def quality_candidate_score(candidate: Record) -> list:
    """Rank horizon-feasible starts, admission wait and fixed-trip distance, then stable IDs/stops."""
    starts = candidate['predicted_first_start'] is not None
    return [0 if starts else 1, candidate['predicted_inventory_wait'] if starts else 0.0,
            candidate['added_distance'], candidate['product_id'], candidate['pickup_stop_index'], candidate['delivery_stop_index']]


def choose_quality_aware_products(policy_input: Record) -> tuple[list[str], list[Record], str]:
    """Choose each payload slot by independent candidate projections using only the recorded boundary input."""
    context = policy_input['forecast_context']
    scenario = context['scenario']
    trip = next(trip for trip in scenario['trips'] if trip['vehicle_id'] == policy_input['vehicle_id'])
    vehicle = next(vehicle for vehicle in scenario['vehicles'] if vehicle['id'] == trip['vehicle_id'])
    disposition = 'remanufacture' if trip['stops'][-1] == scenario['plant']['id'] else 'recycle'
    remaining = [request['product_id'] for request in policy_input['requests'] if request['disposition'] == disposition]
    selected: list[str] = []
    audit: list[Record] = []
    if scenario['collection_point']['id'] not in trip['stops']:
        return selected, audit, 'Fixed trip has no pickup stop'
    for selection_round in range(min(vehicle['payload_capacity'], len(remaining))):
        candidates = []
        for product_id in remaining:
            candidate = {'selection_round': selection_round, 'product_id': product_id, 'destination': trip['stops'][-1],
                'pickup_stop_index': trip['stops'].index(scenario['collection_point']['id']),
                'delivery_stop_index': len(trip['stops']) - 1, 'added_distance': 0.0,
                **forecast_candidate(context, trip, product_id, selected)}
            candidate['score'] = quality_candidate_score(candidate)
            candidates.append(candidate)
        winner = min(candidates, key=lambda candidate: candidate['score'])
        for candidate in candidates:
            candidate['selected'] = candidate['product_id'] == winner['product_id']
        audit.extend(candidates)
        selected.append(winner['product_id'])
        remaining.remove(winner['product_id'])
    return selected, audit, 'Prefer start within horizon, then admission-to-first-start wait, added distance, product ID and stop indexes'
