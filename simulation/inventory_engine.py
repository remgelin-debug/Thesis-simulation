"""Event batching for fixed trips, shared inventory and optional recycling."""
import heapq
from .models import Record, InventoryRunState, build_inventory_state, record_event
from .plant import finish_inventory_operation, start_inventory_operation, inventory_occupancy
from .routing import dispatch_inventory_trip
from .vehicle_range import advance_vehicle_travel, network_site
from .transport import (depart_inventory_leg, arrive_inventory_leg, collect_inventory_products,
                        retry_inventory_admissions, accept_recycling_products)


def settle_inventory(scenario: Record, state: InventoryRunState) -> list[tuple[float, int, str, str]]:
    """Retry admissions and start one eligible station at a time until no immediate action remains."""
    completions = []
    while True:
        admitted = retry_inventory_admissions(scenario, state)
        started = False
        for station in sorted(scenario['plant']['workstations'], key=lambda station: station['id']):
            end = start_inventory_operation(scenario, state, station)
            if end is not None:
                completions.append((end, 0, station['id'], 'complete'))
                started = True
                break  # Newly freed capacity goes to waiting AVs before another start.
        if not admitted and not started:
            return completions


def check_inventory_state(scenario: Record, state: InventoryRunState) -> None:
    """Assert product conservation, payload and finite occupancy after settling."""
    capacity = scenario['plant']['inventory_capacity']
    assert capacity is None or inventory_occupancy(state) <= capacity
    for product_id, product in state.products.items():
        physical = int(product['state'] == 'at_collection_point') + int(product_id in state.inventory)
        physical += sum(product_id in v['load'] for v in state.vehicles.values())
        physical += sum(op is not None and op['product_id'] == product_id for op in state.workstations.values())
        physical += int(product['state'] in ('remanufactured', 'recycled'))
        assert physical == int(product['state'] != 'unreleased')
    for definition in scenario['vehicles']:
        assert len(state.vehicles[definition['id']]['load']) <= definition['payload_capacity']


def run_inventory_simulation(scenario: Record) -> InventoryRunState:
    """Execute fixed stops, actual finite-range motion, inventory settling and terminal acceptance."""
    state = build_inventory_state(scenario)
    products = {p['id']: p for p in scenario['products']}
    trips = {t['vehicle_id']: t for t in scenario['trips']}
    stop_indices = {vehicle['id']: 0 for vehicle in scenario['vehicles']}
    pending = [(p['release_time'], 1, p['id'], 'release') for p in scenario['products']]
    pending.extend((t['departure_time'], 3, t['vehicle_id'], 'dispatch') for t in scenario['trips'])
    pending.extend((opening, 0, s['id'], 'window') for s in scenario['plant']['workstations'] for opening, _ in s['windows'])
    heapq.heapify(pending)
    while pending and pending[0][0] <= scenario['horizon']:
        advance_vehicle_travel(scenario, state, pending[0][0])
        state.time = pending[0][0]
        batch = []
        while pending and pending[0][0] == state.time:
            batch.append(heapq.heappop(pending))
        departures = []
        for _, _, identifier, action in batch:
            if action == 'dispatch':
                departures.append(identifier)
            elif action == 'complete':
                finish_inventory_operation(scenario, state, identifier)
            elif action == 'release':
                state.products[identifier].update(state='at_collection_point', location=scenario['collection_point']['id'])
                state.products[identifier]['milestones']['release'] = state.time
                record_event(state, 'return_release', product=dict(products[identifier]))
            elif action == 'arrival':
                arrive_inventory_leg(scenario, state, identifier)
                trip = trips[identifier]
                stop_indices[identifier] += 1
                final_stop = stop_indices[identifier] == len(trip['stops'])
                if state.vehicles[identifier]['location'] == scenario['collection_point']['id']:
                    collect_inventory_products(scenario, state, identifier)
                if final_stop:
                    if state.vehicles[identifier]['location'] != scenario['plant']['id']:
                        accept_recycling_products(scenario, state, identifier)
                    elif state.vehicles[identifier]['load']:
                        state.waiting_vehicles[identifier] = state.time
                else:
                    target = network_site(scenario, trip['stops'][stop_indices[identifier]])
                    origin = network_site(scenario, state.vehicles[identifier]['location'])
                    arrival = depart_inventory_leg(scenario, state, identifier, origin, target, stop_indices[identifier])
                    heapq.heappush(pending, (arrival, 2, identifier, 'arrival'))
        for item in settle_inventory(scenario, state):
            heapq.heappush(pending, item)
        for vehicle_id in departures:
            if dispatch_inventory_trip(scenario, state, trips[vehicle_id]):
                origin = network_site(scenario, state.vehicles[vehicle_id]['location'])
                target = network_site(scenario, trips[vehicle_id]['stops'][0])
                arrival = depart_inventory_leg(scenario, state, vehicle_id, origin, target)
                heapq.heappush(pending, (arrival, 2, vehicle_id, 'arrival'))
        check_inventory_state(scenario, state)
    advance_vehicle_travel(scenario, state, scenario['horizon'])
    state.time = scenario['horizon']
    for product_id, entered in state.inventory.items():
        state.products[product_id]['inventory_wait'] += state.time - entered
    for definition in scenario['vehicles']:
        vehicle = state.vehicles[definition['id']]
        if definition['maximum_range'] is None and vehicle['active_leg'] is not None:
            vehicle['distance'] += (state.time - vehicle['active_leg']['departure_time']) * definition['speed']
        if vehicle['waiting_since'] is not None:
            vehicle['waiting_to_unload'] += state.time - vehicle['waiting_since']
    record_event(state, 'horizon', waiting_count=len(state.inventory), held_count=len(state.held_slots))
    return state


def inventory_summary(state: InventoryRunState) -> Record:
    """Derive counts and actual travel/wait totals from settled version-two state."""
    generated = sum(p['state'] != 'unreleased' for p in state.products.values())
    completed = sum(p['state'] == 'remanufactured' for p in state.products.values())
    recycled = sum(p['state'] == 'recycled' for p in state.products.values())
    return {'generated': generated, 'remanufactured': completed, 'recycled': recycled, 'unfinished': generated - completed - recycled,
        'vehicle_distance': sum(v['distance'] for v in state.vehicles.values()),
        'plant_inventory_wait': sum(p['inventory_wait'] for p in state.products.values()),
        'waiting_to_unload': sum(v['waiting_to_unload'] for v in state.vehicles.values())}
