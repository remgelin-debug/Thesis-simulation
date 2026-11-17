"""Dynamic route execution with actual mid-leg motion and stale-arrival invalidation."""
import copy
import heapq
from math import dist
from .models import Record, InventoryRunState, build_inventory_state, record_event
from .plant_network import configured_plants, initialize_plant_states, local_plant_view, settle_plant_network, check_plant_network
from .vehicle_range import reset_vehicle_range, advance_vehicle_travel
from .transport import accept_recycling_products
from .dynamic_routing import build_dynamic_policy_input, choose_dynamic_insertions


def build_dynamic_state(scenario: Record) -> InventoryRunState:
    """Initialize transport positions, route revisions and leg identities for all dynamic AVs."""
    state = build_inventory_state(scenario)
    sites = {site['id']: site for site in [*configured_plants(scenario), *scenario['recycling_facilities']]}
    for definition in scenario['vehicles']:
        state.vehicles[definition['id']].update(position=list(sites[definition['initial_site']]['coordinates']),
            remaining_range=definition['maximum_range'], leg_distance_accounted=0.0,
            remaining_stops=[], route_version=0, leg_serial=0)
    initialize_plant_states(scenario, state)
    return state


def advance_dynamic_motion(scenario: Record, state: InventoryRunState, time: float) -> None:
    """Account each active leg prefix exactly once, for finite and unlimited AVs."""
    advance_vehicle_travel(scenario, state, time)


def start_dynamic_leg(scenario: Record, state: InventoryRunState, vehicle_id: str) -> tuple[float, int, str, str, str] | None:
    """Start the current head stop only if no leg/wait is active; return its uniquely tagged arrival."""
    vehicle = state.vehicles[vehicle_id]
    if vehicle['active_leg'] is not None or vehicle['status'] == 'waiting_to_unload' or not vehicle['remaining_stops']:
        return None
    sites = {site['id']: site for site in [*configured_plants(scenario), *scenario['collection_points'], *scenario['recycling_facilities']]}
    target = sites[vehicle['remaining_stops'][0]['site_id']]
    definition = next(vehicle for vehicle in scenario['vehicles'] if vehicle['id'] == vehicle_id)
    distance = dist(vehicle['position'], target['coordinates'])
    if vehicle['remaining_range'] is not None and distance > vehicle['remaining_range'] + 1e-9:
        raise ValueError(f'Vehicle {vehicle_id} cannot depart on an infeasible leg')
    vehicle['leg_serial'] += 1
    leg = {'leg_id': f"{vehicle_id}:L{vehicle['leg_serial']}", 'vehicle_id': vehicle_id,
        'origin_coordinates': list(vehicle['position']), 'target': target['id'], 'target_coordinates': target['coordinates'],
        'departure_time': state.time, 'arrival_time': state.time + distance / definition['speed'],
        'distance': distance, 'load': list(vehicle['load']), 'remaining_range_at_departure': vehicle['remaining_range']}
    vehicle.update(active_leg=leg, status='travelling', leg_distance_accounted=0.0)
    record_event(state, 'leg_departure', **leg)
    return leg['arrival_time'], 2, vehicle_id, 'arrival', leg['leg_id']


def service_dynamic_stop(scenario: Record, state: InventoryRunState, vehicle_id: str) -> None:
    """Pick up owned products or begin final-destination facility service at an arrived stop."""
    vehicle = state.vehicles[vehicle_id]
    stop = vehicle['remaining_stops'][0]
    definitions = {product['id']: product for product in scenario['products']}
    for product_id in stop['pickups']:
        product = state.products[product_id]
        assert product['state'] == 'at_collection_point' and product['assigned_vehicle'] == vehicle_id
        assert definitions[product_id]['origin'] == stop['site_id']
        product.update(state='onboard_av', location=vehicle_id)
        product['milestones']['pickup'] = state.time
        vehicle['load'].append(product_id)
        record_event(state, 'pickup', product_id=product_id, vehicle_id=vehicle_id, collection_point=stop['site_id'])
    if stop['deliveries']:
        if stop['site_id'] in {plant['id'] for plant in configured_plants(scenario)}:
            _, view = local_plant_view(scenario, state, stop['site_id'])
            view.waiting_vehicles[vehicle_id] = state.time
        else:
            accept_recycling_products(scenario, state, vehicle_id)


def finish_dynamic_stops(scenario: Record, state: InventoryRunState) -> list[str]:
    """Pop fully served arrived stops; report facility availability without disturbing onboard commitments."""
    available = []
    facilities = {*(plant['id'] for plant in configured_plants(scenario)), *(site['id'] for site in scenario['recycling_facilities'])}
    for vehicle_id, vehicle in sorted(state.vehicles.items()):
        if vehicle['active_leg'] is not None or vehicle['status'] == 'waiting_to_unload' or not vehicle['remaining_stops']:
            continue
        stop = vehicle['remaining_stops'][0]
        if vehicle['location'] != stop['site_id']:
            continue
        if any(product_id in vehicle['load'] for product_id in stop['deliveries']):
            continue
        vehicle['remaining_stops'].pop(0)
        record_event(state, 'stop_completed', vehicle_id=vehicle_id, stop=copy.deepcopy(stop))
        if stop['site_id'] in facilities:
            record_event(state, 'vehicle_available', vehicle_id=vehicle_id, reason='facility_stop')
            available.append(vehicle_id)
    return available


def dispatch_dynamic_requests(scenario: Record, state: InventoryRunState, trigger: Record) -> None:
    """Record the chosen policy decision and atomically apply assignments, cancellations and remaining routes."""
    record_event(state, 'dispatch', trigger=trigger, decision_index=len(state.decisions) + 1)
    boundary = build_dynamic_policy_input(scenario, state)
    proposals, audits = choose_dynamic_insertions(boundary)
    state.decisions.append({'time': state.time, 'trigger': copy.deepcopy(trigger), 'last_event_sequence': len(state.events),
        'policy_input': boundary, 'requests': audits, 'proposals': proposals})
    for proposal in proposals:
        vehicle_id, product_id = proposal['vehicle_id'], proposal['product_id']
        vehicle = state.vehicles[vehicle_id]
        state.products[product_id].update(assigned_vehicle=vehicle_id, planned_destination=proposal['destination'])
        record_event(state, 'assignment', product_id=product_id, vehicle_id=vehicle_id, destination=proposal['destination'])
        leg = vehicle['active_leg']
        if leg is not None and proposal['new_remaining_stops'][0]['site_id'] != leg['target']:
            record_event(state, 'leg_cancelled', vehicle_id=vehicle_id, leg_id=leg['leg_id'],
                old_arrival_time=leg['arrival_time'], distance_travelled=vehicle['leg_distance_accounted'],
                position=list(vehicle['position']), remaining_range=vehicle['remaining_range'])
            vehicle.update(active_leg=None, status='idle', leg_distance_accounted=0.0)
        old = copy.deepcopy(vehicle['remaining_stops'])
        vehicle['remaining_stops'] = copy.deepcopy(proposal['new_remaining_stops'])
        vehicle['route_version'] += 1
        record_event(state, 'route_revised', vehicle_id=vehicle_id, product_id=product_id,
            route_version=vehicle['route_version'], position=list(vehicle['position']), remaining_range=vehicle['remaining_range'],
            old_remaining_stops=old, new_remaining_stops=copy.deepcopy(vehicle['remaining_stops']))


def run_dynamic_simulation(scenario: Record, *, initial_state: InventoryRunState | None = None,
                           dispatch_enabled: bool = True, projected_deliveries: list[Record] | None = None) -> InventoryRunState:
    """Execute dynamic insertion with timestamp batching, shared plant rules and stale-arrival rejection."""
    state = build_dynamic_state(scenario) if initial_state is None else initial_state
    products = {product['id']: product for product in scenario['products']}
    pending = []
    if initial_state is None:
        pending.extend((product['release_time'], 1, product['id'], 'release', '') for product in scenario['products'])
        pending.extend((0, 2, vehicle['id'], 'available', '') for vehicle in scenario['vehicles'])
    else:
        for vehicle_id, vehicle in state.vehicles.items():
            if vehicle['active_leg'] is not None:
                leg = vehicle['active_leg']
                pending.append((leg['arrival_time'], 2, vehicle_id, 'arrival', leg['leg_id']))
            else:
                arrival = start_dynamic_leg(scenario, state, vehicle_id)
                if arrival is not None:
                    pending.append(arrival)
        for plant in configured_plants(scenario):
            _, view = local_plant_view(scenario, state, plant['id'])
            for station_id, operation in view.workstations.items():
                if operation is not None:
                    pending.append((operation['end_time'], 0, plant['id'], 'complete', station_id) if state.plant_states else
                                   (operation['end_time'], 0, station_id, 'complete', ''))
        pending.append((state.time, 0, '', 'window', ''))
    pending.extend((opening, 0, plant['id'], 'window', '') for plant in configured_plants(scenario)
                   for station in plant['workstations'] for opening, _ in station['windows'] if opening >= state.time)
    heapq.heapify(pending)
    while pending:
        # Discard superseded arrivals BEFORE advancing the physical clock.
        while pending and pending[0][3] == 'arrival':
            item = pending[0]
            leg = state.vehicles[item[2]]['active_leg']
            if leg is not None and leg['leg_id'] == item[4]:
                break
            heapq.heappop(pending)
        if not pending or pending[0][0] > scenario['horizon']:
            break
        time = pending[0][0]
        advance_dynamic_motion(scenario, state, time)
        state.time = time
        releases, available = set(), set()
        while True:
            due = []
            while pending and pending[0][0] == time:
                due.append(heapq.heappop(pending))
            for _, _, identifier, kind, token in due:
                if kind == 'release':
                    definition = products[identifier]
                    state.products[identifier].update(state='at_collection_point', location=definition['origin'])
                    state.products[identifier]['milestones']['release'] = time
                    record_event(state, 'return_release', product=copy.deepcopy(definition))
                    releases.add(identifier)
                elif kind == 'available':
                    record_event(state, 'vehicle_available', vehicle_id=identifier, reason='initial')
                    available.add(identifier)
                elif kind == 'complete':
                    from .plant import finish_inventory_operation
                    local, view = local_plant_view(scenario, state, identifier if state.plant_states else scenario['plant']['id'])
                    finish_inventory_operation(local, view, token if state.plant_states else identifier)
                elif kind == 'arrival':
                    vehicle = state.vehicles[identifier]
                    leg = vehicle['active_leg']
                    if leg is None or leg['leg_id'] != token:
                        continue
                    details = {**leg, 'remaining_range_on_arrival': vehicle['remaining_range']}
                    vehicle.update(location=leg['target'], position=list(leg['target_coordinates']), active_leg=None,
                        status='idle', leg_distance_accounted=0.0)
                    record_event(state, 'leg_arrival', **details)
                    reset_vehicle_range(scenario, state, identifier)
                    if projected_deliveries is not None:
                        for product_id in vehicle['remaining_stops'][0]['deliveries']:
                            projected_deliveries.append({'product_id': product_id, 'vehicle_id': identifier, 'destination': leg['target'], 'arrival_time': time})
                    service_dynamic_stop(scenario, state, identifier)
            for completion in settle_plant_network(scenario, state):
                heapq.heappush(pending, completion)
            available.update(finish_dynamic_stops(scenario, state))
            for vehicle_id in sorted(state.vehicles):
                arrival = start_dynamic_leg(scenario, state, vehicle_id)
                if arrival is not None:
                    heapq.heappush(pending, arrival)
            if pending and pending[0][0] == time:
                continue
            if dispatch_enabled and (releases or available):
                trigger = {'released_products': sorted(releases), 'available_vehicles': sorted(available)}
                dispatch_dynamic_requests(scenario, state, trigger)
                releases.clear()
                available.clear()
                for vehicle_id in sorted(state.vehicles):
                    arrival = start_dynamic_leg(scenario, state, vehicle_id)
                    if arrival is not None:
                        heapq.heappush(pending, arrival)
                if pending and pending[0][0] == time:
                    continue
            releases.clear()
            available.clear()
            break
        check_plant_network(scenario, state)
    advance_dynamic_motion(scenario, state, scenario['horizon'])
    state.time = scenario['horizon']
    for plant in configured_plants(scenario):
        _, view = local_plant_view(scenario, state, plant['id'])
        for product_id, entered in view.inventory.items():
            state.products[product_id]['inventory_wait'] += state.time - entered
    for vehicle in state.vehicles.values():
        if vehicle['waiting_since'] is not None:
            vehicle['waiting_to_unload'] += state.time - vehicle['waiting_since']
    containers = state.plant_states.values() if state.plant_states else [{'inventory': state.inventory, 'held_slots': state.held_slots}]
    counts = list(containers)
    record_event(state, 'horizon', waiting_count=sum(len(plant['inventory']) for plant in counts),
                 held_count=sum(len(plant['held_slots']) for plant in counts))
    return state
