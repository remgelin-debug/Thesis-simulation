"""Physical movement, pickup and shared-inventory admission for fixed trips."""
from math import dist
from .vehicle_range import evaluate_fixed_trip_range, reset_vehicle_range
from .plant import inventory_occupancy
from .models import Record, SimulationState, InventoryRunState, record_event


def depart_leg(scenario: Record, state: SimulationState, origin: Record, target: Record) -> float:
    """Set the vehicle travelling and return its calculated arrival time."""
    distance = dist(origin["coordinates"], target["coordinates"])
    arrival = state.time + distance / scenario["vehicle"]["speed"]
    leg = {"vehicle_id": scenario["vehicle"]["id"], "origin": origin["id"],
           "target": target["id"], "origin_coordinates": origin["coordinates"],
           "target_coordinates": target["coordinates"], "departure_time": state.time,
           "arrival_time": arrival, "distance": distance, "load": list(state.vehicle_load)}
    state.active_leg = leg
    state.vehicle_status = "travelling"
    state.vehicle_location = origin["id"]
    record_event(state, "leg_departure", **leg)
    return arrival


def arrive_leg(scenario: Record, state: SimulationState) -> None:
    """Finish the active leg, accumulating distance and updating vehicle location."""
    leg = state.active_leg
    assert leg is not None and state.time == leg["arrival_time"]
    state.vehicle_distance += leg["distance"]
    state.vehicle_location = leg["target"]
    state.vehicle_status = "idle"
    state.active_leg = None
    record_event(state, "leg_arrival", **leg)


def pick_up_product(scenario: Record, state: SimulationState) -> None:
    """Move the released product from collection to the empty assigned vehicle."""
    product = scenario["product"]
    assert state.product_state == "at_collection_point" and not state.vehicle_load
    assert state.vehicle_location == product["origin"] and state.assigned_vehicle == scenario["vehicle"]["id"]
    state.product_state = "onboard_av"
    state.product_location = scenario["vehicle"]["id"]
    state.vehicle_load.append(product["id"])
    state.milestones["pickup"] = state.time
    record_event(state, "pickup", product_id=product["id"], vehicle_id=scenario["vehicle"]["id"],
                 collection_point=product["origin"])


def admit_product(scenario: Record, state: SimulationState) -> None:
    """Unload into unlimited shared plant inventory and record admission."""
    product_id = scenario["product"]["id"]
    assert state.vehicle_location == scenario["plant"]["id"] and product_id in state.vehicle_load
    state.vehicle_load.remove(product_id)
    state.inventory.append(product_id)
    state.inventory_entered_at = state.time
    state.product_state = "in_plant_inventory"
    state.product_location = scenario["plant"]["id"]
    state.actual_destination = scenario["plant"]["id"]
    state.milestones["admission"] = state.time
    record_event(state, "plant_admission", product_id=product_id, plant_id=state.product_location,
                 vehicle_id=scenario["vehicle"]["id"])


def depart_inventory_leg(scenario: Record, state: 'InventoryRunState', vehicle_id: str, origin: Record, target: Record,
                         stop_index: int = 0) -> float:
    """Recheck remaining range, begin one fixed leg and return its calculated arrival time."""
    vehicle = state.vehicles[vehicle_id]
    definition = next(v for v in scenario['vehicles'] if v['id'] == vehicle_id)
    distance = dist(origin['coordinates'], target['coordinates'])
    leg = {'vehicle_id': vehicle_id, 'origin': origin['id'], 'target': target['id'],
        'origin_coordinates': origin['coordinates'], 'target_coordinates': target['coordinates'],
        'departure_time': state.time, 'arrival_time': state.time + distance / definition['speed'],
        'distance': distance, 'load': list(vehicle['load'])}
    if definition['maximum_range'] is not None:
        trip = next(trip for trip in scenario['trips'] if trip['vehicle_id'] == vehicle_id)
        suffix_check = evaluate_fixed_trip_range(scenario, state, trip, stop_index)
        if not suffix_check['feasible']:
            raise ValueError(f"Vehicle {vehicle_id}: {suffix_check['rejection']}")
        leg['remaining_range_at_departure'] = vehicle['remaining_range']
        vehicle['leg_distance_accounted'] = 0.0
    vehicle.update(active_leg=leg, status='travelling')
    record_event(state, 'leg_departure', **leg)
    return leg['arrival_time']


def arrive_inventory_leg(scenario: Record, state: 'InventoryRunState', vehicle_id: str) -> None:
    """Finish an active leg and reset finite range on facility arrival before service."""
    vehicle = state.vehicles[vehicle_id]
    leg = vehicle['active_leg']
    assert leg is not None and leg['arrival_time'] == state.time
    limited = 'remaining_range' in vehicle
    if not limited:
        vehicle['distance'] += leg['distance']
    vehicle.update(location=leg['target'], status='idle', active_leg=None)
    details = dict(leg)
    if limited:
        details['remaining_range_on_arrival'] = vehicle['remaining_range']
        vehicle['leg_distance_accounted'] = 0.0
        vehicle['position'] = list(leg['target_coordinates'])
    record_event(state, 'leg_arrival', **details)
    reset_vehicle_range(scenario, state, vehicle_id)


def collect_inventory_products(scenario: Record, state: 'InventoryRunState', vehicle_id: str) -> None:
    """Pick up this vehicle's assigned released products in deterministic ID order."""
    vehicle = state.vehicles[vehicle_id]
    for product_id in sorted(state.products):
        product = state.products[product_id]
        if product['assigned_vehicle'] == vehicle_id and product['state'] == 'at_collection_point':
            product.update(state='onboard_av', location=vehicle_id)
            product['milestones']['pickup'] = state.time
            vehicle['load'].append(product_id)
            record_event(state, 'pickup', product_id=product_id, vehicle_id=vehicle_id,
                         collection_point=scenario['collection_point']['id'])


def retry_inventory_admissions(scenario: Record, state: 'InventoryRunState') -> bool:
    """Admit one-at-a-time by arrival then AV ID; log blocking and retries, preserving onboard loads."""
    changed = False
    definitions = {product['id']: product for product in scenario['products']}
    capacity = scenario['plant']['inventory_capacity']
    for vehicle_id in sorted(state.waiting_vehicles, key=lambda vehicle_id: (state.waiting_vehicles[vehicle_id], vehicle_id)):
        vehicle = state.vehicles[vehicle_id]
        eligible = [product_id for product_id in vehicle['load'] if state.products[product_id]['planned_destination'] == scenario['plant']['id']]
        while eligible and (capacity is None or inventory_occupancy(state) < capacity):
            if vehicle['wait_logged']:
                record_event(state, 'unload_retry', vehicle_id=vehicle_id,
                             arrival_time=state.waiting_vehicles[vehicle_id])
            product_id = eligible.pop(0)
            product = state.products[product_id]
            assert definitions[product_id]['disposition'] == 'remanufacture' and product['planned_destination'] == scenario['plant']['id']
            vehicle['load'].remove(product_id)
            product.update(state='in_plant_inventory', location=scenario['plant']['id'],
                           actual_destination=scenario['plant']['id'])
            product['milestones']['admission'] = state.time
            state.inventory[product_id] = state.time
            record_event(state, 'plant_admission', product_id=product_id, vehicle_id=vehicle_id,
                plant_id=scenario['plant']['id'], waiting_count=len(state.inventory), held_count=len(state.held_slots))
            changed = True
        if eligible:
            vehicle['status'] = 'waiting_to_unload'
            if not vehicle['wait_logged']:
                vehicle.update(wait_logged=True, waiting_since=state.time)
                record_event(state, 'unload_wait', vehicle_id=vehicle_id,
                    arrival_time=state.waiting_vehicles[vehicle_id], load=list(vehicle['load']),
                    waiting_count=len(state.inventory), held_count=len(state.held_slots))
        else:
            if vehicle['waiting_since'] is not None:
                vehicle['waiting_to_unload'] += state.time - vehicle['waiting_since']
            vehicle.update(status='idle', waiting_since=None, wait_logged=False)
            del state.waiting_vehicles[vehicle_id]
    return changed


def delivery_site(scenario: Record, destination_id: str) -> Record:
    """Return the fixed plant or recycler site record; raise for an unknown destination."""
    return next(site for site in [scenario['plant'], *scenario['recycling_facilities']] if site['id'] == destination_id)


def accept_recycling_products(scenario: Record, state: InventoryRunState, vehicle_id: str) -> None:
    """Accept eligible onboard products immediately at the trip's recycler and record terminal histories."""
    vehicle = state.vehicles[vehicle_id]
    assert vehicle['location'] in {recycler['id'] for recycler in scenario['recycling_facilities']}
    definitions = {product['id']: product for product in scenario['products']}
    for product_id in list(vehicle['load']):
        product = state.products[product_id]
        if product['planned_destination'] != vehicle['location']:
            continue
        assert definitions[product_id]['disposition'] == 'recycle' and not definitions[product_id]['operations']
        assert product['planned_destination'] == vehicle['location']
        vehicle['load'].remove(product_id)
        product.update(state='recycled', location=None, actual_destination=vehicle['location'])
        product['milestones']['recycling'] = state.time
        record_event(state, 'recycler_acceptance', product_id=product_id, vehicle_id=vehicle_id,
                     recycler_id=vehicle['location'])
