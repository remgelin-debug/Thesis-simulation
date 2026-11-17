"""Released-state physical projections for quality-aware dynamic insertion candidates."""
import copy
from dataclasses import asdict
from .models import Record, InventoryRunState
from .plant_network import configured_plants, initialize_plant_states


def build_dynamic_forecast_context(scenario: Record, state: InventoryRunState) -> Record:
    """Copy released products and current plant/transport state without future return information."""
    released = {product_id for product_id, product in state.products.items() if product['state'] != 'unreleased'}
    snapshot = asdict(state)
    del snapshot['events'], snapshot['decisions']
    if not snapshot['plant_states']:
        del snapshot['plant_states']
    snapshot['products'] = {product_id: product for product_id, product in snapshot['products'].items() if product_id in released}
    physical = {key: copy.deepcopy(scenario[key]) for key in ('horizon', 'plant', 'collection_points', 'recycling_facilities', 'vehicles')}
    if 'plants' in scenario:
        physical['plants'] = copy.deepcopy(configured_plants(scenario))
    physical.update(format_version=3, policy='dynamic_quality_aware', trips=[])
    physical['products'] = [{key: copy.deepcopy(product[key]) for key in
        ('id', 'origin', 'release_time', 'disposition', 'operations')} for product in scenario['products'] if product['id'] in released]
    return {'scenario': physical, 'state': snapshot}


def forecast_dynamic_insertion(policy_input: Record, candidate: Record) -> Record:
    """Project revised candidate/committed routes through physical unloading and plant work on copied state."""
    from .dynamic_engine import run_dynamic_simulation
    context = policy_input['forecast_context']
    scenario = context['scenario']
    state = InventoryRunState(**copy.deepcopy(context['state']))
    initialize_plant_states(scenario, state)
    for request in policy_input['requests']:
        product = state.products[request['product_id']]
        product.update(assigned_vehicle=request['assigned_vehicle'], planned_destination=request['planned_destination'])
    product = state.products[candidate['product_id']]
    product.update(assigned_vehicle=candidate['vehicle_id'], planned_destination=candidate['destination'])
    for observed in policy_input['vehicles']:
        vehicle = state.vehicles[observed['id']]
        route = candidate['new_remaining_stops'] if observed['id'] == candidate['vehicle_id'] else observed['remaining_stops']
        vehicle['remaining_stops'] = copy.deepcopy(route)
        leg = vehicle['active_leg']
        if leg is not None and route and leg['target'] != route[0]['site_id']:
            vehicle.update(active_leg=None, status='idle', leg_distance_accounted=0.0)
    arrivals: list[Record] = []
    projected = run_dynamic_simulation(scenario, initial_state=state, dispatch_enabled=False, projected_deliveries=arrivals)
    candidate_id = candidate['product_id']
    product = projected.products[candidate_id]
    arrival = next((record['arrival_time'] for record in arrivals if record['product_id'] == candidate_id), None)
    admission = product['milestones'].get('admission', product['milestones'].get('recycling'))
    first_start = next((event['time'] for event in projected.events
                        if event['event_type'] == 'operation_start' and event['product_id'] == candidate_id), None)
    return {'predicted_arrival': arrival, 'predicted_admission': admission, 'predicted_first_start': first_start,
        'predicted_inventory_wait': first_start - admission if first_start is not None else None,
        'projected_deliveries': arrivals}


def dynamic_quality_score(candidate: Record) -> list:
    """Prefer a horizon start, admission-to-start wait, added distance, then existing stable insertion ties."""
    started = candidate['predicted_first_start'] is not None
    return [0 if started else 1, candidate['predicted_inventory_wait'] if started else 0.0,
            round(candidate['added_distance'], 12), *candidate['score'][1:]]
