"""Fixed quality-unaware trip audit; no policy forecasting."""
from .models import Record, SimulationState, InventoryRunState, record_event
from .vehicle_range import evaluate_fixed_trip_range
from .forecasting import build_forecast_context, choose_quality_aware_products


def dispatch_fixed_trip(scenario: Record, state: SimulationState) -> None:
    """Record the fixed plan and reserve the released product for its vehicle."""
    product, vehicle = scenario["product"], scenario["vehicle"]
    policy_input = {
        "requests": [{"product_id": product["id"], "collection_point": product["origin"],
                      "release_time": product["release_time"], "disposition": product["disposition"]}],
        "vehicle_id": vehicle["id"], "stops": scenario["trip"]["stops"],
        "departure_time": scenario["trip"]["departure_time"],
        "destination": scenario["plant"]["id"],
    }
    state.decisions.append({"time": state.time, "trigger": "static_departure",
        "last_event_sequence": len(state.events), "policy_input": policy_input,
        "decision": {"assigned_product": product["id"], "stops": scenario["trip"]["stops"]},
        "candidates": [], "tie_break": "Only one product and one configured trip"})
    state.assigned_vehicle = vehicle["id"]
    state.planned_destination = scenario["plant"]["id"]
    record_event(state, "assignment", product_id=product["id"], vehicle_id=vehicle["id"],
                 destination=scenario["plant"]["id"])


def dispatch_inventory_trip(scenario: Record, state: 'InventoryRunState', trip: Record) -> bool:
    """Check the full plan, assign eligible released products or log rejection; return feasibility."""
    vehicle_id = trip['vehicle_id']
    definitions = {p['id']: p for p in scenario['products']}
    available = sorted((product_id for product_id, p in state.products.items()
        if p['state'] == 'at_collection_point' and p['assigned_vehicle'] is None),
        key=lambda product_id: (definitions[product_id]['release_time'], product_id))
    vehicle = next(v for v in scenario['vehicles'] if v['id'] == vehicle_id)
    destination = trip['stops'][-1]
    disposition = 'remanufacture' if destination == scenario['plant']['id'] else 'recycle'
    range_check = evaluate_fixed_trip_range(scenario, state, trip)
    if vehicle['maximum_range'] is not None:
        record_event(state, 'range_plan_check', vehicle_id=vehicle_id, **range_check)
    selected = [product_id for product_id in available if definitions[product_id]['disposition'] == disposition][:vehicle['payload_capacity']]
    if not range_check['feasible'] or scenario['collection_point']['id'] not in trip['stops']:
        selected = []
    state.decisions.append({'time': state.time, 'trigger': 'static_departure',
        'last_event_sequence': len(state.events), 'policy_input': {'vehicle_id': vehicle_id,
            'stops': trip['stops'], 'requests': [{'product_id': product_id, 'collection_point': definitions[product_id]['origin'],
            'release_time': definitions[product_id]['release_time'], 'disposition': definitions[product_id]['disposition']} for product_id in available]},
        'decision': {'assigned_products': selected, 'stops': trip['stops']}, 'tie_break': 'release_time, product_id'})
    if scenario['policy'] == 'static_quality_aware':
        boundary = state.decisions[-1]['policy_input']
        for request in boundary['requests']:
            request['operations'] = list(definitions[request['product_id']]['operations'])
        boundary['forecast_context'] = build_forecast_context(scenario, state)
        if range_check['feasible']:
            selected, candidates, reason = choose_quality_aware_products(boundary)
        else:
            selected, candidates, reason = [], [], 'Fixed trip rejected by range check'
        state.decisions[-1].update(candidates=candidates, selection_reason=reason,
            tie_break='product_id, pickup_stop_index, delivery_stop_index')
        state.decisions[-1]['decision']['assigned_products'] = selected
    if not range_check['feasible']:
        state.decisions[-1]['decision']['rejection'] = range_check['rejection']
        record_event(state, 'trip_rejected', vehicle_id=vehicle_id, **range_check['rejection'])
        return False
    for product_id in selected:
        state.products[product_id].update(assigned_vehicle=vehicle_id, planned_destination=destination)
        record_event(state, 'assignment', product_id=product_id, vehicle_id=vehicle_id, destination=destination)

    return True
