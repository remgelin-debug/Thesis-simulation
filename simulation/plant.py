"""Calendar fit and ordered operations through unlimited or finite shared inventory."""
from .models import Record, SimulationState, InventoryRunState, record_event


def next_operation_start(scenario: Record, state: SimulationState) -> tuple[float, Record, list[float]] | None:
    """Find the earliest full-duration window for the product's next operation."""
    operation_id = scenario["product"]["operations"][state.next_operation]
    station = next(station for station in scenario["plant"]["workstations"]
                   if station["operation_id"] == operation_id)
    for window in station["windows"]:
        start = max(state.time, window[0])
        if start < window[1] and start + station["duration"] <= window[1]:
            return start, station, window
    return None


def start_operation(scenario: Record, state: SimulationState, station: Record, window: list[float]) -> float:
    """Transfer inventory to the workstation and return the completion time."""
    assert state.active_operation is None and state.inventory_entered_at is not None
    product_id = scenario["product"]["id"]
    state.inventory.remove(product_id)
    state.inventory_wait += state.time - state.inventory_entered_at
    state.inventory_entered_at = None
    operation = {"product_id": product_id, "plant_id": scenario["plant"]["id"],
        "workstation_id": station["id"], "operation_id": station["operation_id"],
        "operation_index": state.next_operation, "start_time": state.time,
        "end_time": state.time + station["duration"], "window": window}
    state.active_operation = operation
    state.product_state = "processing"
    state.product_location = station["id"]
    record_event(state, "operation_start", **operation)
    return operation["end_time"]


def complete_operation(scenario: Record, state: SimulationState) -> None:
    """Advance one operation, returning unfinished work to inventory or terminating it."""
    operation = state.active_operation
    assert operation is not None and state.time == operation["end_time"]
    state.operation_history.append(dict(operation))
    state.active_operation = None
    state.next_operation += 1
    record_event(state, "operation_complete", **operation)
    if state.next_operation == len(scenario["product"]["operations"]):
        state.product_state = "remanufactured"
        state.product_location = None
        state.milestones["completion"] = state.time
        record_event(state, "remanufactured", product_id=scenario["product"]["id"])
    else:
        state.product_state = "in_plant_inventory"
        state.product_location = scenario["plant"]["id"]
        state.inventory.append(scenario["product"]["id"])
        state.inventory_entered_at = state.time
        record_event(state, "inventory_return", product_id=scenario["product"]["id"],
                     plant_id=scenario["plant"]["id"], next_operation_index=state.next_operation)


def inventory_occupancy(state: 'InventoryRunState') -> int:
    """Return admission capacity consumed by physically waiting products and holds."""
    return len(state.inventory) + len(state.held_slots)


def fitting_window(station: Record, time: float) -> list[float] | None:
    """Return a continuous calendar window fitting a start now, merging adjacent intervals."""
    merged: list[list[float]] = []
    for opening, closing in station['windows']:
        if merged and opening == merged[-1][1]:
            merged[-1][1] = closing
        else:
            merged.append([opening, closing])
    return next((window for window in merged if window[0] <= time < window[1]
                 and time + station['duration'] <= window[1]), None)


def start_inventory_operation(scenario: Record, state: 'InventoryRunState', station: Record) -> float | None:
    """Start the oldest eligible product; hold an intermediate slot or release a final slot."""
    window = fitting_window(station, state.time)
    if state.workstations[station['id']] is not None or window is None:
        return None
    definitions = {p['id']: p for p in scenario['products']}
    eligible = [product_id for product_id in state.inventory
                if definitions[product_id]['operations'][state.products[product_id]['next_operation']] == station['operation_id']]
    if not eligible:
        return None
    product_id = min(eligible, key=lambda candidate: (state.inventory[candidate], candidate))
    product = state.products[product_id]
    product['inventory_wait'] += state.time - state.inventory.pop(product_id)
    operation = {'product_id': product_id, 'plant_id': scenario['plant']['id'], 'workstation_id': station['id'],
        'operation_id': station['operation_id'], 'operation_index': product['next_operation'],
        'start_time': state.time, 'end_time': state.time + station['duration'], 'window': window}
    product.update(state='processing', location=station['id'])
    state.workstations[station['id']] = operation
    record_event(state, 'operation_start', **operation)
    if scenario['plant']['inventory_capacity'] is not None:
        if product['next_operation'] + 1 < len(definitions[product_id]['operations']):
            state.held_slots.append(product_id)
            record_event(state, 'inventory_hold', product_id=product_id, plant_id=scenario['plant']['id'],
                         waiting_count=len(state.inventory), held_count=len(state.held_slots))
        else:
            record_event(state, 'inventory_slot_release', product_id=product_id, plant_id=scenario['plant']['id'],
                         waiting_count=len(state.inventory), held_count=len(state.held_slots))
    return operation['end_time']


def finish_inventory_operation(scenario: Record, state: 'InventoryRunState', workstation_id: str) -> None:
    """Complete one operation, replacing its hold with inventory or recording terminal completion."""
    operation = state.workstations[workstation_id]
    assert operation is not None and operation['end_time'] == state.time
    product_id = operation['product_id']
    product = state.products[product_id]
    product['history'].append(dict(operation))
    product['next_operation'] += 1
    state.workstations[workstation_id] = None
    record_event(state, 'operation_complete', **operation)
    definition = next(p for p in scenario['products'] if p['id'] == product_id)
    if product['next_operation'] == len(definition['operations']):
        product.update(state='remanufactured', location=None)
        product['milestones']['completion'] = state.time
        record_event(state, 'remanufactured', product_id=product_id)
    else:
        replaced = product_id in state.held_slots
        if replaced:
            state.held_slots.remove(product_id)
        state.inventory[product_id] = state.time
        product.update(state='in_plant_inventory', location=scenario['plant']['id'])
        record_event(state, 'inventory_return', product_id=product_id, plant_id=scenario['plant']['id'],
            next_operation_index=product['next_operation'], held_slot_replaced=replaced,
            waiting_count=len(state.inventory), held_count=len(state.held_slots))
