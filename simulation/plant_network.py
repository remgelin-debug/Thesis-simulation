"""Per-plant views sharing authoritative products, vehicles and physical plant transitions."""
from .models import Record, InventoryRunState
from .inventory_engine import settle_inventory, check_inventory_state


def configured_plants(scenario: Record) -> list[Record]:
    """Return explicit plant definitions, or the backward-compatible single plant."""
    return scenario['plants'] if 'plants' in scenario else [scenario['plant']]


def initialize_plant_states(scenario: Record, state: InventoryRunState) -> None:
    """Create separate plant containers and bind primary aliases without duplicating physical membership."""
    if len(configured_plants(scenario)) > 1 and not state.plant_states:
        state.plant_states = {plant['id']: {'inventory': {}, 'held_slots': [], 'waiting_vehicles': {},
            'workstations': {station['id']: None for station in plant['workstations']}} for plant in configured_plants(scenario)}
    if state.plant_states:
        primary = state.plant_states[scenario['plant']['id']]
        state.inventory, state.held_slots = primary['inventory'], primary['held_slots']
        state.workstations, state.waiting_vehicles = primary['workstations'], primary['waiting_vehicles']


def local_plant_view(scenario: Record, state: InventoryRunState, plant_id: str) -> tuple[Record, InventoryRunState]:
    """Return a local rule context referencing live containers for exactly one plant."""
    plant = next(plant for plant in configured_plants(scenario) if plant['id'] == plant_id)
    if not state.plant_states:
        return {**scenario, 'plant': plant}, state
    containers = state.plant_states[plant_id]
    view = InventoryRunState(time=state.time, products=state.products, vehicles=state.vehicles,
        inventory=containers['inventory'], held_slots=containers['held_slots'], workstations=containers['workstations'],
        waiting_vehicles=containers['waiting_vehicles'], events=state.events, decisions=state.decisions)
    return {**scenario, 'plant': plant}, view


def settle_plant_network(scenario: Record, state: InventoryRunState) -> list[tuple[float, int, str, str, str]]:
    """Settle every plant using shared physical rules and return unambiguous completion events."""
    completions = []
    for plant in sorted(configured_plants(scenario), key=lambda plant: plant['id']):
        local, view = local_plant_view(scenario, state, plant['id'])
        for time, priority, station_id, kind in settle_inventory(local, view):
            completions.append((time, priority, plant['id'], kind, station_id) if state.plant_states else
                               (time, priority, station_id, kind, ''))
    return completions


def check_plant_network(scenario: Record, state: InventoryRunState) -> None:
    """Assert per-plant capacity and global product conservation without counting primary aliases twice."""
    if not state.plant_states:
        check_inventory_state(scenario, state)
        return
    for plant in configured_plants(scenario):
        containers = state.plant_states[plant['id']]
        assert plant['inventory_capacity'] is None or len(containers['inventory']) + len(containers['held_slots']) <= plant['inventory_capacity']
    for product_id, product in state.products.items():
        physical = int(product['state'] in ('at_collection_point', 'remanufactured', 'recycled'))
        physical += sum(product_id in vehicle['load'] for vehicle in state.vehicles.values())
        physical += sum(product_id in plant['inventory'] for plant in state.plant_states.values())
        physical += sum(operation is not None and operation['product_id'] == product_id
                        for plant in state.plant_states.values() for operation in plant['workstations'].values())
        assert physical == int(product['state'] != 'unreleased')
    for definition in scenario['vehicles']:
        assert len(state.vehicles[definition['id']]['load']) <= definition['payload_capacity']
