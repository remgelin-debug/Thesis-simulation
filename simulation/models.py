"""State records for legacy single-product and multi-product inventory runs."""
from dataclasses import dataclass, field
from typing import Any

Record = dict[str, Any]


@dataclass
class SimulationState:
    """Own mutable physical state and append-only audit records for one run."""

    time: float = 0.0
    product_state: str = "unreleased"
    product_location: str | None = None
    assigned_vehicle: str | None = None
    planned_destination: str | None = None
    actual_destination: str | None = None
    next_operation: int = 0
    inventory: list[str] = field(default_factory=list)
    inventory_entered_at: float | None = None
    inventory_wait: float = 0.0
    vehicle_location: str = "RF1"
    vehicle_status: str = "idle"
    vehicle_load: list[str] = field(default_factory=list)
    vehicle_distance: float = 0.0
    active_leg: Record | None = None
    active_operation: Record | None = None
    milestones: Record = field(default_factory=dict)
    operation_history: list[Record] = field(default_factory=list)
    events: list[Record] = field(default_factory=list)
    decisions: list[Record] = field(default_factory=list)


def record_event(state: 'SimulationState | InventoryRunState', event_type: str, **details: Any) -> None:
    """Append a uniquely sequenced state change at the current simulation time."""
    state.events.append({"sequence": len(state.events) + 1, "time": state.time,
                         "event_type": event_type, **details})


@dataclass
class InventoryRunState:
    """Own version-two product, vehicle, shared-stock and workstation state."""

    time: float = 0.0
    products: dict[str, Record] = field(default_factory=dict)
    vehicles: dict[str, Record] = field(default_factory=dict)
    workstations: dict[str, Record | None] = field(default_factory=dict)
    inventory: dict[str, float] = field(default_factory=dict)
    held_slots: list[str] = field(default_factory=list)
    waiting_vehicles: dict[str, float] = field(default_factory=dict)
    plant_states: dict[str, Record] = field(default_factory=dict)
    events: list[Record] = field(default_factory=list)
    decisions: list[Record] = field(default_factory=list)


def build_inventory_state(scenario: Record) -> InventoryRunState:
    """Create empty per-entity state and full finite allowances without changing inputs."""
    state = InventoryRunState()
    for product in scenario['products']:
        state.products[product['id']] = {'state': 'unreleased', 'location': None, 'next_operation': 0,
            'assigned_vehicle': None, 'planned_destination': None, 'actual_destination': None,
            'milestones': {}, 'history': [], 'inventory_wait': 0.0}
    for vehicle in scenario['vehicles']:
        state.vehicles[vehicle['id']] = {'status': 'idle', 'location': vehicle['initial_site'],
            'load': [], 'active_leg': None, 'distance': 0.0, 'waiting_since': None,
            'waiting_to_unload': 0.0, 'wait_logged': False}
    for definition in scenario['vehicles']:
        if definition['maximum_range'] is not None:
            site = next(site for site in [*scenario.get('plants', [scenario['plant']]), *scenario['recycling_facilities']] if site['id'] == definition['initial_site'])
            state.vehicles[definition['id']].update(remaining_range=definition['maximum_range'],
                position=list(site['coordinates']), leg_distance_accounted=0.0)
    state.workstations = {s['id']: None for s in scenario['plant']['workstations']}
    return state
