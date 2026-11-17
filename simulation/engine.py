"""Event clock for the single fixed journey, including horizon truncation."""
import heapq
from .models import Record, SimulationState, record_event
from .plant import next_operation_start, start_operation, complete_operation
from .routing import dispatch_fixed_trip
from .transport import depart_leg, arrive_leg, pick_up_product, admit_product


def run_simulation(scenario: Record) -> SimulationState:
    """Execute one journey in timestamp order and return its settled horizon state."""
    state = SimulationState(vehicle_location=scenario["plant"]["id"])
    # Priority resolves equal times: completions, releases, arrivals, starts, dispatch.
    pending: list[tuple[float, int, str]] = [
        (scenario["product"]["release_time"], 1, "release"),
        (scenario["trip"]["departure_time"], 4, "dispatch")]
    heapq.heapify(pending)
    scheduled_station: Record | None = None
    scheduled_window: list[float] | None = None
    while pending and pending[0][0] <= scenario["horizon"]:
        state.time, _, action = heapq.heappop(pending)
        if action == "release":
            state.product_state = "at_collection_point"
            state.product_location = scenario["collection_point"]["id"]
            state.milestones["release"] = state.time
            record_event(state, "return_release", product=dict(scenario["product"]))
        elif action == "dispatch":
            dispatch_fixed_trip(scenario, state)
            arrival = depart_leg(scenario, state, scenario["plant"], scenario["collection_point"])
            heapq.heappush(pending, (arrival, 2, "collection_arrival"))
        elif action == "collection_arrival":
            arrive_leg(scenario, state)
            pick_up_product(scenario, state)
            arrival = depart_leg(scenario, state, scenario["collection_point"], scenario["plant"])
            heapq.heappush(pending, (arrival, 2, "plant_arrival"))
        elif action == "plant_arrival":
            arrive_leg(scenario, state)
            admit_product(scenario, state)
        elif action == "start":
            assert scheduled_station is not None and scheduled_window is not None
            end = start_operation(scenario, state, scheduled_station, scheduled_window)
            heapq.heappush(pending, (end, 0, "complete"))
        elif action == "complete":
            complete_operation(scenario, state)
        if action in ("plant_arrival", "complete") and state.inventory:
            opportunity = next_operation_start(scenario, state)
            if opportunity is not None:
                start, scheduled_station, scheduled_window = opportunity
                heapq.heappush(pending, (start, 3, "start"))
    state.time = scenario["horizon"]
    if state.active_leg is not None:
        leg = state.active_leg
        fraction = (state.time - leg["departure_time"]) / (leg["arrival_time"] - leg["departure_time"])
        state.vehicle_distance += fraction * leg["distance"]
    if state.inventory_entered_at is not None:
        state.inventory_wait += state.time - state.inventory_entered_at
    record_event(state, "horizon", product_state=state.product_state)
    return state
