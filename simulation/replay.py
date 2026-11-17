"""Read-only event snapshots and playback clock, without simulation or plotting dependencies."""
from __future__ import annotations

from bisect import bisect_right
import copy
import hashlib
import json
import math
from pathlib import Path

from .models import Record


def normalize_view_scenario(scenario: Record) -> Record:
    """Copy legacy or network inputs into lists used only by the replay display."""
    view = copy.deepcopy(scenario)
    view['plants'] = view['plants'] if 'plants' in view else [view['plant']]
    view['collection_points'] = view['collection_points'] if 'collection_points' in view else [view['collection_point']]
    if view['format_version'] == 1:
        view['vehicles'], view['products'] = [view['vehicle']], [view['product']]
    return view


def initial_view_state(scenario: Record) -> Record:
    """Create empty display containers; actual releases and work come exclusively from recorded events."""
    sites = {site['id']: site for site in [*scenario['plants'], *scenario['collection_points'], *scenario['recycling_facilities']]}
    return {'time': 0.0, 'events_applied': 0, 'last_event': None,
        'products': {product['id']: {'state': 'unreleased', 'location': None, 'next_operation': 0,
            'assigned_vehicle': None, 'planned_destination': None, 'actual_destination': None} for product in scenario['products']},
        'vehicles': {vehicle['id']: {'position': list(sites[vehicle['initial_site']]['coordinates']),
            'location': vehicle['initial_site'], 'status': 'idle', 'load': [], 'active_leg': None,
            'remaining_stops': [], 'remaining_range': vehicle['maximum_range']} for vehicle in scenario['vehicles']},
        'plants': {plant['id']: {'inventory': [], 'held_slots': [], 'waiting_vehicles': [],
            'workstations': {station['id']: None for station in plant['workstations']},
            'processing_history': {station['id']: [] for station in plant['workstations']}} for plant in scenario['plants']}}


def apply_view_event(state: Record, event: Record) -> None:
    """Apply one logged transition to display state in sequence order, without running plant or routing rules."""
    kind = event['event_type']
    product = state['products'].get(event.get('product_id'))
    vehicle = state['vehicles'].get(event.get('vehicle_id'))
    if kind == 'return_release':
        definition = event['product']
        state['products'][definition['id']].update(state='at_collection_point', location=definition['origin'])
    elif kind == 'assignment':
        product.update(assigned_vehicle=event['vehicle_id'], planned_destination=event['destination'])
    elif kind == 'leg_departure':
        vehicle.update(active_leg=copy.deepcopy(event), position=list(event['origin_coordinates']),
            location=None, status='travelling', remaining_range=event.get('remaining_range_at_departure'))
    elif kind == 'leg_arrival':
        vehicle.update(active_leg=None, position=list(event['target_coordinates']), location=event['target'], status='idle')
        if 'remaining_range_on_arrival' in event:
            vehicle['remaining_range'] = event['remaining_range_on_arrival']
    elif kind == 'leg_cancelled':
        vehicle.update(active_leg=None, position=list(event['position']), remaining_range=event['remaining_range'], status='idle')
    elif kind == 'route_revised':
        vehicle['remaining_stops'] = copy.deepcopy(event['new_remaining_stops'])
    elif kind == 'stop_completed':
        vehicle['remaining_stops'].pop(0)
    elif kind == 'pickup':
        vehicle['load'].append(event['product_id'])
        product.update(state='on_vehicle', location=event['vehicle_id'])
    elif kind in ('plant_admission', 'recycler_acceptance'):
        vehicle['load'].remove(event['product_id'])
        destination = event['plant_id'] if kind == 'plant_admission' else event['recycler_id']
        product.update(state='in_plant_inventory' if kind == 'plant_admission' else 'recycled',
            location=destination if kind == 'plant_admission' else None, actual_destination=destination)
        if kind == 'plant_admission':
            state['plants'][destination]['inventory'].append(event['product_id'])
        if not any(state['products'][identifier]['planned_destination'] == destination for identifier in vehicle['load']):
            vehicle['status'] = 'idle'
            for plant in state['plants'].values():
                if event['vehicle_id'] in plant['waiting_vehicles']:
                    plant['waiting_vehicles'].remove(event['vehicle_id'])
    elif kind == 'operation_start':
        plant = state['plants'][event['plant_id']]
        plant['inventory'].remove(event['product_id'])
        plant['workstations'][event['workstation_id']] = copy.deepcopy(event)
        product.update(state='processing', location=event['workstation_id'])
    elif kind == 'operation_complete':
        plant = state['plants'][event['plant_id']]
        completed = copy.deepcopy(plant['workstations'][event['workstation_id']])
        completed.update(end_time=event['time'], completion_sequence=event['sequence'])
        plant['processing_history'][event['workstation_id']].append(completed)
        plant['workstations'][event['workstation_id']] = None
        product.update(state='operation_complete', next_operation=event['operation_index'] + 1)
    elif kind == 'inventory_return':
        plant = state['plants'][event['plant_id']]
        plant['inventory'].append(event['product_id'])
        if event.get('held_slot_replaced'):
            plant['held_slots'].remove(event['product_id'])
        product.update(state='in_plant_inventory', location=event['plant_id'])
    elif kind == 'inventory_hold':
        state['plants'][event['plant_id']]['held_slots'].append(event['product_id'])
    elif kind == 'inventory_slot_release':
        # A one-operation product need not have owned a held slot.
        held = state['plants'][event['plant_id']]['held_slots']
        if event['product_id'] in held:
            held.remove(event['product_id'])
    elif kind == 'remanufactured':
        product.update(state='remanufactured', location=None)
    elif kind in ('unload_wait', 'unload_retry'):
        vehicle['status'] = 'waiting_to_unload'
        waiting = state['plants'][vehicle['location']]['waiting_vehicles']
        if event['vehicle_id'] not in waiting:
            waiting.append(event['vehicle_id'])
    elif kind == 'range_reset':
        vehicle['remaining_range'] = event['range_after']
    elif kind == 'vehicle_available':
        vehicle['status'] = 'idle'
    elif kind not in ('dispatch', 'range_plan_check', 'trip_rejected', 'horizon'):
        raise ValueError(f'Unsupported replay event type: {kind}')
    state.update(time=event['time'], events_applied=event['sequence'], last_event=copy.deepcopy(event))


def interpolate_vehicles(state: Record, time: float) -> None:
    """Sample recorded legs at the requested simulation time, including cancelled or horizon-truncated travel."""
    for vehicle in state['vehicles'].values():
        leg = vehicle['active_leg']
        if leg is None:
            continue
        duration = leg['arrival_time'] - leg['departure_time']
        fraction = min(1.0, max(0.0, (time - leg['departure_time']) / duration)) if duration else 1.0
        vehicle['position'] = [origin + fraction * (target - origin)
            for origin, target in zip(leg['origin_coordinates'], leg['target_coordinates'])]
        allowance = leg.get('remaining_range_at_departure')
        if allowance is not None:
            vehicle['remaining_range'] = max(0.0, allowance - fraction * leg['distance'])
    state['time'] = time


class RunReplay:
    """Read immutable run records and cache every hundredth display state for efficient seeking."""

    def __init__(self, run_folder: Path) -> None:
        """Load raw scenario/manifest/events, check identity/order, and build sparse display checkpoints."""
        self.run_folder = run_folder.resolve()
        scenario_bytes = (self.run_folder / 'scenario.json').read_bytes()
        self.manifest = json.loads((self.run_folder / 'run_manifest.json').read_text(encoding='utf-8'))
        if hashlib.sha256(scenario_bytes).hexdigest() != self.manifest['scenario_sha256']:
            raise ValueError('Replay scenario bytes do not match the saved manifest hash')
        if self.manifest['status'] != 'completed':
            raise ValueError('Replay requires a completed single run')
        self.scenario = normalize_view_scenario(json.loads(scenario_bytes))
        self.horizon = self.scenario['horizon']
        self.events = [json.loads(line) for line in (self.run_folder / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
        self.times = []
        self.checkpoints: dict[int, Record] = {0: initial_view_state(self.scenario)}
        state = copy.deepcopy(self.checkpoints[0])
        previous = 0.0
        for index, event in enumerate(self.events, 1):
            time = event['time']
            if event['sequence'] != index or not math.isfinite(time) or not previous <= time <= self.horizon:
                raise ValueError('Replay requires contiguous event sequences and ordered finite times inside the horizon')
            self.times.append(time)
            previous = time
            apply_view_event(state, event)
            if index % 100 == 0:
                self.checkpoints[index] = copy.deepcopy(state)

    def snapshot(self, time: float, events_applied: int | None = None) -> Record:
        """Return an isolated interpolated state, settled through a time or stopped after one exact event."""
        if not math.isfinite(time) or not 0 <= time <= self.horizon:
            raise ValueError(f'Viewer time must be between 0 and {self.horizon}')
        count = bisect_right(self.times, time) if events_applied is None else events_applied
        if type(count) is not int or not 0 <= count <= len(self.events) or (count and self.times[count - 1] > time):
            raise ValueError('Invalid replay event cursor')
        checkpoint = count // 100 * 100
        state = copy.deepcopy(self.checkpoints[checkpoint])
        for event in self.events[checkpoint:count]:
            apply_view_event(state, event)
        interpolate_vehicles(state, time)
        return state


class ReplayController:
    """Own a paused/resumed viewing clock and exact event cursor without mutating saved evidence."""

    def __init__(self, replay: RunReplay, time: float = 0.0, speed: float = 1.0) -> None:
        """Start paused at a settled simulation time with a positive playback speed."""
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError('Playback speed must be positive and finite')
        self.replay, self.speed, self.paused = replay, speed, True
        self.state = replay.snapshot(time)

    def seek_time(self, time: float) -> None:
        """Pause and apply every event through the requested time, including all same-time transitions."""
        self.state = self.replay.snapshot(time)
        self.paused = True

    def seek_event(self, sequence: int) -> None:
        """Pause immediately after one numbered log record, retaining its same-timestamp microstate."""
        if type(sequence) is not int or not 1 <= sequence <= len(self.replay.events):
            raise ValueError(f'Event sequence must be between 1 and {len(self.replay.events)}')
        self.state = self.replay.snapshot(self.replay.times[sequence - 1], sequence)
        self.paused = True

    def step_next_event(self) -> None:
        """Pause and consume exactly one next event, even when its timestamp equals the current time."""
        sequence = self.state['events_applied'] + 1
        self.paused = True
        if sequence <= len(self.replay.events):
            self.seek_event(sequence)

    def step_previous_event(self) -> None:
        """Pause and remove exactly one applied record; cursor zero is the state before any events."""
        sequence = self.state['events_applied'] - 1
        self.paused = True
        if sequence <= 0:
            self.state = self.replay.snapshot(0.0, 0)
        else:
            self.seek_event(sequence)

    def toggle_play(self) -> None:
        """Toggle pause/resume; at the horizon keep playback paused until explicitly seeking earlier."""
        if self.state['time'] < self.replay.horizon:
            self.paused = not self.paused

    def advance(self, elapsed_seconds: float) -> None:
        """Advance a running view by elapsed wall time times speed, never by a frame counter."""
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError('Elapsed playback time must be nonnegative and finite')
        if self.paused:
            return
        time = min(self.replay.horizon, self.state['time'] + elapsed_seconds * self.speed)
        self.state = self.replay.snapshot(time)
        if time == self.replay.horizon:
            self.paused = True
