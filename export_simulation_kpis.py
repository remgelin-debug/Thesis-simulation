"""Export one Excel-ready KPI row per completed thesis simulation run, using saved artifacts only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Record = dict[str, Any]
RUN_MARKERS = {'run_manifest.json', 'events.jsonl', 'summary.json', 'final_state.json'}
COUNT_FIELDS = {'Generated': 'generated', 'Remanufactured': 'remanufactured', 'Recycled': 'recycled', 'Unfinished': 'unfinished'}
PASSIVE_EVENTS = {'assignment', 'dispatch', 'route_revised', 'range_plan_check', 'range_reset',
                  'trip_rejected', 'vehicle_available', 'stop_completed', 'unload_retry', 'horizon'}


@dataclass
class PlantTotals:
    """Accumulate waiting stock, held capacity and workstation activity for one facility."""

    capacity: int | None
    available_time: float
    waiting: set[str] = field(default_factory=set)
    held: set[str] = field(default_factory=set)
    busy_stations: set[str] = field(default_factory=set)
    waiting_area: float = 0.0
    occupied_area: float = 0.0
    busy_time: float = 0.0
    processing_time: float = 0.0
    peak_occupied: int = 0

    @property
    def occupied(self) -> int:
        """Count waiting products plus reserved return slots, excluding unreserved in-process products."""
        return len(self.waiting) + len(self.held)

    def accumulate(self, elapsed: float) -> None:
        """Integrate the preceding state over one positive simulation-time interval."""
        self.waiting_area += len(self.waiting) * elapsed
        self.occupied_area += self.occupied * elapsed
        self.busy_time += elapsed if self.busy_stations else 0.0
        self.processing_time += len(self.busy_stations) * elapsed

    def observe_peak(self) -> None:
        """Include each post-event capacity state in the peak, even at zero-duration timestamps."""
        self.peak_occupied = max(self.peak_occupied, self.occupied)


def reject_nonfinite(token: str) -> None:
    """Reject nonstandard JSON NaN/Infinity tokens before they can contaminate a CSV metric."""
    raise ValueError(f'Nonfinite JSON number: {token}')


def read_json(path: Path) -> Record:
    """Read a UTF-8 JSON object, accepting a BOM and leaving the source unchanged."""
    with path.open(encoding='utf-8-sig') as source:
        try:
            value = json.load(source, parse_constant=reject_nonfinite)
        except ValueError as error:
            raise ValueError(f'{path.name}: {error}') from error
    if not isinstance(value, dict):
        raise ValueError(f'Expected a JSON object in {path}')
    return value


def finite_nonnegative(value: Any, name: str) -> float:
    """Require a finite nonnegative metric/time without accepting booleans as numbers."""
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return value


def ratio(numerator: float, denominator: float, multiplier: float = 1) -> float | None:
    """Return an optional scaled ratio; CSV writes None as blank when the denominator is zero."""
    return numerator / denominator * multiplier if denominator > 0 else None


def scheduled_time(windows: list[list[float]], horizon: float) -> float:
    """Measure the union of saved availability windows intersected with [0, horizon]."""
    intervals = []
    for opening, closing in windows:
        opening = finite_nonnegative(opening, 'Window opening')
        closing = finite_nonnegative(closing, 'Window closing')
        if closing <= opening:
            raise ValueError('Availability windows must have positive duration')
        start, end = max(0, opening), min(horizon, closing)
        if start < end:
            intervals.append((start, end))
    total = 0.0
    previous_end = 0.0
    for start, end in sorted(intervals):
        total += max(0.0, end - max(start, previous_end))
        previous_end = max(previous_end, end)
    return total


def require_summary_match(summary: Record, field_name: str, calculated: float) -> None:
    """Reject missing/nonfinite or inconsistent saved totals rather than combining incompatible artifacts."""
    saved = finite_nonnegative(summary[field_name], f'Summary {field_name}')
    if not math.isclose(calculated, saved, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f'Event-derived {field_name}={calculated} disagrees with summary.json value {saved}')


def measure_run(run: Path, manifest: Record) -> Record:
    """Integrate recorded actual motion, collection stock, plant capacity and processing through the horizon."""
    scenario_path = run / 'scenario.json'
    scenario = read_json(scenario_path)
    if hashlib.sha256(scenario_path.read_bytes()).hexdigest() != manifest['scenario_sha256']:
        raise ValueError('Saved scenario SHA-256 does not match run_manifest.json')
    summary = read_json(run / 'summary.json')
    horizon = finite_nonnegative(scenario['horizon'], 'Simulation horizon')
    definitions = scenario['plants'] if 'plants' in scenario else [scenario['plant']]
    if not definitions:
        raise ValueError('At least one plant is needed for KPI export')
    plants = {plant['id']: PlantTotals(plant['inventory_capacity'], sum(
        scheduled_time(station['windows'], horizon) for station in plant['workstations'])) for plant in definitions}
    if any(plant.capacity is not None and (type(plant.capacity) is not int or plant.capacity <= 0) for plant in plants.values()):
        raise ValueError('Plant capacity must be a positive integer or null')
    stations = {plant['id']: {station['id'] for station in plant['workstations']} for plant in definitions}
    vehicles = scenario['vehicles'] if 'vehicles' in scenario else [scenario['vehicle']]
    speeds = {vehicle['id']: finite_nonnegative(vehicle['speed'], 'Vehicle speed') for vehicle in vehicles}
    if any(speed == 0 for speed in speeds.values()):
        raise ValueError('Vehicle speed must be positive')
    loads: dict[str, set[str]] = {identifier: set() for identifier in speeds}
    destinations = {}
    travelling: set[str] = set()
    waiting_to_unload: set[str] = set()
    released, completed, recycled = set(), set(), set()
    awaiting_pickup: dict[str, float] = {}
    travel_time = distance = unload_wait = last_time = 0.0
    collection_area = pickup_wait = 0.0
    picked_up = peak_occupied = 0
    last_event_type = None
    sequence = 0

    with (run / 'events.jsonl').open(encoding='utf-8-sig') as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line, parse_constant=reject_nonfinite)
            except ValueError as error:
                raise ValueError(f'events.jsonl line {line_number}: {error}') from error
            if not isinstance(event, dict):
                raise ValueError(f'Event line {line_number} must be a JSON object')
            sequence += 1
            if type(event['sequence']) is not int or event['sequence'] != sequence:
                raise ValueError(f'Expected event sequence {sequence} at line {line_number}')
            time, kind = finite_nonnegative(event['time'], 'Event time'), event['event_type']
            if not last_time <= time <= horizon:
                raise ValueError(f'Event time out of order or beyond horizon: {time}')
            elapsed = time - last_time
            travel_time += elapsed * len(travelling)
            distance += elapsed * sum(speeds[vehicle_id] for vehicle_id in travelling)
            unload_wait += elapsed * len(waiting_to_unload)
            collection_area += elapsed * len(awaiting_pickup)
            for plant in plants.values():
                plant.accumulate(elapsed)
            last_time, last_event_type = time, kind

            if kind == 'assignment':
                destinations[event['product_id']] = event['destination']
            elif kind == 'leg_departure':
                vehicle_id = event['vehicle_id']
                if vehicle_id in travelling or vehicle_id in waiting_to_unload:
                    raise ValueError(f'AV {vehicle_id} departed while travelling or waiting to unload')
                speeds[vehicle_id]  # Require a configured vehicle.
                travelling.add(vehicle_id)
            elif kind in ('leg_arrival', 'leg_cancelled'):
                travelling.remove(event['vehicle_id'])
            elif kind == 'return_release':
                product_id = event['product']['id']
                if product_id in released:
                    raise ValueError(f'Duplicate return release: {product_id}')
                released.add(product_id)
                awaiting_pickup[product_id] = time
            elif kind == 'pickup':
                product_id = event['product_id']
                pickup_wait += time - awaiting_pickup.pop(product_id)
                picked_up += 1
                loads[event['vehicle_id']].add(product_id)
            elif kind == 'unload_wait':
                loads[event['vehicle_id']]  # Require a configured vehicle.
                waiting_to_unload.add(event['vehicle_id'])
            elif kind in ('plant_admission', 'recycler_acceptance'):
                product_id, vehicle_id = event['product_id'], event['vehicle_id']
                destination = event['plant_id'] if kind == 'plant_admission' else event['recycler_id']
                loads[vehicle_id].remove(product_id)
                if not any(destinations[remaining] == destination for remaining in loads[vehicle_id]):
                    waiting_to_unload.discard(vehicle_id)
                if kind == 'plant_admission':
                    plants[destination].waiting.add(product_id)
                else:
                    recycled.add(product_id)
            elif kind in ('inventory_return', 'inventory_hold', 'inventory_slot_release', 'operation_start', 'operation_complete'):
                plant_id, product_id = event.get('plant_id', definitions[0]['id']), event['product_id']
                plant = plants[plant_id]
                if kind == 'inventory_return':
                    plant.held.discard(product_id)
                    plant.waiting.add(product_id)
                elif kind == 'inventory_hold':
                    plant.held.add(product_id)
                elif kind == 'inventory_slot_release':
                    plant.held.discard(product_id)
                else:
                    station_id = event['workstation_id']
                    if station_id not in stations[plant_id]:
                        raise ValueError(f'Unknown workstation: {plant_id} / {station_id}')
                    if kind == 'operation_start':
                        plant.waiting.remove(product_id)
                        if station_id in plant.busy_stations:
                            raise ValueError(f'Workstation already processing: {plant_id} / {station_id}')
                        plant.busy_stations.add(station_id)
                    else:
                        plant.busy_stations.remove(station_id)
            elif kind == 'remanufactured':
                completed.add(event['product_id'])
            elif kind not in PASSIVE_EVENTS:
                raise ValueError(f'Unsupported event type: {kind}')
            for plant in plants.values():
                plant.observe_peak()
            peak_occupied = max(peak_occupied, sum(plant.occupied for plant in plants.values()))

    if last_event_type != 'horizon' or last_time != horizon:
        raise ValueError('Event log does not end at the simulation horizon')
    counts = {'generated': len(released), 'remanufactured': len(completed), 'recycled': len(recycled),
              'unfinished': len(released) - len(completed) - len(recycled)}
    waiting_area = sum(plant.waiting_area for plant in plants.values())
    for name, calculated in {**counts, 'plant_inventory_wait': waiting_area}.items():
        require_summary_match(summary, name, calculated)
    # Motion is derived from the log, including interrupted legs, not from planned arrival times.
    # Legacy format-one summaries did not always include partial-leg distance; keep actual travel authoritative.
    if scenario['format_version'] != 1:
        require_summary_match(summary, 'vehicle_distance', distance)
    if 'waiting_to_unload' in summary:
        require_summary_match(summary, 'waiting_to_unload', unload_wait)
    occupied_area = sum(plant.occupied_area for plant in plants.values())
    total_capacity = sum(plant.capacity for plant in plants.values()) if all(plant.capacity is not None for plant in plants.values()) else None
    verification_path = run / 'verification.json'
    verification = read_json(verification_path).get('status') if verification_path.exists() else 'NOT_VERIFIED'
    if not isinstance(verification, str) or not verification:
        raise ValueError('Saved verification.json status must be a nonempty string')
    provenance = manifest.get('seed_provenance', scenario.get('seed_provenance', {}))
    row = {'Seed': provenance.get('master_seed', provenance.get('seed')), 'Policy': manifest['policy'],
        'Verification': verification, 'Horizon': horizon,
        **{column: counts[name] for column, name in COUNT_FIELDS.items()},
        'AV travel time': travel_time, 'Vehicle distance': distance, 'AV unload wait': unload_wait,
        'Mean release-to-pickup time': ratio(pickup_wait, picked_up),
        'Mean collection-point inventory': ratio(collection_area, horizon),
        'Awaiting pickup at horizon': len(awaiting_pickup),
        'Plant inventory wait': waiting_area, 'Mean waiting inventory': ratio(waiting_area, horizon),
        'Mean occupied slots': ratio(occupied_area, horizon), 'Peak occupied slots': peak_occupied,
        'Mean inventory capacity used (%)': ratio(occupied_area, horizon * total_capacity, 100) if total_capacity is not None else None,
        'Mean plant busy (%)': ratio(sum(plant.busy_time for plant in plants.values()), horizon * len(plants), 100),
        'Workstation utilization of available time (%)': ratio(sum(plant.processing_time for plant in plants.values()),
            sum(plant.available_time for plant in plants.values()), 100),
        'Scenario SHA256': manifest['scenario_sha256']}
    for name, plant in sorted(plants.items()):
        row[f'Plant {name} busy (%)'] = ratio(plant.busy_time, horizon, 100)
        row[f'Plant {name} mean waiting inventory'] = ratio(plant.waiting_area, horizon)
        row[f'Plant {name} mean occupied slots'] = ratio(plant.occupied_area, horizon)
        row[f'Plant {name} peak occupied slots'] = plant.peak_occupied
        row[f'Plant {name} inventory capacity used (%)'] = ratio(plant.occupied_area, horizon * plant.capacity, 100) if plant.capacity is not None else None
        row[f'Plant {name} workstation utilization of available time (%)'] = ratio(plant.processing_time, plant.available_time, 100)
    return row


def discover_runs(root: Path) -> list[Path]:
    """Find run evidence folders, including missing-manifest runs, without treating frozen batch inputs as runs."""
    candidates = {path.parent for path in root.rglob('*') if path.name in RUN_MARKERS and path.is_file()}
    if not candidates and (root / 'scenario.json').is_file():
        candidates.add(root)
    return sorted(candidates)


def export_kpis(run_root: Path, output: Path) -> tuple[int, int]:
    """Export readable completed runs as BOM UTF-8 CSV; report skipped folders and return exported/skipped counts."""
    if not run_root.is_dir():
        raise ValueError(f'Run folder does not exist: {run_root}')
    if output.suffix.lower() != '.csv':
        raise ValueError('KPI output must have a .csv suffix')
    rows = []
    skipped = 0
    for run in discover_runs(run_root):
        try:
            manifest = read_json(run / 'run_manifest.json')
            if manifest['status'] != 'completed':
                raise ValueError('Run is not marked completed')
            metrics = measure_run(run, manifest)
            name = run.relative_to(run_root).as_posix()
            rows.append({'Run': name if name != '.' else run.name, **metrics})
        except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError) as error:
            print(f'Skipped {run}: {error}', file=sys.stderr)
            skipped += 1
    if not rows:
        print('No completed runs with readable KPI data were found; no CSV written.', file=sys.stderr)
        return 0, skipped
    columns = list(dict.fromkeys(key for row in rows for key in row))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8-sig', newline='') as destination:
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Exported {len(rows)} run(s) to {output}; skipped {skipped}.')
    return len(rows), skipped


def main() -> int:
    """Parse a run/batch path and CSV destination; return 0 on success, 1 for skips/no rows, 2 for setup errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path, help='One run folder or a parent containing completed runs')
    parser.add_argument('--output', type=Path, required=True, help='Destination Excel-ready UTF-8 CSV')
    arguments = parser.parse_args()
    try:
        exported, skipped = export_kpis(arguments.run_root, arguments.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0 if exported and not skipped else 1


if __name__ == '__main__':
    raise SystemExit(main())
