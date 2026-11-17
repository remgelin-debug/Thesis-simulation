"""Independently replay legacy and checkpoint-3a saved runs using the standard library."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

Record = dict[str, Any]
TOLERANCE = 1e-9
REQUIRED_FILES = ("scenario.json", "run_manifest.json", "events.jsonl",
                  "dispatch_decisions.jsonl", "summary.json", "final_state.json")


class EvidenceProblem(Exception):
    """Stop an unsafe check when evidence is missing, malformed or unsupported."""


class ReplayViolation(Exception):
    """Stop replay after recording a proved transition violation."""


def values_match(expected: Any, observed: Any) -> bool:
    """Compare JSON trees with absolute/relative numeric tolerance and exact structure."""
    if isinstance(expected, bool) or isinstance(observed, bool):
        return type(expected) is type(observed) and expected == observed
    if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        return math.isfinite(expected) and math.isfinite(observed) and math.isclose(
            expected, observed, rel_tol=TOLERANCE, abs_tol=TOLERANCE)
    if isinstance(expected, dict) and isinstance(observed, dict):
        return expected.keys() == observed.keys() and all(
            values_match(value, observed[key]) for key, value in expected.items())
    if isinstance(expected, list) and isinstance(observed, list):
        return len(expected) == len(observed) and all(
            values_match(left, right) for left, right in zip(expected, observed))
    return type(expected) is type(observed) and expected == observed


def add_finding(findings: list[Record], category: str, rule: str, expected: Any,
                observed: Any, event: Record | None = None, **identifiers: Any) -> None:
    """Append one diagnostic with evidence location and expected/observed values."""
    event_identifiers = {key: event[key] for key in
                         ("product_id", "vehicle_id", "plant_id", "workstation_id", "facility_id", "recycler_id")
                         if event is not None and key in event}
    event_identifiers.update(identifiers)
    findings.append({"category": category, "rule": rule,
        "time": event.get("time") if event else None,
        "event_sequence": event.get("sequence") if event else None,
        "ids": event_identifiers, "expected": expected, "observed": observed})


def check_value(findings: list[Record], rule: str, expected: Any, observed: Any,
                event: Record | None = None, **identifiers: Any) -> bool:
    """Record a rule violation on mismatch; return whether the comparison passed."""
    matched = values_match(expected, observed)
    if not matched:
        add_finding(findings, "violation", rule, expected, observed, event, **identifiers)
    return matched


def reject_nonfinite_json(token: str) -> None:
    """Reject nonstandard NaN/Infinity JSON constants instead of admitting unsafe evidence."""
    raise ValueError(f"Nonfinite JSON constant: {token}")


def parse_json_evidence(contents: str) -> Any:
    """Parse JSON and reject numeric overflow anywhere in the resulting tree."""
    value = json.loads(contents, parse_constant=reject_nonfinite_json)
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
        elif type(item) in (int, float):
            try:
                finite = math.isfinite(item)
            except OverflowError as error:
                raise ValueError("JSON number exceeds finite checking range") from error
            if not finite:
                raise ValueError("JSON number exceeds finite checking range")
    return value


def read_evidence(run_directory: Path, findings: list[Record]) -> dict[str, Any]:
    """Read every required artifact independently, collecting missing/parse findings."""
    evidence: dict[str, Any] = {}
    for filename in REQUIRED_FILES:
        path = run_directory / filename
        try:
            contents = path.read_text(encoding="utf-8-sig")
            if filename.endswith(".jsonl"):
                value = [parse_json_evidence(line) for line in contents.splitlines() if line.strip()]
                if not all(isinstance(record, dict) for record in value):
                    raise EvidenceProblem("Each JSONL record must be an object")
            else:
                value = parse_json_evidence(contents)
                if not isinstance(value, dict):
                    raise EvidenceProblem("Expected a JSON object")
            evidence[filename] = value
        except (OSError, UnicodeError, ValueError, EvidenceProblem) as error:
            add_finding(findings, "incomplete", "evidence.read", f"Readable {filename}",
                        str(error), file=filename)
    return evidence


def validate_supported_scenario(scenario: Record) -> None:
    """Validate the independent verifier's single-journey scope without simulator code."""
    if scenario["format_version"] == 3:
        validate_dynamic_inputs(scenario)
        return
    if scenario["format_version"] == 2:
        validate_inventory_inputs(scenario)
        return
    if scenario["format_version"] != 1 or scenario["policy"] != "static_quality_unaware":
        raise EvidenceProblem("Supported scenario: format 1, static_quality_unaware")
    plant, point, vehicle, product, trip = (scenario[name] for name in
        ("plant", "collection_point", "vehicle", "product", "trip"))
    if (plant["inventory_capacity"] is not None or vehicle["maximum_range"] is not None
            or scenario["recycling_facilities"] or product["disposition"] != "remanufacture"
            or vehicle["payload_capacity"] != 1):
        raise EvidenceProblem("Only checkpoint-one unlimited, non-recycling, payload-one runs are supported")
    if (vehicle["initial_site"] != plant["id"] or product["origin"] != point["id"]
            or trip["stops"] != [point["id"], plant["id"]]):
        raise EvidenceProblem("Expected one plant-to-collection-to-plant trip")
    if not all(isinstance(identifier, str) and identifier for identifier in
               (plant["id"], point["id"], vehicle["id"], product["id"])):
        raise EvidenceProblem("Entity IDs must be nonempty strings")
    numbers = [scenario["horizon"], product["release_time"], trip["departure_time"], vehicle["speed"]]
    for site in (plant, point):
        if len(site["coordinates"]) != 2:
            raise EvidenceProblem("Expected two-dimensional coordinates")
        numbers.extend(site["coordinates"])
    stations = plant["workstations"]
    if len({station["operation_id"] for station in stations}) != len(stations):
        raise EvidenceProblem("Expected one workstation per operation")
    if len({station["id"] for station in stations}) != len(stations):
        raise EvidenceProblem("Expected unique workstation IDs")
    if not product["operations"] or any(operation not in {s["operation_id"] for s in stations}
                                        for operation in product["operations"]):
        raise EvidenceProblem("Product route has no compatible workstation")
    for station in stations:
        numbers.append(station["duration"])
        previous_end = -math.inf
        for start, end in station["windows"]:
            numbers.extend((start, end))
            if not 0 <= start < end or start <= previous_end:
                raise EvidenceProblem("Unsupported or invalid calendar intervals")
            previous_end = end
        if station["duration"] <= 0:
            raise EvidenceProblem("Expected positive processing duration")
    if not all(type(value) in (int, float) and math.isfinite(value) for value in numbers):
        raise EvidenceProblem("Expected finite numeric times, coordinates and speed")
    if not (vehicle["speed"] > 0 and 0 <= product["release_time"] <= trip["departure_time"] <= scenario["horizon"]):
        raise EvidenceProblem("Expected positive speed and release <= departure <= horizon")
    scenario["seed_provenance"]


def initial_replay_state(scenario: Record) -> Record:
    """Build independent empty physical state matching the documented output format."""
    return {"time": 0, "product_state": "unreleased", "product_location": None,
        "assigned_vehicle": None, "planned_destination": None, "actual_destination": None,
        "next_operation": 0, "inventory": [], "inventory_entered_at": None, "inventory_wait": 0,
        "vehicle_location": scenario["vehicle"]["initial_site"], "vehicle_status": "idle",
        "vehicle_load": [], "vehicle_distance": 0, "active_leg": None, "active_operation": None,
        "milestones": {}, "operation_history": [], "product_id": scenario["product"]["id"]}


def calculate_leg(scenario: Record, state: Record, origin: Record, target: Record) -> Record:
    """Calculate a leg from coordinates, speed and replayed load; mutate nothing."""
    dx = target["coordinates"][0] - origin["coordinates"][0]
    dy = target["coordinates"][1] - origin["coordinates"][1]
    distance = math.sqrt(dx * dx + dy * dy)
    return {"vehicle_id": scenario["vehicle"]["id"], "origin": origin["id"], "target": target["id"],
        "origin_coordinates": origin["coordinates"], "target_coordinates": target["coordinates"],
        "departure_time": state["time"], "arrival_time": state["time"] + distance / scenario["vehicle"]["speed"],
        "distance": distance, "load": list(state["vehicle_load"])}


def find_operation_opportunity(scenario: Record, state: Record) -> Record | None:
    """Independently calculate the next eligible full-fit start, or no opportunity."""
    operation_id = scenario["product"]["operations"][state["next_operation"]]
    station = next(item for item in scenario["plant"]["workstations"] if item["operation_id"] == operation_id)
    for opening, closing in station["windows"]:
        start = max(state["time"], opening)
        if start < closing and start + station["duration"] <= closing:
            return {"product_id": scenario["product"]["id"], "plant_id": scenario["plant"]["id"],
                "workstation_id": station["id"], "operation_id": operation_id,
                "operation_index": state["next_operation"], "start_time": start,
                "end_time": start + station["duration"], "window": [opening, closing]}
    return None


def expected_transition(scenario: Record, state: Record, phase: str) -> tuple[float, str, Record]:
    """Derive the next legal transition from replay state, never from future log records."""
    product, plant, vehicle, point = (scenario[name] for name in
                                    ("product", "plant", "vehicle", "collection_point"))
    time = state["time"]
    if phase == "release":
        return product["release_time"], "return_release", {"product": product}
    if phase == "assignment":
        return scenario["trip"]["departure_time"], phase, {"product_id": product["id"],
            "vehicle_id": vehicle["id"], "destination": plant["id"]}
    if phase in ("outbound", "inbound"):
        origin, target = (plant, point) if phase == "outbound" else (point, plant)
        return time, "leg_departure", calculate_leg(scenario, state, origin, target)
    if phase in ("collection_arrival", "plant_arrival"):
        leg = state["active_leg"]
        return leg["arrival_time"], "leg_arrival", leg
    if phase == "pickup":
        return time, phase, {"product_id": product["id"], "vehicle_id": vehicle["id"],
                             "collection_point": point["id"]}
    if phase == "plant_admission":
        return time, phase, {"product_id": product["id"], "plant_id": plant["id"], "vehicle_id": vehicle["id"]}
    if phase == "operation_start":
        operation = find_operation_opportunity(scenario, state)
        if operation is not None:
            return operation["start_time"], phase, operation
        return scenario["horizon"], "horizon", {"product_state": state["product_state"]}
    if phase == "operation_complete":
        operation = state["active_operation"]
        return operation["end_time"], phase, operation
    if phase == "inventory_return":
        return time, phase, {"product_id": product["id"], "plant_id": plant["id"],
                             "next_operation_index": state["next_operation"]}
    if phase == "remanufactured":
        return time, phase, {"product_id": product["id"]}
    return scenario["horizon"], "horizon", {"product_state": state["product_state"]}


def apply_transition(scenario: Record, state: Record, phase: str, event: Record) -> str:
    """Apply a validated logged event to replay state; return the next protocol phase."""
    product, plant, vehicle = (scenario[name] for name in ("product", "plant", "vehicle"))
    state["time"] = event["time"]
    if phase == "release":
        state.update(product_state="at_collection_point", product_location=product["origin"])
        state["milestones"]["release"] = state["time"]
        return "assignment"
    if phase == "assignment":
        state.update(assigned_vehicle=vehicle["id"], planned_destination=plant["id"])
        return "outbound"
    if phase in ("outbound", "inbound"):
        state["active_leg"] = {key: value for key, value in event.items()
                               if key not in ("sequence", "time", "event_type")}
        state["vehicle_status"] = "travelling"
        return "collection_arrival" if phase == "outbound" else "plant_arrival"
    if phase in ("collection_arrival", "plant_arrival"):
        state["vehicle_distance"] += state["active_leg"]["distance"]
        state.update(vehicle_location=state["active_leg"]["target"], vehicle_status="idle", active_leg=None)
        return "pickup" if phase == "collection_arrival" else "plant_admission"
    if phase == "pickup":
        state.update(product_state="onboard_av", product_location=vehicle["id"])
        state["vehicle_load"].append(product["id"])
        state["milestones"]["pickup"] = state["time"]
        return "inbound"
    if phase == "plant_admission":
        state["vehicle_load"].remove(product["id"])
        state.update(actual_destination=plant["id"])
        state["milestones"]["admission"] = state["time"]
    if phase in ("plant_admission", "inventory_return"):
        state["inventory"].append(product["id"])
        state.update(product_state="in_plant_inventory", product_location=plant["id"],
                     inventory_entered_at=state["time"])
        return "operation_start"
    if phase == "operation_start":
        state["inventory_wait"] += state["time"] - state["inventory_entered_at"]
        state["inventory"].remove(product["id"])
        operation = {key: value for key, value in event.items()
                     if key not in ("sequence", "time", "event_type")}
        state.update(inventory_entered_at=None, active_operation=operation,
                     product_state="processing", product_location=event["workstation_id"])
        return "operation_complete"
    if phase == "operation_complete":
        state["operation_history"].append(dict(state["active_operation"]))
        state["active_operation"] = None
        state["next_operation"] += 1
        return "remanufactured" if state["next_operation"] == len(product["operations"]) else "inventory_return"
    if phase == "remanufactured":
        state.update(product_state="remanufactured", product_location=None)
        state["milestones"]["completion"] = state["time"]
        return "horizon"
    return phase


def check_conservation(scenario: Record, state: Record, event: Record, findings: list[Record]) -> None:
    """Check exclusive physical membership after a fully settled timestamp."""
    locations = [state["product_state"] == "at_collection_point", bool(state["vehicle_load"]),
                 bool(state["inventory"]), state["active_operation"] is not None,
                 state["product_state"] == "remanufactured"]
    check_value(findings, "product.conservation", int(state["product_state"] != "unreleased"),
                sum(locations), event, product_id=scenario["product"]["id"])
    check_value(findings, "vehicle.payload", True,
                len(state["vehicle_load"]) <= scenario["vehicle"]["payload_capacity"], event)


def replay_events(scenario: Record, events: list[Record], findings: list[Record]) -> Record | None:
    """Check logged transitions and reconstruct the horizon state; stop unsafe replay on violation."""
    state = initial_replay_state(scenario)
    phase = "release"
    finished = False
    for index, event in enumerate(events, start=1):
        if not all(key in event for key in ("sequence", "time", "event_type")):
            raise EvidenceProblem(f"Event record {index} lacks sequence, time or event_type")
        if type(event["time"]) not in (int, float) or not math.isfinite(event["time"]):
            raise EvidenceProblem(f"Event record {index} has an invalid time")
        if event["event_type"] not in {"return_release", "assignment", "leg_departure", "leg_arrival",
                "pickup", "plant_admission", "operation_start", "operation_complete", "inventory_return",
                "remanufactured", "horizon"}:
            raise EvidenceProblem(f"Unsupported event type at record {index}: {event['event_type']}")
        if finished:
            check_value(findings, "time.after_horizon", "No records after horizon", event, event)
            raise ReplayViolation()
        due, kind, details = expected_transition(scenario, state, phase)
        if due > scenario["horizon"]:
            due, kind, details = scenario["horizon"], "horizon", {"product_state": state["product_state"]}
        checks = [check_value(findings, "events.sequence", index, event["sequence"], event),
                  check_value(findings, "events.transition_order", kind, event["event_type"], event),
                  check_value(findings, "events.transition_time", due, event["time"], event)]
        for key, expected in details.items():
            if key not in event:
                raise EvidenceProblem(f"Event sequence {event['sequence']} lacks {key}")
            rule = ("workstation.identity" if key == "workstation_id" else
                    "workstation.duration" if key == "end_time" and kind.startswith("operation_") else
                    f"{kind}.{key}")
            checks.append(check_value(findings, rule, expected, event[key], event,
                                       product_id=scenario["product"]["id"]))
        extra = set(event) - set(details) - {"sequence", "time", "event_type"}
        if extra:
            raise EvidenceProblem(f"Unsupported event fields at sequence {index}: {sorted(extra)}")
        if not all(checks):
            raise ReplayViolation()
        if kind == "horizon":
            state["time"] = scenario["horizon"]
            if state["active_leg"] is not None:
                leg = state["active_leg"]
                state["vehicle_distance"] += (state["time"] - leg["departure_time"]) * scenario["vehicle"]["speed"]
            if state["inventory_entered_at"] is not None:
                state["inventory_wait"] += state["time"] - state["inventory_entered_at"]
            finished = True
        else:
            phase = apply_transition(scenario, state, phase, event)
        if index == len(events) or events[index].get("time") != event["time"]:
            check_conservation(scenario, state, event, findings)
    if not finished:
        raise EvidenceProblem("Missing horizon record or required journey transitions")
    return state


def audit_dispatch(scenario: Record, events: list[Record], decisions: list[Record], findings: list[Record]) -> None:
    """Check actual policy-boundary input, fixed-plan decision and its event anchor."""
    assignments = [event for event in events if event.get("event_type") == "assignment"]
    if len(decisions) != 1 or len(assignments) != 1:
        raise EvidenceProblem("Expected one dispatch record and one assignment event")
    assignment = assignments[0]
    product, vehicle = scenario["product"], scenario["vehicle"]
    expected = {"time": scenario["trip"]["departure_time"], "trigger": "static_departure",
        "last_event_sequence": assignment["sequence"] - 1,
        "policy_input": {"requests": [{"product_id": product["id"], "collection_point": product["origin"],
            "release_time": product["release_time"], "disposition": product["disposition"]}],
            "vehicle_id": vehicle["id"], "stops": scenario["trip"]["stops"],
            "departure_time": scenario["trip"]["departure_time"], "destination": scenario["plant"]["id"]},
        "decision": {"assigned_product": product["id"], "stops": scenario["trip"]["stops"]},
        "candidates": [], "tie_break": "Only one product and one configured trip"}
    for key, value in expected.items():
        if key not in decisions[0]:
            raise EvidenceProblem(f"Dispatch record lacks {key}")
        check_value(findings, f"policy.{key}", value, decisions[0][key], assignment,
                    vehicle_id=vehicle["id"], product_id=product["id"])
    if set(decisions[0]) != set(expected):
        raise EvidenceProblem("Unsupported fields in dispatch record")


def reconcile_exports(state: Record, evidence: dict[str, Any], findings: list[Record]) -> Record:
    """Compute totals from replay and compare every exported summary and final-state field."""
    summary = {"generated": int(state["product_state"] != "unreleased"),
        "remanufactured": int(state["product_state"] == "remanufactured"), "recycled": 0,
        "unfinished": int(state["product_state"] not in ("unreleased", "remanufactured")),
        "vehicle_distance": state["vehicle_distance"], "plant_inventory_wait": state["inventory_wait"]}
    state["summary"] = summary
    for filename, expected in (("summary.json", summary), ("final_state.json", state)):
        if filename not in evidence:
            continue
        for key, value in expected.items():
            if key not in evidence[filename]:
                raise EvidenceProblem(f"{filename} lacks {key}")
            check_value(findings, f"results.{filename}.{key}", value, evidence[filename][key],
                        file=filename, product_id=state["product_id"])
        if set(evidence[filename]) != set(expected):
            raise EvidenceProblem(f"Unsupported fields in {filename}")
    return summary


def write_reports(run_directory: Path, report: Record) -> None:
    """Write only verification.json and verification.txt beside the saved evidence."""
    (run_directory / "verification.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = [f"{report['status']} (exit {report['exit_code']})", report["scope"],
             f"Tolerance: absolute and relative {TOLERANCE}",
             f"Recalculated summary: {json.dumps(report['recalculated_summary'])}"]
    for finding in report["findings"]:
        lines.append(f"{finding['category'].upper()} {finding['rule']} "
            f"time={finding['time']} sequence={finding['event_sequence']} ids={finding['ids']} "
            f"expected={finding['expected']!r} observed={finding['observed']!r}")
    lines.append("Unchecked scope: " + "; ".join(report["unchecked_scope"]))
    (run_directory / "verification.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_inventory_inputs(scenario: Record) -> None:
    """Independently recognize and validate the version-two fixed-trip inventory scope."""
    if scenario['format_version'] != 2 or scenario['policy'] not in ('static_quality_unaware', 'static_quality_aware'):
        raise EvidenceProblem('Expected inventory format 2 and supported static policy')
    plant, point = scenario['plant'], scenario['collection_point']
    capacity = plant['inventory_capacity']
    if capacity is not None and (type(capacity) is not int or capacity <= 0):
        raise EvidenceProblem('Capacity must be a positive integer or null')
    ids = [plant['id'], point['id']]
    numbers = [scenario['horizon']]
    ids.extend(recycler['id'] for recycler in scenario['recycling_facilities'])
    for site in (plant, point, *scenario['recycling_facilities']):
        if len(site['coordinates']) != 2:
            raise EvidenceProblem('Expected two-dimensional coordinates')
        numbers.extend(site['coordinates'])
    operations = set()
    station_ids = set()
    for station in plant['workstations']:
        if station['id'] in station_ids or station['operation_id'] in operations:
            raise EvidenceProblem('Duplicate workstation or operation')
        station_ids.add(station['id'])
        operations.add(station['operation_id'])
        if not isinstance(station['id'], str) or not station['id'] or station['duration'] <= 0:
            raise EvidenceProblem('Invalid workstation ID/duration')
        numbers.append(station['duration'])
        previous = -math.inf
        for start, end in station['windows']:
            if not 0 <= start < end or start < previous:
                raise EvidenceProblem('Invalid inventory calendar')
            numbers.extend((start, end))
            previous = end
    vehicle_ids = set()
    for vehicle in scenario['vehicles']:
        ids.append(vehicle['id'])
        vehicle_ids.add(vehicle['id'])
        if vehicle['initial_site'] not in {plant['id'], *(site['id'] for site in scenario['recycling_facilities'])}:
            raise EvidenceProblem('Expected a configured facility start')
        maximum = vehicle['maximum_range']
        if maximum is not None:
            if type(maximum) not in (int, float) or not math.isfinite(maximum) or maximum <= 0:
                raise EvidenceProblem('Expected positive finite range or null')
            numbers.append(maximum)
        if vehicle['speed'] <= 0 or type(vehicle['payload_capacity']) is not int or vehicle['payload_capacity'] <= 0:
            raise EvidenceProblem('Invalid vehicle speed/payload')
        numbers.append(vehicle['speed'])
    for product in scenario['products']:
        ids.append(product['id'])
        if product['origin'] != point['id']:
            raise EvidenceProblem('Invalid origin')
        if product['disposition'] == 'remanufacture':
            if not product['operations'] or any(operation not in operations for operation in product['operations']):
                raise EvidenceProblem('Incompatible operation route')
        elif product['disposition'] != 'recycle' or product['operations'] or not scenario['recycling_facilities']:
            raise EvidenceProblem('Recycling needs a facility and an empty operation route')
        if not 0 <= product['release_time'] <= scenario['horizon']:
            raise EvidenceProblem('Release outside horizon')
        numbers.append(product['release_time'])
    if len(set(ids)) != len(ids) or not all(isinstance(i, str) and i for i in ids):
        raise EvidenceProblem('Expected unique entity IDs')
    used = set()
    for trip in scenario['trips']:
        if trip['vehicle_id'] not in vehicle_ids or trip['vehicle_id'] in used:
            raise EvidenceProblem('Expected at most one fixed trip per AV')
        used.add(trip['vehicle_id'])
        facilities = {plant['id'], *(site['id'] for site in scenario['recycling_facilities'])}
        if not trip['stops'] or trip['stops'][-1] not in facilities or trip['stops'].count(point['id']) > 1 or any(stop not in facilities | {point['id']} for stop in trip['stops']) or not 0 <= trip['departure_time'] <= scenario['horizon']:
            raise EvidenceProblem('Invalid fixed trip')
        numbers.append(trip['departure_time'])
    if scenario['horizon'] < 0 or not all(type(n) in (int, float) and math.isfinite(n) for n in numbers):
        raise EvidenceProblem('Expected finite inputs')
    scenario['seed_provenance']


def audit_weighted_index(generator: random.Random, weights: list[float]) -> int:
    """Independently draw a weighted index for the documented one-uniform CDF algorithm."""
    threshold = generator.random() * sum(weights)
    for index, weight in enumerate(weights):
        if threshold < weight:
            return index
        threshold -= weight
    return max(index for index, weight in enumerate(weights) if weight > 0)


def audit_seeded_generation(scenario: Record, findings: list[Record]) -> None:
    """Independently reconstruct seed streams, batches and quality samples from saved generation inputs."""
    if scenario.get('seed_provenance', {}).get('source') == 'seeded_network_configuration':
        audit_network_generation(scenario, findings)
        return
    if not any(key in scenario for key in ('generation', 'quality_classes', 'arrival_batches')) and scenario.get('seed_provenance', {}).get('source') != 'seeded_configuration':
        return
    if not all(key in scenario for key in ('generation', 'quality_classes', 'arrival_batches', 'seed_provenance', 'products')):
        raise EvidenceProblem('Missing generation settings, quality classes, batches, provenance or products')
    provenance = scenario['seed_provenance']
    seed = provenance['master_seed']
    if type(seed) is not int or provenance['generation_version'] != 1:
        raise EvidenceProblem('Unsupported generation seed/version')
    expected_seeds = {name: int(hashlib.sha256(f'thesis-return-v1|{seed}|{name}'.encode('utf-8')).hexdigest(), 16)
                      for name in ('arrivals_batches', 'qualities')}
    check_value(findings, 'generation.seed_provenance', {'source': 'seeded_configuration', 'master_seed': seed,
        'generation_version': 1, 'algorithm': 'Random.random/exponential-inverse-cdf/weighted-cdf-v1',
        'stream_seeds': expected_seeds}, provenance)
    settings = scenario['generation']
    rate, beginning, end = (settings[key] for key in ('arrival_rate', 'start_time', 'end_time'))
    if any(type(value) not in (int, float) for value in (rate, beginning, end)) or rate < 0 or not 0 <= beginning <= end <= scenario['horizon']:
        raise EvidenceProblem('Invalid arrival generation settings')
    distribution = settings['batch_size']
    if type(distribution) is int:
        if distribution <= 0:
            raise EvidenceProblem('Invalid fixed batch size')
    elif (not isinstance(distribution, dict) or not distribution['values']
            or len(distribution['values']) != len(distribution['weights'])
            or any(type(value) is not int or value <= 0 for value in distribution['values'])
            or any(type(weight) not in (int, float) or weight < 0 for weight in distribution['weights'])
            or not 0 < sum(distribution['weights']) < math.inf):
        raise EvidenceProblem('Invalid batch size distribution')
    qualities = scenario['quality_classes']
    identifiers = [quality['id'] for quality in qualities]
    if len(set(identifiers)) != len(identifiers) or not all(isinstance(identifier, str) and identifier for identifier in identifiers):
        raise EvidenceProblem('Invalid quality identifiers')
    overrides = scenario['collection_point'].get('quality_weights', {})
    if set(overrides) - set(identifiers):
        raise EvidenceProblem('Unknown quality weight override')
    operation_ids = {station['operation_id'] for station in scenario['plant']['workstations']}
    for quality in qualities:
        weights = [quality['weight'], overrides.get(quality['id'], quality['weight'])]
        if any(type(weight) not in (int, float) or weight < 0 for weight in weights):
            raise EvidenceProblem('Invalid quality weights')
        if quality['disposition'] == 'recycle':
            if quality['operations']:
                raise EvidenceProblem('Recycling class contains processing operations')
        elif quality['disposition'] != 'remanufacture' or not quality['operations'] or (
                weights[-1] > 0 and any(operation not in operation_ids for operation in quality['operations'])):
            raise EvidenceProblem('Unsupported remanufacturing quality route')
    effective = sorted((quality for quality in qualities
        if scenario['recycling_facilities'] or quality['disposition'] == 'remanufacture'), key=lambda quality: quality['id'])
    weights = [overrides.get(quality['id'], quality['weight']) for quality in effective]
    total = sum(weights)
    if not math.isfinite(total) or (rate > 0 and end > beginning and total <= 0):
        raise EvidenceProblem('No positive effective quality mass')
    probabilities = [weight / total for weight in weights] if total else []
    arrivals = random.Random(expected_seeds['arrivals_batches'])
    quality_stream = random.Random(expected_seeds['qualities'])
    batches, products = [], []
    time = beginning
    if rate > 0:
        while True:
            next_time = time - math.log1p(-arrivals.random()) / rate
            if next_time <= time:
                raise EvidenceProblem('Generated interval cannot advance the clock')
            time = next_time
            if time > end:
                break
            size = distribution if type(distribution) is int else distribution['values'][audit_weighted_index(arrivals, distribution['weights'])]
            product_ids = [f'P{index:06d}' for index in range(len(products) + 1, len(products) + size + 1)]
            batch_id = f'B{len(batches) + 1:06d}'
            batches.append({'id': batch_id, 'release_time': time, 'batch_size': size, 'product_ids': product_ids})
            for product_id in product_ids:
                quality = effective[audit_weighted_index(quality_stream, probabilities)]
                products.append({'id': product_id, 'batch_id': batch_id, 'origin': scenario['collection_point']['id'],
                    'release_time': time, 'quality_class': quality['id'], 'disposition': quality['disposition'],
                    'operations': quality['operations']})
    check_value(findings, 'generation.arrival_batches', batches, scenario['arrival_batches'])
    check_value(findings, 'generation.products', products, scenario['products'])


def audit_generated_coordinates(scenario: Record, findings: list[Record]) -> Record:
    """Independently reconstruct each site's uniform XY coordinates and return the expected geometry stream seeds."""
    settings = scenario['coordinate_generation']
    if not isinstance(settings, dict):
        raise EvidenceProblem('Missing coordinate bounds')
    for name in ('x_range', 'y_range'):
        bounds = settings.get(name)
        if (not isinstance(bounds, list) or len(bounds) != 2 or
                any(type(value) not in (int, float) or not math.isfinite(value) for value in bounds) or
                bounds[0] > bounds[1] or not math.isfinite(bounds[1] - bounds[0])):
            raise EvidenceProblem(f'Invalid coordinate {name}')
    seed = scenario['seed_provenance']['master_seed']
    seeds = {}
    for site_type, group in (('collection_point', 'collection_points'), ('plant', 'plants'), ('recycler', 'recycling_facilities')):
        seeds[site_type] = {}
        for site in scenario[group]:
            encoded = json.dumps(['coordinates-v1', site_type, site['id']], separators=(',', ':'), ensure_ascii=True)
            site_seed = int(hashlib.sha256(f'thesis-return-v1|{seed}|{encoded}'.encode('utf-8')).hexdigest(), 16)
            seeds[site_type][site['id']] = site_seed
            generator = random.Random(site_seed)
            expected = [generator.uniform(*settings['x_range']), generator.uniform(*settings['y_range'])]
            check_value(findings, 'generation.site_coordinates', expected, site['coordinates'], site_type=site_type, site_id=site['id'])
    return seeds


def audit_network_generation(scenario: Record, findings: list[Record]) -> None:
    """Independently audit per-site return/quality streams and seeded workstation windows without simulator imports."""
    provenance = scenario['seed_provenance']
    seed = provenance['master_seed']
    if scenario['format_version'] != 3 or type(seed) is not int or provenance['generation_version'] != 2:
        raise EvidenceProblem('Unsupported seeded network format, seed or generation version')
    point_seeds = {}
    seeds: Record = {'returns': {}, 'qualities': {}, 'workstation_schedules': {}}
    for point in sorted(scenario['collection_points'], key=lambda item: item['id']):
        identifier = point['id']
        encoded = json.dumps(['network-v2', 'collection', identifier], separators=(',', ':'), ensure_ascii=True)
        point_seed = int(hashlib.sha256(f'thesis-return-v1|{seed}|{encoded}'.encode('utf-8')).hexdigest(), 16)
        point_seeds[identifier] = point_seed
        local_seeds = {name: int(hashlib.sha256(f'thesis-return-v1|{point_seed}|{name}'.encode('utf-8')).hexdigest(), 16)
                       for name in ('arrivals_batches', 'qualities')}
        seeds['returns'][identifier], seeds['qualities'][identifier] = local_seeds['arrivals_batches'], local_seeds['qualities']
        prefix = identifier + ':'
        proxy = copy.deepcopy(scenario)
        proxy.update(format_version=2, plant=scenario['plants'][0], collection_point=point, generation=point['generation'],
            seed_provenance={'source': 'seeded_configuration', 'master_seed': point_seed, 'generation_version': 1,
                'algorithm': 'Random.random/exponential-inverse-cdf/weighted-cdf-v1', 'stream_seeds': local_seeds})
        proxy['products'] = [{**product, 'id': product['id'].removeprefix(prefix),
            'batch_id': product['batch_id'].removeprefix(prefix)} for product in scenario['products'] if product['origin'] == identifier]
        proxy['arrival_batches'] = [{'id': batch['id'].removeprefix(prefix), 'release_time': batch['release_time'],
            'batch_size': batch['batch_size'], 'product_ids': [product_id.removeprefix(prefix) for product_id in batch['product_ids']]}
            for batch in scenario['arrival_batches'] if batch['origin'] == identifier]
        audit_seeded_generation(proxy, findings)
        check_value(findings, 'generation.product_namespace', True,
            all(product['id'].startswith(prefix) and product['batch_id'].startswith(prefix)
                for product in scenario['products'] if product['origin'] == identifier))
        check_value(findings, 'generation.batch_namespace', True,
            all(batch['id'].startswith(prefix) and all(product_id.startswith(prefix) for product_id in batch['product_ids'])
                for batch in scenario['arrival_batches'] if batch['origin'] == identifier))
    known_points = {point['id'] for point in scenario['collection_points']}
    check_value(findings, 'generation.batch_origins', True, all(batch['origin'] in known_points for batch in scenario['arrival_batches']))
    for field in ('products', 'arrival_batches'):
        check_value(findings, f'generation.{field}_order', sorted(scenario[field],
            key=lambda item: (item['release_time'], item['origin'], item['id'])), scenario[field])
    for plant in scenario['plants']:
        station_seeds = {}
        for station in plant['workstations']:
            if 'window_generation' not in station:
                continue
            settings = station['window_generation']
            beginning, end = settings['start_time'], settings['end_time']
            if not 0 <= beginning <= end <= scenario['horizon']:
                raise EvidenceProblem('Invalid generated schedule time bounds')
            for name in ('initial_delay', 'open_duration', 'closed_duration'):
                bounds = settings[name]
                if (not isinstance(bounds, list) or len(bounds) != 2 or
                        any(type(value) not in (int, float) or not math.isfinite(value) for value in bounds) or
                        not 0 <= bounds[0] <= bounds[1] or (name != 'initial_delay' and bounds[0] <= 0)):
                    raise EvidenceProblem(f'Invalid generated schedule {name}')
            encoded = json.dumps(['network-v2', 'schedule', plant['id'], station['id']], separators=(',', ':'), ensure_ascii=True)
            station_seed = int(hashlib.sha256(f'thesis-return-v1|{seed}|{encoded}'.encode('utf-8')).hexdigest(), 16)
            station_seeds[station['id']] = station_seed
            generator = random.Random(station_seed)
            windows = []
            opening = beginning + generator.uniform(*settings['initial_delay'])
            while opening < end:
                closing = min(end, opening + generator.uniform(*settings['open_duration']))
                if closing <= opening:
                    raise EvidenceProblem('Generated opening does not advance the clock')
                windows.append([opening, closing])
                if closing == end:
                    break
                opening = closing + generator.uniform(*settings['closed_duration'])
                if opening <= closing:
                    raise EvidenceProblem('Generated closure does not advance the clock')
            check_value(findings, 'generation.workstation_windows', windows, station['windows'],
                        plant_id=plant['id'], workstation_id=station['id'])
        if station_seeds:
            seeds['workstation_schedules'][plant['id']] = station_seeds
    if 'coordinate_generation' in scenario:
        seeds['coordinates'] = audit_generated_coordinates(scenario, findings)
    expected_provenance = {'source': 'seeded_network_configuration', 'master_seed': seed, 'generation_version': 2,
        'algorithm': 'site-streams-v2/exponential-inverse-cdf/weighted-cdf/alternating-uniform-windows-v1',
        'point_seeds': point_seeds, 'stream_seeds': seeds}
    if 'coordinate_generation' in scenario:
        expected_provenance['coordinate_algorithm'] = 'site-type-id/uniform-xy-v1'
    # Seeds identify random streams exactly; physical-value tolerances must not hide integer changes.
    if expected_provenance != provenance:
        add_finding(findings, 'violation', 'generation.network_seed_provenance', expected_provenance, provenance)


def audit_forecast_context(scenario: Record, state: Record, departed_legs: dict[str, int]) -> Record:
    """Independently form the released-only observation snapshot supplied to the aware policy."""
    released = {product_id for product_id, product in state['products'].items() if product['state'] != 'unreleased'}
    snapshot = copy.deepcopy(state)
    snapshot['products'] = {product_id: product for product_id, product in snapshot['products'].items() if product_id in released}
    physical = {key: copy.deepcopy(scenario[key]) for key in
                ('horizon', 'plant', 'collection_point', 'recycling_facilities', 'vehicles', 'trips')}
    physical['products'] = [{key: copy.deepcopy(product[key]) for key in
        ('id', 'origin', 'release_time', 'disposition', 'operations')} for product in scenario['products'] if product['id'] in released]
    return {'scenario': physical, 'state': snapshot, 'departed_legs': dict(departed_legs)}


def audit_delivery_time(scenario: Record, state: Record, trip: Record, departed_legs: int = 0) -> float:
    """Independently calculate final arrival through the remaining fixed stops."""
    vehicle = state['vehicles'][trip['vehicle_id']]
    speed = next(vehicle['speed'] for vehicle in scenario['vehicles'] if vehicle['id'] == trip['vehicle_id'])
    sites = {site['id']: site for site in [scenario['plant'], scenario['collection_point'], *scenario['recycling_facilities']]}
    leg = vehicle['active_leg']
    time = leg['arrival_time'] if leg is not None else state['time']
    position = leg['target_coordinates'] if leg is not None else sites[vehicle['location']]['coordinates']
    for stop in trip['stops'][departed_legs:]:
        target = sites[stop]['coordinates']
        time += math.hypot(target[0] - position[0], target[1] - position[1]) / speed
        position = target
    return time


def audit_candidate_forecast(context: Record, trip: Record, candidate_id: str, selected: list[str]) -> Record:
    """Project the plant independently with dictionary transitions and a chronological sweep."""
    scenario, state = context['scenario'], copy.deepcopy(context['state'])
    arrival = audit_delivery_time(scenario, state, trip)
    result = {'predicted_arrival': arrival, 'predicted_admission': None,
              'predicted_first_start': None, 'predicted_inventory_wait': None}
    if trip['stops'][-1] != scenario['plant']['id']:
        result['predicted_admission'] = arrival if arrival <= scenario['horizon'] else None
        return result
    definitions = {product['id']: product for product in scenario['products']}
    deliveries = []
    for committed in scenario['trips']:
        vehicle_id = committed['vehicle_id']
        if vehicle_id == trip['vehicle_id'] or committed['stops'][-1] != scenario['plant']['id'] or vehicle_id in state['waiting_vehicles']:
            continue
        carried = sorted(product_id for product_id, product in state['products'].items()
            if product['assigned_vehicle'] == vehicle_id and product['state'] in ('at_collection_point', 'onboard_av'))
        if carried:
            deliveries.append((audit_delivery_time(scenario, state, committed, context['departed_legs'][vehicle_id]), vehicle_id, carried))
    deliveries.append((arrival, trip['vehicle_id'], sorted([*selected, candidate_id])))
    calendar = []
    for station in scenario['plant']['workstations']:
        intervals = []
        for opening, closing in station['windows']:
            if intervals and intervals[-1][1] == opening:
                intervals[-1][1] = closing
            else:
                intervals.append([opening, closing])
        calendar.append((station, intervals))
    calendar.sort(key=lambda item: item[0]['id'])
    while state['time'] <= scenario['horizon']:
        time = state['time']
        # Complete all due active work before arrivals and instantaneous settling.
        for station_id in sorted(state['workstations']):
            operation = state['workstations'][station_id]
            if operation is None or operation['end_time'] > time:
                continue
            product_id = operation['product_id']
            product = state['products'][product_id]
            product['next_operation'] += 1
            state['workstations'][station_id] = None
            if product['next_operation'] < len(definitions[product_id]['operations']):
                if product_id in state['held_slots']:
                    state['held_slots'].remove(product_id)
                state['inventory'][product_id] = time
        due = sorted((delivery for delivery in deliveries if delivery[0] <= time), key=lambda delivery: delivery[1])
        for _, vehicle_id, load in due:
            state['vehicles'][vehicle_id]['load'] = list(load)
            state['waiting_vehicles'][vehicle_id] = time
        deliveries = [delivery for delivery in deliveries if delivery[0] > time]
        while True:
            changed = False
            for vehicle_id in sorted(state['waiting_vehicles'], key=lambda vehicle_id: (state['waiting_vehicles'][vehicle_id], vehicle_id)):
                load = state['vehicles'][vehicle_id]['load']
                capacity = scenario['plant']['inventory_capacity']
                while load and (capacity is None or len(state['inventory']) + len(state['held_slots']) < capacity):
                    product_id = load.pop(0)
                    state['inventory'][product_id] = time
                    if product_id == candidate_id:
                        result['predicted_admission'] = time
                    changed = True
                if not load:
                    del state['waiting_vehicles'][vehicle_id]
            started = False
            for station, intervals in calendar:
                if state['workstations'][station['id']] is not None:
                    continue
                window = next((interval for interval in intervals if interval[0] <= time < interval[1]
                               and time + station['duration'] <= interval[1]), None)
                eligible = sorted((product_id for product_id in state['inventory']
                    if definitions[product_id]['operations'][state['products'][product_id]['next_operation']] == station['operation_id']),
                    key=lambda product_id: (state['inventory'][product_id], product_id))
                if window is None or not eligible:
                    continue
                product_id = eligible[0]
                state['inventory'].pop(product_id)
                operation_index = state['products'][product_id]['next_operation']
                state['workstations'][station['id']] = {'product_id': product_id,
                    'end_time': time + station['duration']}
                if scenario['plant']['inventory_capacity'] is not None and operation_index + 1 < len(definitions[product_id]['operations']):
                    state['held_slots'].append(product_id)
                if product_id == candidate_id:
                    result.update(predicted_first_start=time, predicted_inventory_wait=time - result['predicted_admission'])
                    return result
                changed, started = True, True
                break
            if not changed and not started:
                break
        future = [delivery[0] for delivery in deliveries]
        future.extend(operation['end_time'] for operation in state['workstations'].values() if operation is not None)
        future.extend(opening for _, intervals in calendar for opening, _ in intervals if opening > time)
        if not future or min(future) > scenario['horizon']:
            return result
        state['time'] = min(future)
    return result


def audit_quality_selection(policy_input: Record) -> tuple[list[str], list[Record], str]:
    """Independently enumerate conditional payload-round candidates and apply the documented score tuple."""
    context = policy_input['forecast_context']
    scenario = context['scenario']
    trip = next(trip for trip in scenario['trips'] if trip['vehicle_id'] == policy_input['vehicle_id'])
    capacity = next(vehicle['payload_capacity'] for vehicle in scenario['vehicles'] if vehicle['id'] == trip['vehicle_id'])
    disposition = 'remanufacture' if trip['stops'][-1] == scenario['plant']['id'] else 'recycle'
    remaining = [request['product_id'] for request in policy_input['requests'] if request['disposition'] == disposition]
    selected, audit = [], []
    if scenario['collection_point']['id'] not in trip['stops']:
        return selected, audit, 'Fixed trip has no pickup stop'
    for round_index in range(min(capacity, len(remaining))):
        candidates = []
        for product_id in remaining:
            prediction = audit_candidate_forecast(context, trip, product_id, selected)
            candidate = {'selection_round': round_index, 'product_id': product_id, 'destination': trip['stops'][-1],
                'pickup_stop_index': trip['stops'].index(scenario['collection_point']['id']),
                'delivery_stop_index': len(trip['stops']) - 1, 'added_distance': 0.0, **prediction}
            feasible_start = prediction['predicted_first_start'] is not None
            candidate['score'] = [int(not feasible_start), prediction['predicted_inventory_wait'] if feasible_start else 0.0,
                0.0, product_id, candidate['pickup_stop_index'], candidate['delivery_stop_index']]
            candidates.append(candidate)
        winner = min(candidates, key=lambda candidate: candidate['score'])['product_id']
        for candidate in candidates:
            candidate['selected'] = candidate['product_id'] == winner
        audit.extend(candidates)
        selected.append(winner)
        remaining.remove(winner)
    return selected, audit, 'Prefer start within horizon, then admission-to-first-start wait, added distance, product ID and stop indexes'


def audit_aware_dispatch_records(expected_records: list[Record], actual_records: list[Record], findings: list[Record]) -> None:
    """Compare independently reconstructed candidates and selections with precise decision/event diagnostics."""
    check_value(findings, 'policy.decision_count', len(expected_records), len(actual_records))
    for expected, actual in zip(expected_records, actual_records):
        event = {'time': expected['time'], 'sequence': expected['last_event_sequence'] + 1,
                 'vehicle_id': expected['policy_input']['vehicle_id']}
        for field in ('policy_input', 'candidates', 'decision', 'selection_reason', 'tie_break'):
            if field not in actual:
                raise EvidenceProblem(f'Aware dispatch record lacks {field}')
        for field in ('policy_input', 'selection_reason', 'tie_break'):
            check_value(findings, f'policy.{field}', expected[field], actual[field], event)
        if 'assigned_products' not in actual['decision']:
            raise EvidenceProblem('Aware dispatch selection lacks assigned_products')
        check_value(findings, 'policy.selection', expected['decision']['assigned_products'], actual['decision']['assigned_products'], event)
        expected_keys = [(candidate['selection_round'], candidate['product_id']) for candidate in expected['candidates']]
        actual_keys = [(candidate['selection_round'], candidate['product_id']) for candidate in actual['candidates']]
        check_value(findings, 'policy.candidate_coverage', expected_keys, actual_keys, event)
        for wanted, observed in zip(expected['candidates'], actual['candidates']):
            for field, value in wanted.items():
                if field not in observed:
                    raise EvidenceProblem(f"Candidate {wanted['product_id']} lacks {field}")
                check_value(findings, f'policy.candidate.{field}', value, observed[field], event,
                            product_id=wanted['product_id'], selection_round=wanted['selection_round'])


class InventoryAudit:
    """Reconstruct version-two physical state from observed events, independently of simulation."""

    def __init__(self, scenario: Record, findings: list[Record]) -> None:
        """Initialize empty replay state, input lookups and mandatory immediate follow-ups."""
        self.scenario = scenario
        self.findings = findings
        self.products = {p['id']: p for p in scenario['products']}
        self.vehicles = {v['id']: v for v in scenario['vehicles']}
        self.stations = {s['id']: s for s in scenario['plant']['workstations']}
        self.trips = {t['vehicle_id']: t for t in scenario['trips']}
        self.delivery_sites = {site['id']: site for site in [scenario['plant'], *scenario['recycling_facilities']]}
        self.route_sites = {**self.delivery_sites, scenario['collection_point']['id']: scenario['collection_point']}
        self.range_plans: dict[str, Record] = {}
        self.state = {'time': 0, 'products': {product_id: {'state': 'unreleased', 'location': None,
            'next_operation': 0, 'assigned_vehicle': None, 'planned_destination': None,
            'actual_destination': None, 'milestones': {}, 'history': [], 'inventory_wait': 0.0} for product_id in self.products},
            'vehicles': {vehicle_id: {'status': 'idle', 'location': v['initial_site'], 'load': [],
                'active_leg': None, 'distance': 0.0, 'waiting_since': None,
                'waiting_to_unload': 0.0, 'wait_logged': False} for vehicle_id, v in self.vehicles.items()},
            'workstations': dict.fromkeys(self.stations), 'inventory': {}, 'held_slots': [], 'waiting_vehicles': {}}
        for vehicle_id, definition in self.vehicles.items():
            if definition['maximum_range'] is not None:
                self.state['vehicles'][vehicle_id].update(remaining_range=definition['maximum_range'],
                    position=list(self.route_sites[definition['initial_site']]['coordinates']), leg_distance_accounted=0.0)
        self.required: list[tuple[str, str, str]] = []
        self.dispatched: set[str] = set()
        self.leg_counts = dict.fromkeys(self.vehicles, 0)
        self.expected_decisions: list[Record] = []
        self.finished = False

    def require(self, rule: str, expected: Any, observed: Any, event: Record) -> None:
        """Record a proved violation and stop replay if an independent comparison fails."""
        if not check_value(self.findings, rule, expected, observed, event):
            raise ReplayViolation()

    def match_fields(self, event: Record, details: Record) -> None:
        """Check complete version-two event payload against independently calculated values."""
        for field, expected in details.items():
            if field not in event:
                raise EvidenceProblem(f"Event {event['sequence']} lacks {field}")
            self.require(f"{event['event_type']}.{field}", expected, event[field], event)
        if set(event) != set(details) | {'sequence', 'time', 'event_type'}:
            raise EvidenceProblem(f"Unsupported fields in event {event['sequence']}")

    def occupancy(self) -> int:
        """Count physical inventory plus held return slots without double-counting processing."""
        return len(self.state['inventory']) + len(self.state['held_slots'])

    def window_opportunity(self, station: Record, earliest: float) -> tuple[float, list[float]] | None:
        """Find the earliest full-fit window, independently coalescing continuous availability."""
        intervals = []
        for start, end in station['windows']:
            if intervals and intervals[-1][1] == start:
                intervals[-1] = (intervals[-1][0], end)
            else:
                intervals.append((start, end))
        for opening, closing in intervals:
            start = max(earliest, opening)
            if start < closing and start + station['duration'] <= closing:
                return start, [opening, closing]
        return None

    def eligible_products(self, station_id: str) -> list[str]:
        """Return inventory candidates in latest-entry-time then product-ID order."""
        return sorted((product_id for product_id in self.state['inventory'] if
            self.products[product_id]['operations'][self.state['products'][product_id]['next_operation']] == self.stations[station_id]['operation_id']),
            key=lambda product_id: (self.state['inventory'][product_id], product_id))

    def waiting_order(self) -> list[str]:
        """Return AV admission priority by original plant arrival time then vehicle ID."""
        return sorted(self.state['waiting_vehicles'], key=lambda vehicle_id: (self.state['waiting_vehicles'][vehicle_id], vehicle_id))

    def pending_deadlines(self) -> list[float]:
        """Return earliest required release/trip/travel/processing/start/admission times from replay state."""
        now = self.state['time']
        deadlines = [p['release_time'] for product_id, p in self.products.items() if self.state['products'][product_id]['state'] == 'unreleased']
        deadlines.extend(t['departure_time'] for vehicle_id, t in self.trips.items() if vehicle_id not in self.dispatched)
        deadlines.extend(v['active_leg']['arrival_time'] for v in self.state['vehicles'].values() if v['active_leg'] is not None)
        deadlines.extend(op['end_time'] for op in self.state['workstations'].values() if op is not None)
        for workstation_id, active in self.state['workstations'].items():
            if active is None and self.eligible_products(workstation_id):
                opportunity = self.window_opportunity(self.stations[workstation_id], now)
                if opportunity:
                    deadlines.append(opportunity[0])
        capacity = self.scenario['plant']['inventory_capacity']
        if self.waiting_order() and (capacity is None or self.occupancy() < capacity):
            deadlines.append(now)
        return deadlines

    def check_clock_boundary(self, event: Record) -> None:
        """Prevent skipped deadlines, unsettled actions and conservation errors when time advances."""
        now = self.state['time']
        if event['time'] <= now:
            return
        self.require('events.immediate_followup', [], self.required, event)
        deadlines = self.pending_deadlines()
        if deadlines:
            self.require('events.no_skipped_action', True, event['time'] <= min(deadlines) + TOLERANCE, event)
        self.check_physical_state(event)

    def check_physical_state(self, event: Record) -> None:
        """Check exclusive physical membership, held-slot ownership and logged waiting state after settling."""
        for product_id, product in self.state['products'].items():
            physical = int(product['state'] == 'at_collection_point') + int(product_id in self.state['inventory'])
            physical += sum(product_id in v['load'] for v in self.state['vehicles'].values())
            physical += sum(op is not None and op['product_id'] == product_id for op in self.state['workstations'].values())
            physical += int(product['state'] in ('remanufactured', 'recycled'))
            self.require('product.conservation', int(product['state'] != 'unreleased'), physical, event)

        expected_holds = sorted(op['product_id'] for op in self.state['workstations'].values()
            if op is not None and op['operation_index'] + 1 < len(self.products[op['product_id']]['operations']))
        if self.scenario['plant']['inventory_capacity'] is None:
            expected_holds = []
        self.require('inventory.hold_ownership', expected_holds, sorted(self.state['held_slots']), event)
        self.require('inventory.wait_evidence', [], sorted(vehicle_id for vehicle_id in self.state['waiting_vehicles']
            if not self.state['vehicles'][vehicle_id]['wait_logged']), event)

    def evaluate_range_plan(self, vehicle_id: str, start_index: int = 0) -> Record:
        """Independently calculate every remaining segment's required distance and available allowance."""
        maximum = self.vehicles[vehicle_id]['maximum_range']
        if maximum is None:
            return {'feasible': True, 'segments': []}
        vehicle = self.state['vehicles'][vehicle_id]
        coordinates = vehicle['position']
        allowance = vehicle['remaining_range']
        beginning = vehicle['location']
        total = 0.0
        segments = []
        for site_id in self.trips[vehicle_id]['stops'][start_index:]:
            site = self.route_sites[site_id]
            total += math.hypot(site['coordinates'][0] - coordinates[0], site['coordinates'][1] - coordinates[1])
            coordinates = site['coordinates']
            if site_id in self.delivery_sites:
                segments.append({'start': beginning, 'end': site_id, 'required_distance': total, 'available_range': allowance})
                beginning, total, allowance = site_id, 0.0, maximum
        rejection = next((segment for segment in segments if segment['required_distance'] > segment['available_range'] + TOLERANCE), None)
        result = {'feasible': rejection is None, 'segments': segments}
        if rejection is not None:
            result['rejection'] = {'reason': 'range_infeasible', **rejection}
        return result

    def advance_range_travel(self, time: float, event: Record) -> None:
        """Independently integrate finite-vehicle actual motion before checking an event at a new time."""
        for vehicle_id, definition in self.vehicles.items():
            if definition['maximum_range'] is None:
                continue
            vehicle = self.state['vehicles'][vehicle_id]
            leg = vehicle['active_leg']
            if leg is None:
                continue
            covered = min(leg['distance'], max(0, time - leg['departure_time']) * definition['speed'])
            change = covered - vehicle['leg_distance_accounted']
            self.require('range.actual_distance', True, -TOLERANCE <= change <= vehicle['remaining_range'] + TOLERANCE, event)
            vehicle['distance'] += change
            vehicle['remaining_range'] = max(0.0, vehicle['remaining_range'] - change)
            vehicle['leg_distance_accounted'] = covered
            ratio = covered / leg['distance'] if leg['distance'] else 1.0
            vehicle['position'] = [leg['origin_coordinates'][axis] + ratio *
                (leg['target_coordinates'][axis] - leg['origin_coordinates'][axis]) for axis in (0, 1)]

    def consume_range_event(self, event: Record) -> None:
        """Check a full-plan decision, rejection or mandatory facility-only arrival reset."""
        vehicle_id = event['vehicle_id']
        vehicle = self.state['vehicles'][vehicle_id]
        maximum = self.vehicles[vehicle_id]['maximum_range']
        self.require('range.limited_vehicle', True, maximum is not None, event)
        kind = event['event_type']
        if kind == 'range_plan_check':
            self.require('range.single_plan_check', False, vehicle_id in self.range_plans, event)
            self.require('range.plan_time', self.trips[vehicle_id]['departure_time'], event['time'], event)
            expected = self.evaluate_range_plan(vehicle_id)
            self.match_fields(event, {'vehicle_id': vehicle_id, **expected})
            self.range_plans[vehicle_id] = expected
        elif kind == 'trip_rejected':
            plan = self.range_plans[vehicle_id]
            self.require('range.justified_rejection', False, plan['feasible'], event)
            self.match_fields(event, {'vehicle_id': vehicle_id, **plan['rejection']})
            self.require('range.no_departure_on_rejection', ('idle', []), (vehicle['status'], vehicle['load']), event)
        else:
            self.require('range.reset_facility', True, event['facility_id'] in self.delivery_sites, event)
            self.require('range.reset_at_vehicle_site', vehicle['location'], event['facility_id'], event)
            self.match_fields(event, {'vehicle_id': vehicle_id, 'facility_id': vehicle['location'],
                'range_before': vehicle['remaining_range'], 'range_after': maximum, 'load': list(vehicle['load'])})
            vehicle['remaining_range'] = maximum

    def capture_dispatch(self, event: Record, vehicle_id: str) -> None:
        """Reconstruct the actual policy-boundary request list before a fixed trip assigns products."""
        self.require('policy.configured_trip', True, vehicle_id in self.trips, event)
        trip = self.trips[vehicle_id]
        due_vehicles = sorted(vehicle_id for vehicle_id, plan in self.trips.items()
                              if vehicle_id not in self.dispatched and values_match(plan['departure_time'], event['time']))
        self.require('policy.vehicle_id_tie', due_vehicles[0] if due_vehicles else None, vehicle_id, event)
        self.require('policy.departure_time', trip['departure_time'], event['time'], event)
        available = sorted((product_id for product_id, p in self.state['products'].items()
            if p['state'] == 'at_collection_point' and p['assigned_vehicle'] is None),
            key=lambda product_id: (self.products[product_id]['release_time'], product_id))
        disposition = 'remanufacture' if trip['stops'][-1] == self.scenario['plant']['id'] else 'recycle'
        selected = [product_id for product_id in available if self.products[product_id]['disposition'] == disposition][:self.vehicles[vehicle_id]['payload_capacity']]
        plan = {'feasible': True, 'segments': []}
        if self.vehicles[vehicle_id]['maximum_range'] is not None:
            self.require('range.plan_evidence', True, vehicle_id in self.range_plans, event)
            plan = self.range_plans[vehicle_id]
        if not plan['feasible'] or self.scenario['collection_point']['id'] not in trip['stops']:
            selected = []
        self.expected_decisions.append({'time': event['time'], 'trigger': 'static_departure',
            'last_event_sequence': event['sequence'] - 1, 'policy_input': {'vehicle_id': vehicle_id,
                'stops': trip['stops'], 'requests': [{'product_id': product_id, 'collection_point': self.products[product_id]['origin'],
                'release_time': self.products[product_id]['release_time'], 'disposition': self.products[product_id]['disposition']} for product_id in available]},
            'decision': {'assigned_products': selected, 'stops': trip['stops']}, 'tie_break': 'release_time, product_id'})
        if self.scenario['policy'] == 'static_quality_aware':
            boundary = self.expected_decisions[-1]['policy_input']
            for request in boundary['requests']:
                request['operations'] = list(self.products[request['product_id']]['operations'])
            boundary['forecast_context'] = audit_forecast_context(self.scenario, self.state, self.leg_counts)
            if plan['feasible']:
                selected, candidates, reason = audit_quality_selection(boundary)
            else:
                selected, candidates, reason = [], [], 'Fixed trip rejected by range check'
            self.expected_decisions[-1].update(candidates=candidates, selection_reason=reason,
                tie_break='product_id, pickup_stop_index, delivery_stop_index')
            self.expected_decisions[-1]['decision']['assigned_products'] = selected
        self.dispatched.add(vehicle_id)
        if not plan['feasible']:
            self.expected_decisions[-1]['decision']['rejection'] = plan['rejection']
            self.required.append(('trip_rejected', 'vehicle_id', vehicle_id))
        else:
            self.required.extend(('assignment', 'product_id', product_id) for product_id in selected)
            self.required.append(('leg_departure', 'vehicle_id', vehicle_id))

    def consume_event(self, event: Record) -> None:
        """Validate and apply one observed transition, including capacity and mandatory hold records."""
        kind, time = event['event_type'], event['time']
        plant = self.scenario['plant']
        product_id, vehicle_id, workstation_id = event.get('product_id'), event.get('vehicle_id'), event.get('workstation_id')
        if vehicle_id is not None:
            self.require('vehicle.known_identity', True, vehicle_id in self.vehicles, event)
        if product_id is not None:
            self.require('product.known_identity', True, product_id in self.products, event)
        if kind == 'recycler_acceptance':
            self.require('recycling.disposition', 'recycle', self.products[product_id]['disposition'], event)
        if kind == 'plant_admission':
            self.require('plant.disposition', 'remanufacture', self.products[product_id]['disposition'], event)
        if kind == 'range_reset' and event.get('facility_id') not in self.delivery_sites:
            self.require('range.reset_facility', sorted(self.delivery_sites), event.get('facility_id'), event)
        if kind in ('assignment', 'leg_departure', 'trip_rejected') and vehicle_id not in self.dispatched:
            self.capture_dispatch(event, vehicle_id)
        had_required = bool(self.required)
        if self.required:
            expected_kind, identity_field, identifier = self.required.pop(0)
            self.require('events.immediate_followup_type', expected_kind, kind, event)
            self.require('events.immediate_followup_identity', identifier, event.get(identity_field), event)
        # Capacity is checked against replayed stock BEFORE accepting an admission.
        if kind == 'plant_admission':
            capacity = plant['inventory_capacity']
            if capacity is not None and self.occupancy() + 1 > capacity:
                self.require('inventory.capacity', f'waiting + held <= {capacity}',
                    {'waiting_after_admission': len(self.state['inventory']) + 1,
                     'held_slots': len(self.state['held_slots']), 'attempted_occupancy': self.occupancy() + 1}, event)
            if self.state['vehicles'][vehicle_id]['wait_logged']:
                self.require('inventory.retry_required', True, had_required, event)
        if kind in ('inventory_hold', 'inventory_slot_release', 'inventory_return', 'remanufactured', 'pickup', 'recycler_acceptance', 'range_reset'):
            self.require('events.required_transition', True, had_required, event)
        if workstation_id is not None:
            self.require('workstation.known_identity', True, workstation_id in self.stations, event)
        if kind not in ('operation_complete', 'inventory_return', 'remanufactured'):
            self.require('events.completion_precedence', [],
                sorted(s for s, op in self.state['workstations'].items() if op is not None and op['end_time'] <= time), event)
        if kind in ('assignment', 'leg_departure', 'plant_admission', 'unload_wait', 'unload_retry', 'operation_start'):
            self.require('events.release_precedence', [], sorted(product_id for product_id, definition in self.products.items()
                if self.state['products'][product_id]['state'] == 'unreleased' and definition['release_time'] <= time), event)
        if kind in ('plant_admission', 'unload_wait', 'unload_retry', 'operation_start'):
            self.require('events.arrival_batch_precedence', [], sorted(vehicle_id for vehicle_id, vehicle in self.state['vehicles'].items()
                if vehicle['active_leg'] is not None and vehicle['active_leg']['arrival_time'] <= time
                and vehicle['active_leg']['departure_time'] < time), event)
        if kind in ('range_plan_check', 'trip_rejected', 'range_reset'):
            self.consume_range_event(event)
        elif kind == 'return_release':
            definition = event['product']
            product_id = definition['id']
            self.require('product.known_release', True, product_id in self.products, event)
            self.match_fields(event, {'product': self.products[product_id]})
            self.require('product.release_time', self.products[product_id]['release_time'], time, event)
            product = self.state['products'][product_id]
            self.require('product.unique_release', 'unreleased', product['state'], event)
            product.update(state='at_collection_point', location=self.products[product_id]['origin'])
            product['milestones']['release'] = time
        elif kind == 'assignment':
            product = self.state['products'][product_id]
            destination = self.trips[vehicle_id]['stops'][-1]
            self.match_fields(event, {'product_id': product_id, 'vehicle_id': vehicle_id, 'destination': destination})
            self.require('product.assignment_state', ('at_collection_point', None),
                         (product['state'], product['assigned_vehicle']), event)
            product.update(assigned_vehicle=vehicle_id, planned_destination=destination)
        elif kind == 'leg_departure':
            vehicle = self.state['vehicles'][vehicle_id]
            self.require('vehicle.not_busy', (None, 'idle'), (vehicle['active_leg'], vehicle['status']), event)
            count = self.leg_counts[vehicle_id]
            stops = self.trips[vehicle_id]['stops']
            self.require('vehicle.fixed_leg_count', True, count < len(stops), event)
            origin = self.route_sites[self.vehicles[vehicle_id]['initial_site'] if count == 0 else stops[count - 1]]
            target = self.route_sites[stops[count]]
            self.require('vehicle.departure_site', origin['id'], vehicle['location'], event)
            distance = math.hypot(target['coordinates'][0] - origin['coordinates'][0], target['coordinates'][1] - origin['coordinates'][1])
            leg = {'vehicle_id': vehicle_id, 'origin': origin['id'], 'target': target['id'],
                'origin_coordinates': origin['coordinates'], 'target_coordinates': target['coordinates'],
                'departure_time': time, 'arrival_time': time + distance / self.vehicles[vehicle_id]['speed'],
                'distance': distance, 'load': list(vehicle['load'])}
            if self.vehicles[vehicle_id]['maximum_range'] is not None:
                self.require('range.remaining_plan', True, self.evaluate_range_plan(vehicle_id, count)['feasible'], event)
                leg['remaining_range_at_departure'] = vehicle['remaining_range']
                vehicle['leg_distance_accounted'] = 0.0
            self.match_fields(event, leg)
            vehicle.update(status='travelling', active_leg=leg)
            self.leg_counts[vehicle_id] += 1
        elif kind == 'leg_arrival':
            vehicle = self.state['vehicles'][vehicle_id]
            leg = vehicle['active_leg']
            self.require('vehicle.active_arrival', True, leg is not None, event)
            details = dict(leg)
            limited = self.vehicles[vehicle_id]['maximum_range'] is not None
            if limited:
                details['remaining_range_on_arrival'] = vehicle['remaining_range']
                vehicle['leg_distance_accounted'] = 0.0
                vehicle['position'] = list(leg['target_coordinates'])
            else:
                vehicle['distance'] += leg['distance']
            self.match_fields(event, details)
            self.require('vehicle.arrival_time', leg['arrival_time'], time, event)
            vehicle.update(status='idle', location=leg['target'], active_leg=None)
            if limited and leg['target'] in self.delivery_sites:
                self.required.append(('range_reset', 'vehicle_id', vehicle_id))
            final_stop = self.leg_counts[vehicle_id] == len(self.trips[vehicle_id]['stops'])
            if leg['target'] == self.scenario['collection_point']['id']:
                assigned = sorted(product_id for product_id, product in self.state['products'].items()
                                  if product['assigned_vehicle'] == vehicle_id and product['state'] == 'at_collection_point')
                self.required.extend(('pickup', 'product_id', product_id) for product_id in assigned)
            if final_stop:
                if leg['target'] == plant['id'] and vehicle['load']:
                    self.state['waiting_vehicles'][vehicle_id] = time
                elif leg['target'] in {recycler['id'] for recycler in self.scenario['recycling_facilities']}:
                    self.required.extend(('recycler_acceptance', 'product_id', product_id) for product_id in vehicle['load'])
            else:
                self.required.append(('leg_departure', 'vehicle_id', vehicle_id))
        elif kind == 'pickup':
            product, vehicle = self.state['products'][product_id], self.state['vehicles'][vehicle_id]
            self.match_fields(event, {'product_id': product_id, 'vehicle_id': vehicle_id, 'collection_point': self.scenario['collection_point']['id']})
            self.require('product.pickup_ownership', ('at_collection_point', vehicle_id), (product['state'], product['assigned_vehicle']), event)
            self.require('vehicle.pickup_site', self.scenario['collection_point']['id'], vehicle['location'], event)
            self.require('vehicle.payload', True, len(vehicle['load']) < self.vehicles[vehicle_id]['payload_capacity'], event)
            product.update(state='onboard_av', location=vehicle_id)
            product['milestones']['pickup'] = time
            vehicle['load'].append(product_id)
        elif kind == 'recycler_acceptance':
            product, vehicle = self.state['products'][product_id], self.state['vehicles'][vehicle_id]
            recycler_id = event['recycler_id']
            self.require('recycling.configured_facility', True, recycler_id in {r['id'] for r in self.scenario['recycling_facilities']}, event)
            self.require('recycling.operation_route', [], self.products[product_id]['operations'], event)
            self.require('recycling.onboard_ownership', ('onboard_av', vehicle_id), (product['state'], product['location']), event)
            self.require('recycling.destination', recycler_id, product['planned_destination'], event)
            self.require('recycling.vehicle_location', recycler_id, vehicle['location'], event)
            self.require('recycling.payload', True, product_id in vehicle['load'], event)
            self.match_fields(event, {'product_id': product_id, 'vehicle_id': vehicle_id, 'recycler_id': recycler_id})
            vehicle['load'].remove(product_id)
            product.update(state='recycled', location=None, actual_destination=recycler_id)
            product['milestones']['recycling'] = time
        elif kind == 'unload_wait':
            self.require('inventory.registered_arrival', True, vehicle_id in self.state['waiting_vehicles'], event)
            vehicle = self.state['vehicles'][vehicle_id]
            capacity = plant['inventory_capacity']
            self.require('inventory.wait_only_when_full', True, capacity is not None and self.occupancy() == capacity, event)
            self.require('inventory.unique_wait', False, vehicle['wait_logged'], event)
            self.match_fields(event, {'vehicle_id': vehicle_id, 'arrival_time': self.state['waiting_vehicles'][vehicle_id],
                'load': vehicle['load'], 'waiting_count': len(self.state['inventory']), 'held_count': len(self.state['held_slots'])})
            vehicle.update(status='waiting_to_unload', waiting_since=time, wait_logged=True)
        elif kind in ('unload_retry', 'plant_admission'):
            self.require('inventory.registered_arrival', True, bool(self.waiting_order()), event)
            self.require('inventory.waiting_vehicle_priority', self.waiting_order()[0], vehicle_id, event)
            vehicle = self.state['vehicles'][vehicle_id]
            if kind == 'unload_retry':
                capacity = plant['inventory_capacity']
                self.require('inventory.retry_when_space', True, capacity is None or self.occupancy() < capacity, event)
                self.require('inventory.retry_after_wait', True, vehicle['wait_logged'], event)
                self.match_fields(event, {'vehicle_id': vehicle_id, 'arrival_time': self.state['waiting_vehicles'][vehicle_id]})
                self.required.append(('plant_admission', 'product_id', vehicle['load'][0]))
            else:
                self.require('inventory.onboard_before_admission', vehicle['load'][0], product_id, event)
                self.match_fields(event, {'product_id': product_id, 'vehicle_id': vehicle_id, 'plant_id': plant['id'],
                    'waiting_count': len(self.state['inventory']) + 1, 'held_count': len(self.state['held_slots'])})
                self.state['inventory'][product_id] = time
                product = self.state['products'][product_id]
                product.update(state='in_plant_inventory', location=plant['id'], actual_destination=plant['id'])
                product['milestones']['admission'] = time
                vehicle['load'].pop(0)
                if not vehicle['load']:
                    self.state['waiting_vehicles'].pop(vehicle_id)
                    if vehicle['waiting_since'] is not None:
                        vehicle['waiting_to_unload'] += time - vehicle['waiting_since']
                    vehicle.update(status='idle', waiting_since=None, wait_logged=False)
        elif kind == 'operation_start':
            self.require('workstation.idle', None, self.state['workstations'][workstation_id], event)
            capacity = plant['inventory_capacity']
            self.require('inventory.admission_before_next_start', False,
                         bool(self.waiting_order()) and (capacity is None or self.occupancy() < capacity), event)
            self.require('workstation.eligible_inventory', True, bool(self.eligible_products(workstation_id)), event)
            self.require('workstation.product_priority', self.eligible_products(workstation_id)[0], product_id, event)
            product = self.state['products'][product_id]
            opportunity = self.window_opportunity(self.stations[workstation_id], time)
            self.require('workstation.full_window_fit', True, opportunity is not None and values_match(opportunity[0], time), event)
            operation = {'product_id': product_id, 'plant_id': plant['id'], 'workstation_id': workstation_id,
                'operation_id': self.stations[workstation_id]['operation_id'], 'operation_index': product['next_operation'],
                'start_time': time, 'end_time': time + self.stations[workstation_id]['duration'], 'window': opportunity[1]}
            self.match_fields(event, operation)
            product['inventory_wait'] += time - self.state['inventory'].pop(product_id)
            product.update(state='processing', location=workstation_id)
            self.state['workstations'][workstation_id] = operation
            if capacity is not None:
                intermediate = product['next_operation'] + 1 < len(self.products[product_id]['operations'])
                if intermediate:
                    self.state['held_slots'].append(product_id)
                self.required.append(('inventory_hold' if intermediate else 'inventory_slot_release', 'product_id', product_id))
        elif kind in ('inventory_hold', 'inventory_slot_release'):
            self.match_fields(event, {'product_id': product_id, 'plant_id': plant['id'],
                'waiting_count': len(self.state['inventory']), 'held_count': len(self.state['held_slots'])})
        elif kind == 'operation_complete':
            operation = self.state['workstations'][workstation_id]
            self.require('workstation.active_completion', True, operation is not None, event)
            self.match_fields(event, operation)
            self.require('workstation.fixed_completion_time', operation['end_time'], time, event)
            product = self.state['products'][product_id]
            product['history'].append(dict(operation))
            product['next_operation'] += 1
            self.state['workstations'][workstation_id] = None
            self.required.append(('remanufactured' if product['next_operation'] == len(self.products[product_id]['operations']) else 'inventory_return', 'product_id', product_id))
        elif kind == 'inventory_return':
            product = self.state['products'][product_id]
            replaced = product_id in self.state['held_slots']
            self.require('inventory.intermediate_hold', plant['inventory_capacity'] is not None, replaced, event)
            self.match_fields(event, {'product_id': product_id, 'plant_id': plant['id'], 'next_operation_index': product['next_operation'],
                'held_slot_replaced': replaced, 'waiting_count': len(self.state['inventory']) + 1,
                'held_count': len(self.state['held_slots']) - int(replaced)})
            if replaced:
                self.state['held_slots'].remove(product_id)
            self.state['inventory'][product_id] = time
            product.update(state='in_plant_inventory', location=plant['id'])
        elif kind == 'remanufactured':
            self.match_fields(event, {'product_id': product_id})
            product = self.state['products'][product_id]
            self.require('product.final_operation', len(self.products[product_id]['operations']), product['next_operation'], event)
            product.update(state='remanufactured', location=None)
            product['milestones']['completion'] = time
        elif kind == 'horizon':
            self.require('time.horizon', self.scenario['horizon'], time, event)
            self.require('events.settled_deadlines', True,
                         all(due > time + TOLERANCE for due in self.pending_deadlines()), event)
            self.check_physical_state(event)
            self.require('events.settled_horizon', [], self.required, event)
            self.match_fields(event, {'waiting_count': len(self.state['inventory']), 'held_count': len(self.state['held_slots'])})
            self.finished = True
        else:
            raise EvidenceProblem(f'Unsupported inventory event type {kind}')
        capacity = plant['inventory_capacity']
        self.require('inventory.capacity', True, capacity is None or self.occupancy() <= capacity, event)

    def replay(self, events: list[Record]) -> Record:
        """Read events in order, enforce deadlines and return the independently settled horizon state."""
        for sequence, event in enumerate(events, 1):
            if not all(key in event for key in ('sequence', 'time', 'event_type')):
                raise EvidenceProblem(f'Incomplete event record {sequence}')
            if type(event['time']) not in (int, float):
                raise EvidenceProblem(f'Invalid time in event {sequence}')
            self.require('events.sequence', sequence, event['sequence'], event)
            self.require('events.after_horizon', False, self.finished, event)
            self.require('events.chronological', True, self.state['time'] <= event['time'] <= self.scenario['horizon'], event)
            self.check_clock_boundary(event)
            self.advance_range_travel(event['time'], event)
            self.state['time'] = event['time']
            self.consume_event(event)
        if not self.finished:
            raise EvidenceProblem('Missing horizon record')
        horizon = self.scenario['horizon']
        for product_id, entered in self.state['inventory'].items():
            self.state['products'][product_id]['inventory_wait'] += horizon - entered
        for vehicle_id, vehicle in self.state['vehicles'].items():
            if self.vehicles[vehicle_id]['maximum_range'] is None and vehicle['active_leg'] is not None:
                vehicle['distance'] += (horizon - vehicle['active_leg']['departure_time']) * self.vehicles[vehicle_id]['speed']
            if vehicle['waiting_since'] is not None:
                vehicle['waiting_to_unload'] += horizon - vehicle['waiting_since']
        return self.state

    def reconcile(self, evidence: dict[str, Any]) -> Record:
        """Calculate totals independently and compare complete final-state, summary and dispatch exports."""
        generated = sum(p['state'] != 'unreleased' for p in self.state['products'].values())
        completed = sum(p['state'] == 'remanufactured' for p in self.state['products'].values())
        recycled = sum(p['state'] == 'recycled' for p in self.state['products'].values())
        summary = {'generated': generated, 'remanufactured': completed, 'recycled': recycled, 'unfinished': generated - completed - recycled,
            'vehicle_distance': sum(v['distance'] for v in self.state['vehicles'].values()),
            'plant_inventory_wait': sum(p['inventory_wait'] for p in self.state['products'].values()),
            'waiting_to_unload': sum(v['waiting_to_unload'] for v in self.state['vehicles'].values())}
        self.state['summary'] = summary
        if self.scenario['policy'] == 'static_quality_aware' and 'dispatch_decisions.jsonl' in evidence:
            audit_aware_dispatch_records(self.expected_decisions, evidence['dispatch_decisions.jsonl'], self.findings)
        for filename, expected in (('summary.json', summary), ('final_state.json', self.state),
                                   ('dispatch_decisions.jsonl', self.expected_decisions)):
            if filename in evidence:
                check_value(self.findings, f'results.{filename}', expected, evidence[filename], file=filename)
        return summary


def audit_configured_plants(scenario: Record) -> list[Record]:
    """Read explicit plant definitions or the legacy single-plant definition independently."""
    return scenario['plants'] if 'plants' in scenario else [scenario['plant']]


def audit_plant_containers(scenario: Record, state: Record) -> dict[str, Record]:
    """Return authoritative per-plant containers without counting primary compatibility aliases twice."""
    if state.get('plant_states'):
        return state['plant_states']
    return {scenario['plant']['id']: {key: state[key] for key in ('inventory', 'held_slots', 'workstations', 'waiting_vehicles')}}


def audit_dynamic_forecast_context(scenario: Record, state: Record) -> Record:
    """Independently rebuild the aware released-only physical observation."""
    released = {product_id for product_id, product in state['products'].items() if product['state'] != 'unreleased'}
    snapshot = copy.deepcopy(state)
    snapshot['products'] = {product_id: product for product_id, product in snapshot['products'].items() if product_id in released}
    physical = {key: copy.deepcopy(scenario[key]) for key in ('horizon', 'plant', 'collection_points', 'recycling_facilities', 'vehicles')}
    if 'plants' in scenario:
        physical['plants'] = copy.deepcopy(scenario['plants'])
    physical.update(format_version=3, policy='dynamic_quality_aware', trips=[])
    physical['products'] = [{key: copy.deepcopy(product[key]) for key in ('id', 'origin', 'release_time', 'disposition', 'operations')}
                            for product in scenario['products'] if product['id'] in released]
    return {'scenario': physical, 'state': snapshot}


def audit_dynamic_forecast(boundary: Record, candidate: Record) -> Record:
    """Independently sweep revised routes, unloading queues and plant work, propagating all delivery delays."""
    context = boundary['forecast_context']
    scenario, state = context['scenario'], copy.deepcopy(context['state'])
    plants = {plant['id']: plant for plant in audit_configured_plants(scenario)}
    containers = audit_plant_containers(scenario, state)
    sites = {site['id']: site for site in [*plants.values(), *scenario['collection_points'], *scenario['recycling_facilities']]}
    definitions = {product['id']: product for product in scenario['products']}
    vehicle_definitions = {vehicle['id']: vehicle for vehicle in scenario['vehicles']}
    for request in boundary['requests']:
        state['products'][request['product_id']].update(assigned_vehicle=request['assigned_vehicle'], planned_destination=request['planned_destination'])
    state['products'][candidate['product_id']].update(assigned_vehicle=candidate['vehicle_id'], planned_destination=candidate['destination'])
    arrivals: dict[str, float] = {}
    for observed in boundary['vehicles']:
        vehicle = state['vehicles'][observed['id']]
        vehicle['remaining_stops'] = copy.deepcopy(candidate['new_remaining_stops'] if observed['id'] == candidate['vehicle_id'] else observed['remaining_stops'])
        if vehicle['remaining_stops'] and vehicle['status'] != 'waiting_to_unload':
            target = sites[vehicle['remaining_stops'][0]['site_id']]['coordinates']
            arrivals[observed['id']] = state['time'] + math.hypot(target[0] - vehicle['position'][0], target[1] - vehicle['position'][1]) / observed['speed']
    calendars = {}
    for plant_id, plant in plants.items():
        calendars[plant_id] = []
        for station in sorted(plant['workstations'], key=lambda station: station['id']):
            intervals = []
            for opening, closing in station['windows']:
                if intervals and intervals[-1][1] == opening:
                    intervals[-1][1] = closing
                else:
                    intervals.append([opening, closing])
            calendars[plant_id].append((station, intervals))
    deliveries = []
    first_start = None
    time = state['time']
    while time <= scenario['horizon']:
        for plant_id in sorted(plants):
            plant_state = containers[plant_id]
            for station_id in sorted(plant_state['workstations']):
                operation = plant_state['workstations'][station_id]
                if operation is None or operation['end_time'] > time:
                    continue
                product_id = operation['product_id']
                product = state['products'][product_id]
                product['next_operation'] += 1
                plant_state['workstations'][station_id] = None
                if product['next_operation'] < len(definitions[product_id]['operations']):
                    if product_id in plant_state['held_slots']:
                        plant_state['held_slots'].remove(product_id)
                    plant_state['inventory'][product_id] = time
        for vehicle_id in sorted(identifier for identifier, arrival in arrivals.items() if arrival <= time):
            del arrivals[vehicle_id]
            vehicle = state['vehicles'][vehicle_id]
            stop = vehicle['remaining_stops'][0]
            vehicle.update(location=stop['site_id'], position=list(sites[stop['site_id']]['coordinates']), status='idle')
            for product_id in stop['pickups']:
                vehicle['load'].append(product_id)
                state['products'][product_id].update(state='onboard_av', location=vehicle_id)
            for product_id in stop['deliveries']:
                deliveries.append({'product_id': product_id, 'vehicle_id': vehicle_id, 'destination': stop['site_id'], 'arrival_time': time})
            if stop['deliveries']:
                if stop['site_id'] in plants:
                    containers[stop['site_id']]['waiting_vehicles'][vehicle_id] = time
                else:
                    for product_id in stop['deliveries']:
                        vehicle['load'].remove(product_id)
                        state['products'][product_id].update(actual_destination=stop['site_id'], state='recycled')
                        state['products'][product_id]['milestones']['recycling'] = time
        for plant_id in sorted(plants):
            plant_state, plant = containers[plant_id], plants[plant_id]
            while True:
                changed = False
                for vehicle_id in sorted(plant_state['waiting_vehicles'], key=lambda identifier: (plant_state['waiting_vehicles'][identifier], identifier)):
                    vehicle = state['vehicles'][vehicle_id]
                    eligible = [product_id for product_id in vehicle['load'] if state['products'][product_id]['planned_destination'] == plant_id]
                    while eligible and (plant['inventory_capacity'] is None or len(plant_state['inventory']) + len(plant_state['held_slots']) < plant['inventory_capacity']):
                        product_id = eligible.pop(0)
                        vehicle['load'].remove(product_id)
                        plant_state['inventory'][product_id] = time
                        state['products'][product_id].update(actual_destination=plant_id, state='in_plant_inventory')
                        state['products'][product_id]['milestones']['admission'] = time
                        changed = True
                    if not eligible:
                        del plant_state['waiting_vehicles'][vehicle_id]
                        vehicle['status'] = 'idle'
                    else:
                        vehicle['status'] = 'waiting_to_unload'
                for station, intervals in calendars[plant_id]:
                    if plant_state['workstations'][station['id']] is not None:
                        continue
                    window = next((interval for interval in intervals if interval[0] <= time < interval[1] and time + station['duration'] <= interval[1]), None)
                    eligible = sorted((product_id for product_id in plant_state['inventory'] if
                        definitions[product_id]['operations'][state['products'][product_id]['next_operation']] == station['operation_id']),
                        key=lambda product_id: (plant_state['inventory'][product_id], product_id))
                    if window is None or not eligible:
                        continue
                    product_id = eligible[0]
                    plant_state['inventory'].pop(product_id)
                    index = state['products'][product_id]['next_operation']
                    plant_state['workstations'][station['id']] = {'product_id': product_id, 'end_time': time + station['duration']}
                    if plant['inventory_capacity'] is not None and index + 1 < len(definitions[product_id]['operations']):
                        plant_state['held_slots'].append(product_id)
                    if product_id == candidate['product_id'] and first_start is None:
                        first_start = time
                    changed = True
                    break
                if not changed:
                    break
        for vehicle_id, vehicle in sorted(state['vehicles'].items()):
            if vehicle_id in arrivals or vehicle['status'] == 'waiting_to_unload' or not vehicle['remaining_stops']:
                continue
            stop = vehicle['remaining_stops'][0]
            if vehicle['location'] != stop['site_id'] or any(product_id in vehicle['load'] for product_id in stop['deliveries']):
                continue
            vehicle['remaining_stops'].pop(0)
            if vehicle['remaining_stops']:
                target = sites[vehicle['remaining_stops'][0]['site_id']]['coordinates']
                arrivals[vehicle_id] = time + math.hypot(target[0] - vehicle['position'][0], target[1] - vehicle['position'][1]) / vehicle_definitions[vehicle_id]['speed']
        future = list(arrivals.values())
        future.extend(operation['end_time'] for plant_state in containers.values() for operation in plant_state['workstations'].values() if operation is not None)
        future.extend(opening for entries in calendars.values() for _, intervals in entries for opening, _ in intervals if opening > time)
        if not future or min(future) > scenario['horizon']:
            break
        time = min(future)
    product = state['products'][candidate['product_id']]
    admission = product['milestones'].get('admission', product['milestones'].get('recycling'))
    return {'predicted_arrival': next((delivery['arrival_time'] for delivery in deliveries if delivery['product_id'] == candidate['product_id']), None),
        'predicted_admission': admission, 'predicted_first_start': first_start,
        'predicted_inventory_wait': first_start - admission if first_start is not None else None,
        'projected_deliveries': deliveries}


def audit_dynamic_quality_records(expected: Record, observed: Record, findings: list[Record], event: Record) -> None:
    """Audit every recorded aware forecast/score against independent candidate projections."""
    if 'requests' not in observed:
        raise EvidenceProblem('Missing dynamic quality candidate records')
    for wanted_request, actual_request in zip(expected['requests'], observed['requests']):
        for wanted, actual in zip(wanted_request['candidates'], actual_request['candidates']):
            for field in ('predicted_arrival', 'predicted_admission', 'predicted_first_start', 'predicted_inventory_wait', 'projected_deliveries', 'score'):
                if field not in wanted:
                    continue
                if field not in actual:
                    raise EvidenceProblem(f'Missing candidate forecast field {field}')
                check_value(findings, f'dynamic.forecast.{field}', wanted[field], actual[field], event,
                            product_id=wanted['product_id'], vehicle_id=wanted['vehicle_id'], destination=wanted['destination'])


def audit_dynamic_shape(scenario: Record) -> Record:
    """Copy resolved inputs into the dynamic multi-point shape without altering source evidence."""
    result = copy.deepcopy(scenario)
    if result['format_version'] == 2:
        result['collection_points'] = [result.pop('collection_point')]
    if 'plants' in result:
        result['plant'] = copy.deepcopy(result['plants'][0])
    policy = result.get('policy') if result.get('policy') in ('dynamic_quality_unaware', 'dynamic_quality_aware') else 'dynamic_quality_unaware'
    result.update(format_version=3, policy=policy, trips=[])
    return result


def validate_dynamic_inputs(scenario: Record) -> None:
    """Independently validate dynamic network identifiers/origins and existing physical constraints."""
    if scenario['format_version'] != 3 or scenario['policy'] not in ('dynamic_quality_unaware', 'dynamic_quality_aware') or scenario.get('trips', []):
        raise EvidenceProblem('Expected dynamic format 3 with no fixed trips')
    plants = audit_configured_plants(scenario)
    if not plants:
        raise EvidenceProblem('No remanufacturing plants')
    points = scenario['collection_points']
    identifiers = [site['id'] for site in [*plants, *points, *scenario['recycling_facilities']]]
    identifiers += [vehicle['id'] for vehicle in scenario['vehicles']] + [product['id'] for product in scenario['products']]
    if len(set(identifiers)) != len(identifiers) or not all(isinstance(identifier, str) and identifier for identifier in identifiers):
        raise EvidenceProblem('Dynamic entity IDs must be unique strings')
    for point in points:
        if len(point['coordinates']) != 2 or any(type(value) not in (int, float) or not math.isfinite(value) for value in point['coordinates']):
            raise EvidenceProblem('Invalid collection point coordinates')
    if any(product['origin'] not in {point['id'] for point in points} for product in scenario['products']):
        raise EvidenceProblem('Unknown dynamic product origin')
    proxy = copy.deepcopy(scenario)
    dummy = '__empty_collection__'
    while dummy in identifiers:
        dummy += '_'
    proxy['plant'] = plants[0]
    proxy.update(format_version=2, policy='static_quality_unaware', trips=[],
                 collection_point=points[0] if points else {'id': dummy, 'coordinates': [0, 0]})
    for product in proxy['products']:
        product['origin'] = proxy['collection_point']['id']
    facilities = {plant['id'] for plant in plants} | {site['id'] for site in scenario['recycling_facilities']}
    if any(vehicle['initial_site'] not in facilities for vehicle in scenario['vehicles']):
        raise EvidenceProblem('Unknown initial facility')
    for plant in plants:
        proxy['plant'] = plant
        for vehicle in proxy['vehicles']:
            vehicle['initial_site'] = plant['id']
        validate_inventory_inputs(proxy)


def audit_merge_stops(stops: list[Record]) -> list[Record]:
    """Independently canonicalize consecutive same-site service into one route stop."""
    result = []
    for item in stops:
        if result and result[-1]['site_id'] == item['site_id']:
            result[-1]['pickups'] = sorted(set(result[-1]['pickups'] + item['pickups']))
            result[-1]['deliveries'] = sorted(set(result[-1]['deliveries'] + item['deliveries']))
        else:
            result.append(copy.deepcopy(item))
    return result


def audit_route_distance(position: list[float], stops: list[Record], sites: dict[str, Record]) -> float:
    """Independently sum route geometry from the replayed current position."""
    result = 0.0
    for item in stops:
        target = sites[item['site_id']]['coordinates']
        result += math.hypot(target[0] - position[0], target[1] - position[1])
        position = target
    return result


def audit_dynamic_feasibility(vehicle: Record, stops: list[Record], requests: dict[str, Record],
                              sites: dict[str, Record], candidate_id: str, destination: str) -> tuple[bool, str]:
    """Independently replay projected payload, ownership, committed destinations and range through all stops."""
    if vehicle['status'] == 'waiting_to_unload':
        return False, 'vehicle_waiting_to_unload'
    contents = set(vehicle['load'])
    pickups = set(contents)
    delivered = set()
    owed = {product_id for product_id, request in requests.items() if request['assigned_vehicle'] == vehicle['id']} | {candidate_id}
    position, remaining = vehicle['position'], vehicle['remaining_range']
    for item in stops:
        site = sites[item['site_id']]
        distance = math.hypot(site['coordinates'][0] - position[0], site['coordinates'][1] - position[1])
        if remaining is not None:
            if remaining + TOLERANCE < distance:
                return False, 'range_exceeded_before_facility'
            remaining -= distance
        position = site['coordinates']
        if site['kind'] != 'collection' and remaining is not None:
            remaining = vehicle['maximum_range']
        for product_id in item['pickups']:
            request = requests[product_id]
            if site['kind'] != 'collection' or request['origin'] != site['id'] or product_id in pickups:
                return False, 'invalid_or_duplicate_pickup'
            if request['assigned_vehicle'] not in (None, vehicle['id']):
                return False, 'pickup_owned_by_another_vehicle'
            contents.add(product_id)
            pickups.add(product_id)
            if len(contents) > vehicle['payload_capacity']:
                return False, 'payload_exceeded'
        for product_id in item['deliveries']:
            request = requests[product_id]
            target = destination if product_id == candidate_id else request['planned_destination']
            expected_kind = 'plant' if request['disposition'] == 'remanufacture' else 'recycler'
            if product_id not in contents or product_id in delivered or target != site['id'] or site['kind'] != expected_kind:
                return False, 'precedence_or_destination_violation'
            contents.remove(product_id)
            delivered.add(product_id)
    if contents or delivered != owed:
        return False, 'unserved_commitment'
    if not stops or sites[stops[-1]['site_id']]['kind'] == 'collection':
        return False, 'missing_facility_escape'
    return True, 'feasible'


def audit_dynamic_candidates(boundary: Record, request: Record) -> list[Record]:
    """Independently enumerate the declared insertion domain, scores and all rejected alternatives."""
    sites = {site['id']: site for site in boundary['sites']}
    requests = {item['product_id']: item for item in boundary['requests']}
    destination_kind = 'plant' if request['disposition'] == 'remanufacture' else 'recycler'
    destinations = sorted(identifier for identifier, site in sites.items() if site['kind'] == destination_kind)
    facilities = sorted(identifier for identifier, site in sites.items() if site['kind'] != 'collection')
    result, seen = [], set()
    for vehicle in sorted(boundary['vehicles'], key=lambda item: item['id']):
        if vehicle['status'] == 'waiting_to_unload':
            continue
        old = vehicle['remaining_stops']
        original_distance = audit_route_distance(vehicle['position'], old, sites)
        for destination in destinations:
            for pickup_index in range(len(old) + 1):
                first = copy.deepcopy(old)
                first.insert(pickup_index, {'site_id': request['origin'], 'pickups': [request['product_id']], 'deliveries': []})
                for delivery_index in range(pickup_index + 1, len(first) + 1):
                    base = copy.deepcopy(first)
                    base.insert(delivery_index, {'site_id': destination, 'pickups': [], 'deliveries': [request['product_id']]})
                    variants = [(base, 0)]
                    if vehicle['maximum_range'] is not None:
                        level = [base]
                        for count in (1, 2):
                            next_level = []
                            for route in level:
                                for position in range(len(route) + 1):
                                    for facility in facilities:
                                        candidate = copy.deepcopy(route)
                                        candidate.insert(position, {'site_id': facility, 'pickups': [], 'deliveries': []})
                                        next_level.append(candidate)
                                        variants.append((candidate, count))
                            level = next_level
                    for variant, count in variants:
                        route = audit_merge_stops(variant)
                        canonical = json.dumps(route, sort_keys=True, separators=(',', ':'))
                        key = vehicle['id'], destination, canonical
                        if key in seen:
                            continue
                        seen.add(key)
                        feasible, reason = audit_dynamic_feasibility(vehicle, route, requests, sites, request['product_id'], destination)
                        distance = audit_route_distance(vehicle['position'], route, sites)
                        added = max(0.0, distance - original_distance)
                        result.append({'candidate_id': f'C{len(result) + 1:05d}', 'product_id': request['product_id'],
                            'vehicle_id': vehicle['id'], 'destination': destination, 'pickup_index': pickup_index,
                            'delivery_index': delivery_index, 'extra_visits': count, 'old_remaining_stops': copy.deepcopy(old),
                            'new_remaining_stops': route, 'position': list(vehicle['position']), 'remaining_range': vehicle['remaining_range'],
                            'feasible': feasible, 'reason': reason, 'remaining_distance': distance, 'added_distance': added,
                            'score': [round(added, 12), vehicle['id'], destination, pickup_index, delivery_index, count, canonical]})
    return result


def audit_dynamic_selection(boundary: Record) -> tuple[list[Record], list[Record]]:
    """Independently rank complete candidates in release/ID order while retaining accepted commitments."""
    working = copy.deepcopy(boundary)
    proposals, audits = [], []
    outstanding = sorted((request for request in working['requests'] if request['state'] == 'at_collection_point'
        and request['assigned_vehicle'] is None), key=lambda item: (item['release_time'], item['product_id']))
    for request in outstanding:
        candidates = audit_dynamic_candidates(working, request)
        if working.get('policy') == 'dynamic_quality_aware':
            for candidate in candidates:
                if candidate['feasible']:
                    candidate.update(audit_dynamic_forecast(working, candidate))
                    start_flag = candidate['predicted_first_start'] is not None
                    candidate['score'] = [0 if start_flag else 1, candidate['predicted_inventory_wait'] if start_flag else 0.0,
                                          round(candidate['added_distance'], 12), *candidate['score'][1:]]
        feasible = [candidate for candidate in candidates if candidate['feasible']]
        selected = min(feasible, key=lambda item: item['score']) if feasible else None
        audits.append({'product_id': request['product_id'], 'candidates': candidates,
            'selected_candidate_id': selected['candidate_id'] if selected else None,
            'reason': 'minimum_added_distance_then_stable_ids_and_positions' if selected else 'no_feasible_insertion'})
        if working.get('policy') == 'dynamic_quality_aware' and selected is not None:
            audits[-1]['reason'] = 'start_within_horizon_then_admission_wait_then_added_distance_then_stable_ties'
        if selected is not None:
            proposals.append(copy.deepcopy(selected))
            for vehicle in working['vehicles']:
                if vehicle['id'] == selected['vehicle_id']:
                    vehicle['remaining_stops'] = copy.deepcopy(selected['new_remaining_stops'])
            request.update(assigned_vehicle=selected['vehicle_id'], planned_destination=selected['destination'])
    return proposals, audits


class DynamicAudit(InventoryAudit):
    """Independent dynamic transport/decision replay sharing only the verifier's plant checks."""

    def __init__(self, scenario: Record, findings: list[Record]) -> None:
        """Initialize independent dynamic vehicle positions, routes, notices and expected proposals."""
        proxy = copy.deepcopy(scenario)
        proxy['collection_point'] = scenario['collection_points'][0] if scenario['collection_points'] else {'id': '__empty__', 'coordinates': [0, 0]}
        super().__init__(proxy, findings)
        self.plant_definitions = {plant['id']: plant for plant in audit_configured_plants(scenario)}
        self.delivery_sites = {site['id']: site for site in [*self.plant_definitions.values(), *scenario['recycling_facilities']]}
        self.route_sites = {site['id']: site for site in [*self.plant_definitions.values(), *scenario['collection_points'], *scenario['recycling_facilities']]}
        if len(self.plant_definitions) > 1:
            self.state['plant_states'] = {plant_id: {'inventory': {}, 'held_slots': [], 'waiting_vehicles': {},
                'workstations': dict.fromkeys(station['id'] for station in plant['workstations'])} for plant_id, plant in self.plant_definitions.items()}
            self.state.update(self.state['plant_states'][scenario['plant']['id']])
        self._plant_context_active = False
        for vehicle_id, definition in self.vehicles.items():
            self.state['vehicles'][vehicle_id].update(position=list(self.route_sites[definition['initial_site']]['coordinates']),
                remaining_range=definition['maximum_range'], leg_distance_accounted=0.0, remaining_stops=[], route_version=0, leg_serial=0)
        self.notices = {'released_products': set(), 'available_vehicles': set()}
        self.initial_available: set[str] = set()
        self.proposals: list[Record] = []

    def consume_plant_event(self, event: Record, plant_id: str) -> None:
        """Temporarily bind one plant's independent containers for existing local transition checks."""
        self.require('plant.known_facility', True, plant_id in self.plant_definitions, event)
        fields = ('inventory', 'held_slots', 'workstations', 'waiting_vehicles')
        previous = {key: self.state[key] for key in fields}
        previous_plant, previous_stations = self.scenario['plant'], self.stations
        self.state.update(self.state['plant_states'][plant_id])
        self.scenario['plant'] = self.plant_definitions[plant_id]
        self.stations = {station['id']: station for station in self.scenario['plant']['workstations']}
        self._plant_context_active = True
        try:
            self.consume_event(event)
        finally:
            self._plant_context_active = False
            self.scenario['plant'], self.stations = previous_plant, previous_stations
            self.state.update(previous)

    def check_physical_state(self, event: Record) -> None:
        """Check network-wide physical conservation and each plant's independent capacity and holds."""
        if not self.state.get('plant_states'):
            super().check_physical_state(event)
            return
        for product_id, product in self.state['products'].items():
            count = int(product['state'] in ('at_collection_point', 'remanufactured', 'recycled'))
            count += sum(product_id in vehicle['load'] for vehicle in self.state['vehicles'].values())
            count += sum(product_id in plant['inventory'] for plant in self.state['plant_states'].values())
            count += sum(operation is not None and operation['product_id'] == product_id for plant in self.state['plant_states'].values() for operation in plant['workstations'].values())
            self.require('product.conservation', int(product['state'] != 'unreleased'), count, event)
        for plant_id, plant in self.state['plant_states'].items():
            capacity = self.plant_definitions[plant_id]['inventory_capacity']
            self.require('inventory.capacity', True, capacity is None or len(plant['inventory']) + len(plant['held_slots']) <= capacity, event)
            holds = sorted(operation['product_id'] for operation in plant['workstations'].values() if operation is not None and
                           operation['operation_index'] + 1 < len(self.products[operation['product_id']]['operations'])) if capacity is not None else []
            self.require('inventory.hold_ownership', holds, sorted(plant['held_slots']), event)

    def policy_input(self) -> Record:
        """Rebuild the selected policy observation from independent transport and plant replay."""
        sites = [{'id': site['id'], 'coordinates': copy.deepcopy(site['coordinates']), 'kind': kind}
            for kind, definitions in (('plant', list(self.plant_definitions.values())), ('recycler', self.scenario['recycling_facilities']),
                                     ('collection', self.scenario['collection_points'])) for site in definitions]
        requests = [{'product_id': product_id, 'origin': self.products[product_id]['origin'],
            'release_time': self.products[product_id]['release_time'], 'disposition': self.products[product_id]['disposition'],
            'state': product['state'], 'assigned_vehicle': product['assigned_vehicle'], 'planned_destination': product['planned_destination']}
            for product_id, product in self.state['products'].items() if product['state'] in ('at_collection_point', 'onboard_av')]
        vehicles = [{**copy.deepcopy(self.state['vehicles'][definition['id']]), 'id': definition['id'], 'speed': definition['speed'],
            'payload_capacity': definition['payload_capacity'], 'maximum_range': definition['maximum_range']} for definition in self.scenario['vehicles']]
        boundary = {'time': self.state['time'], 'sites': sites, 'requests': requests, 'vehicles': vehicles,
                    'insertion_rules': {'max_extra_facility_visits': 2, 'distance_score_decimals': 12}}
        if self.scenario['policy'] == 'dynamic_quality_aware':
            boundary['policy'] = self.scenario['policy']
            boundary['forecast_context'] = audit_dynamic_forecast_context(self.scenario, self.state)
            for request in requests:
                request['operations'] = list(self.products[request['product_id']]['operations'])
        return boundary

    def pending_deadlines(self) -> list[float]:
        """Require pending dispatch/service/departure as well as ordinary physical deadlines before advancing time."""
        deadlines = super().pending_deadlines()
        for plant_id, plant_state in audit_plant_containers(self.scenario, self.state).items():
            if plant_id == self.scenario['plant']['id']:
                continue
            plant = self.plant_definitions[plant_id]
            deadlines.extend(operation['end_time'] for operation in plant_state['workstations'].values() if operation is not None)
            for station in plant['workstations']:
                eligible = any(self.products[product_id]['operations'][self.state['products'][product_id]['next_operation']] == station['operation_id'] for product_id in plant_state['inventory'])
                if plant_state['workstations'][station['id']] is None and eligible:
                    opportunity = self.window_opportunity(station, self.state['time'])
                    if opportunity:
                        deadlines.append(opportunity[0])
            if plant_state['waiting_vehicles'] and (plant['inventory_capacity'] is None or len(plant_state['inventory']) + len(plant_state['held_slots']) < plant['inventory_capacity']):
                deadlines.append(self.state['time'])
        if any(self.notices.values()) or any(vehicle['active_leg'] is None and vehicle['status'] != 'waiting_to_unload'
            and vehicle['remaining_stops'] for vehicle in self.state['vehicles'].values()):
            deadlines.append(self.state['time'])
        return deadlines

    def advance_range_travel(self, time: float, event: Record) -> None:
        """Reconstruct finite or unlimited actual motion including cancelled-leg prefixes."""
        for vehicle_id, definition in self.vehicles.items():
            vehicle = self.state['vehicles'][vehicle_id]
            leg = vehicle['active_leg']
            if leg is None:
                continue
            distance = min(leg['distance'], max(0, time - leg['departure_time']) * definition['speed'])
            increment = distance - vehicle['leg_distance_accounted']
            self.require('dynamic.actual_motion', True, increment >= -TOLERANCE, event)
            if vehicle['remaining_range'] is not None:
                self.require('dynamic.range_consumption', True, increment <= vehicle['remaining_range'] + TOLERANCE, event)
                vehicle['remaining_range'] = max(0.0, vehicle['remaining_range'] - increment)
            vehicle['distance'] += increment
            vehicle['leg_distance_accounted'] = distance
            ratio = distance / leg['distance'] if leg['distance'] else 1.0
            vehicle['position'] = [leg['origin_coordinates'][axis] + ratio * (leg['target_coordinates'][axis] - leg['origin_coordinates'][axis]) for axis in (0, 1)]

    def prepare_dispatch(self, event: Record) -> None:
        """Audit batched triggers and reconstruct every candidate, chosen plan and required atomic event."""
        trigger = {key: sorted(values) for key, values in self.notices.items()}
        self.require('dynamic.valid_trigger', True, any(self.notices.values()), event)
        self.require('dynamic.settled_completions', [], [(plant_id, station_id) for plant_id, plant in audit_plant_containers(self.scenario, self.state).items()
            for station_id, operation in plant['workstations'].items() if operation is not None and operation['end_time'] <= event['time']], event)
        self.match_fields(event, {'trigger': trigger, 'decision_index': len(self.expected_decisions) + 1})
        self.require('dynamic.batched_releases', [], sorted(product_id for product_id, definition in self.products.items()
            if self.state['products'][product_id]['state'] == 'unreleased' and definition['release_time'] <= event['time']), event)
        self.require('dynamic.settled_arrivals', [], sorted(vehicle_id for vehicle_id, vehicle in self.state['vehicles'].items()
            if vehicle['active_leg'] is not None and vehicle['active_leg']['arrival_time'] <= event['time']), event)
        self.require('dynamic.settled_service', [], sorted(vehicle_id for vehicle_id, vehicle in self.state['vehicles'].items()
            if vehicle['active_leg'] is None and vehicle['status'] != 'waiting_to_unload' and vehicle['remaining_stops']), event)
        boundary = self.policy_input()
        proposals, audits = audit_dynamic_selection(boundary)
        self.expected_decisions.append({'time': event['time'], 'trigger': trigger, 'last_event_sequence': event['sequence'],
            'policy_input': boundary, 'requests': audits, 'proposals': proposals})
        self.proposals = copy.deepcopy(proposals)
        shadow = copy.deepcopy(self.state['vehicles'])
        for proposal in proposals:
            vehicle_id, product_id = proposal['vehicle_id'], proposal['product_id']
            vehicle = shadow[vehicle_id]
            self.required.append(('assignment', 'product_id', product_id))
            if vehicle['active_leg'] is not None and vehicle['active_leg']['target'] != proposal['new_remaining_stops'][0]['site_id']:
                self.required.append(('leg_cancelled', 'vehicle_id', vehicle_id))
                vehicle['active_leg'] = None
            self.required.append(('route_revised', 'vehicle_id', vehicle_id))
            vehicle['remaining_stops'] = proposal['new_remaining_stops']
        for vehicle_id, vehicle in sorted(shadow.items()):
            if vehicle['active_leg'] is None and vehicle['status'] != 'waiting_to_unload' and vehicle['remaining_stops']:
                self.required.append(('leg_departure', 'vehicle_id', vehicle_id))
        for values in self.notices.values():
            values.clear()

    def consume_event(self, event: Record) -> None:
        """Check dynamic transport/revision/service events; delegate unchanged workstation rules to independent plant replay."""
        kind, time = event['event_type'], event['time']
        if kind == 'horizon' and self.state.get('plant_states'):
            self.require('time.horizon', self.scenario['horizon'], time, event)
            self.require('events.settled_horizon', [], self.required, event)
            self.require('events.settled_deadlines', True, all(due > time + TOLERANCE for due in self.pending_deadlines()), event)
            self.check_physical_state(event)
            self.match_fields(event, {'waiting_count': sum(len(plant['inventory']) for plant in self.state['plant_states'].values()),
                'held_count': sum(len(plant['held_slots']) for plant in self.state['plant_states'].values())})
            self.finished = True
            return
        plant_kinds = {'operation_start', 'operation_complete', 'inventory_return', 'inventory_hold', 'inventory_slot_release', 'plant_admission', 'unload_wait', 'unload_retry'}
        if self.state.get('plant_states') and not self._plant_context_active and kind in plant_kinds:
            plant_id = event.get('plant_id') or self.state['vehicles'][event['vehicle_id']]['location']
            self.consume_plant_event(event, plant_id)
            return
        vehicle_id, product_id = event.get('vehicle_id'), event.get('product_id')
        custom = {'vehicle_available', 'dispatch', 'assignment', 'leg_cancelled', 'route_revised', 'leg_departure',
                  'leg_arrival', 'pickup', 'stop_completed', 'plant_admission', 'unload_retry', 'unload_wait'}
        if kind not in custom:
            super().consume_event(event)
            if kind == 'return_release':
                self.notices['released_products'].add(event['product']['id'])
            return
        had_required = bool(self.required)
        if self.required:
            expected_kind, field, identity = self.required.pop(0)
            self.require('dynamic.atomic_event', expected_kind, kind, event)
            self.require('dynamic.atomic_identity', identity, event.get(field), event)
        if vehicle_id is not None:
            self.require('vehicle.known_identity', True, vehicle_id in self.vehicles, event)
        vehicle = self.state['vehicles'].get(vehicle_id)
        if kind == 'dispatch':
            self.prepare_dispatch(event)
        elif kind == 'vehicle_available':
            reason = event['reason']
            if reason == 'initial':
                self.require('dynamic.initial_availability', True, time == 0 and vehicle_id not in self.initial_available, event)
                self.initial_available.add(vehicle_id)
            else:
                self.require('dynamic.availability_reason', 'facility_stop', reason, event)
                self.require('dynamic.availability_after_service', True, had_required, event)
                self.require('dynamic.facility_availability', True, vehicle['location'] in self.delivery_sites, event)
            self.require('dynamic.available_state', True, vehicle['status'] != 'waiting_to_unload', event)
            self.match_fields(event, {'vehicle_id': vehicle_id, 'reason': reason})
            self.notices['available_vehicles'].add(vehicle_id)
        elif kind == 'assignment':
            proposal = self.proposals[0]
            self.match_fields(event, {'product_id': proposal['product_id'], 'vehicle_id': proposal['vehicle_id'], 'destination': proposal['destination']})
            product = self.state['products'][product_id]
            self.require('dynamic.unassigned_request', ('at_collection_point', None), (product['state'], product['assigned_vehicle']), event)
            product.update(assigned_vehicle=vehicle_id, planned_destination=proposal['destination'])
        elif kind == 'leg_cancelled':
            leg = vehicle['active_leg']
            self.require('dynamic.active_cancellation', True, leg is not None, event)
            self.match_fields(event, {'vehicle_id': vehicle_id, 'leg_id': leg['leg_id'], 'old_arrival_time': leg['arrival_time'],
                'distance_travelled': vehicle['leg_distance_accounted'], 'position': vehicle['position'], 'remaining_range': vehicle['remaining_range']})
            vehicle.update(active_leg=None, status='idle', leg_distance_accounted=0.0)
        elif kind == 'route_revised':
            proposal = self.proposals.pop(0)
            self.require('dynamic.replan_position', vehicle['position'], event.get('position'), event)
            self.require('dynamic.replan_range', vehicle['remaining_range'], event.get('remaining_range'), event)
            self.match_fields(event, {'vehicle_id': proposal['vehicle_id'], 'product_id': proposal['product_id'],
                'route_version': vehicle['route_version'] + 1, 'position': vehicle['position'], 'remaining_range': vehicle['remaining_range'],
                'old_remaining_stops': vehicle['remaining_stops'], 'new_remaining_stops': proposal['new_remaining_stops']})
            vehicle['route_version'] += 1
            vehicle['remaining_stops'] = copy.deepcopy(proposal['new_remaining_stops'])
        elif kind == 'leg_departure':
            self.require('dynamic.departure_enabled', True, vehicle['active_leg'] is None and vehicle['remaining_stops'] != [] and vehicle['status'] != 'waiting_to_unload', event)
            target = self.route_sites[vehicle['remaining_stops'][0]['site_id']]
            distance = math.hypot(target['coordinates'][0] - vehicle['position'][0], target['coordinates'][1] - vehicle['position'][1])
            vehicle['leg_serial'] += 1
            leg = {'leg_id': f"{vehicle_id}:L{vehicle['leg_serial']}", 'vehicle_id': vehicle_id,
                'origin_coordinates': list(vehicle['position']), 'target': target['id'], 'target_coordinates': target['coordinates'],
                'departure_time': time, 'arrival_time': time + distance / self.vehicles[vehicle_id]['speed'],
                'distance': distance, 'load': list(vehicle['load']), 'remaining_range_at_departure': vehicle['remaining_range']}
            self.match_fields(event, leg)
            vehicle.update(active_leg=leg, status='travelling', leg_distance_accounted=0.0)
        elif kind == 'leg_arrival':
            leg = vehicle['active_leg']
            self.require('dynamic.live_arrival', True, leg is not None, event)
            self.match_fields(event, {**leg, 'remaining_range_on_arrival': vehicle['remaining_range']})
            self.require('dynamic.arrival_time', leg['arrival_time'], time, event)
            vehicle.update(location=leg['target'], position=list(leg['target_coordinates']), active_leg=None,
                           status='idle', leg_distance_accounted=0.0)
            stop = vehicle['remaining_stops'][0]
            if self.vehicles[vehicle_id]['maximum_range'] is not None and leg['target'] in self.delivery_sites:
                self.required.append(('range_reset', 'vehicle_id', vehicle_id))
            self.required.extend(('pickup', 'product_id', product_id) for product_id in stop['pickups'])
            if stop['deliveries']:
                if leg['target'] in self.plant_definitions:
                    audit_plant_containers(self.scenario, self.state)[leg['target']]['waiting_vehicles'][vehicle_id] = time
                else:
                    # Facility service accepts eligible onboard products in their recorded pickup order.
                    self.required.extend(('recycler_acceptance', 'product_id', product_id) for product_id in vehicle['load']
                                         if self.state['products'][product_id]['planned_destination'] == leg['target'])
        elif kind == 'pickup':
            product = self.state['products'][product_id]
            self.require('dynamic.pickup_ownership', ('at_collection_point', vehicle_id), (product['state'], product['assigned_vehicle']), event)
            self.require('dynamic.pickup_location', self.products[product_id]['origin'], vehicle['location'], event)
            self.require('vehicle.payload', True, len(vehicle['load']) < self.vehicles[vehicle_id]['payload_capacity'], event)
            self.match_fields(event, {'product_id': product_id, 'vehicle_id': vehicle_id, 'collection_point': vehicle['location']})
            vehicle['load'].append(product_id)
            product.update(state='onboard_av', location=vehicle_id)
            product['milestones']['pickup'] = time
        elif kind == 'stop_completed':
            stop = vehicle['remaining_stops'][0]
            self.match_fields(event, {'vehicle_id': vehicle_id, 'stop': stop})
            self.require('dynamic.completed_site', stop['site_id'], vehicle['location'], event)
            self.require('dynamic.completed_deliveries', True, all(product_id not in vehicle['load'] and self.state['products'][product_id]['actual_destination'] == stop['site_id'] for product_id in stop['deliveries']), event)
            vehicle['remaining_stops'].pop(0)
            if stop['site_id'] in self.delivery_sites:
                self.required.append(('vehicle_available', 'vehicle_id', vehicle_id))
        elif kind in ('unload_wait', 'unload_retry', 'plant_admission'):
            capacity = self.scenario['plant']['inventory_capacity']
            if kind == 'plant_admission':
                self.require('inventory.capacity', True, capacity is None or self.occupancy() < capacity, event)
            order = self.waiting_order()
            self.require('inventory.registered_arrival', True, vehicle_id in order, event)
            if kind != 'unload_wait':
                self.require('inventory.waiting_vehicle_priority', order[0], vehicle_id, event)
            if kind == 'unload_wait':
                self.require('inventory.wait_only_when_full', True, capacity is not None and self.occupancy() == capacity, event)
                self.match_fields(event, {'vehicle_id': vehicle_id, 'arrival_time': self.state['waiting_vehicles'][vehicle_id],
                    'load': vehicle['load'], 'waiting_count': len(self.state['inventory']), 'held_count': len(self.state['held_slots'])})
                vehicle.update(status='waiting_to_unload', waiting_since=time, wait_logged=True)
            elif kind == 'unload_retry':
                self.require('inventory.retry_after_wait', True, vehicle['wait_logged'], event)
                self.match_fields(event, {'vehicle_id': vehicle_id, 'arrival_time': self.state['waiting_vehicles'][vehicle_id]})
                eligible = [product_id for product_id in vehicle['load'] if self.state['products'][product_id]['planned_destination'] == self.scenario['plant']['id']]
                self.required.append(('plant_admission', 'product_id', eligible[0]))
            else:
                eligible = [product_id for product_id in vehicle['load'] if self.state['products'][product_id]['planned_destination'] == self.scenario['plant']['id']]
                self.require('inventory.delivery_order', eligible[0], product_id, event)
                self.require('plant.disposition', 'remanufacture', self.products[product_id]['disposition'], event)
                self.match_fields(event, {'product_id': product_id, 'vehicle_id': vehicle_id, 'plant_id': self.scenario['plant']['id'],
                    'waiting_count': len(self.state['inventory']) + 1, 'held_count': len(self.state['held_slots'])})
                vehicle['load'].remove(product_id)
                self.state['inventory'][product_id] = time
                product = self.state['products'][product_id]
                product.update(state='in_plant_inventory', location=self.scenario['plant']['id'], actual_destination=self.scenario['plant']['id'])
                product['milestones']['admission'] = time
                if len(eligible) == 1:
                    del self.state['waiting_vehicles'][vehicle_id]
                    if vehicle['waiting_since'] is not None:
                        vehicle['waiting_to_unload'] += time - vehicle['waiting_since']
                    vehicle.update(status='idle', waiting_since=None, wait_logged=False)

    def replay(self, events: list[Record]) -> Record:
        """Replay dynamic events without counting partial travel a second time at the horizon."""
        for sequence, event in enumerate(events, 1):
            if not all(key in event for key in ('sequence', 'time', 'event_type')):
                raise EvidenceProblem(f'Incomplete dynamic event {sequence}')
            self.require('events.sequence', sequence, event['sequence'], event)
            self.require('events.after_horizon', False, self.finished, event)
            self.require('events.chronological', True, self.state['time'] <= event['time'] <= self.scenario['horizon'], event)
            self.check_clock_boundary(event)
            self.advance_range_travel(event['time'], event)
            self.state['time'] = event['time']
            self.consume_event(event)
        if not self.finished:
            raise EvidenceProblem('Missing dynamic horizon')
        for plant in audit_plant_containers(self.scenario, self.state).values():
            for product_id, entered in plant['inventory'].items():
                self.state['products'][product_id]['inventory_wait'] += self.scenario['horizon'] - entered
        for vehicle in self.state['vehicles'].values():
            if vehicle['waiting_since'] is not None:
                vehicle['waiting_to_unload'] += self.scenario['horizon'] - vehicle['waiting_since']
        return self.state

    def reconcile(self, evidence: dict[str, Any]) -> Record:
        """Audit transport visibility/candidate decisions precisely, then reconcile independent physical totals."""
        actual = evidence.get('dispatch_decisions.jsonl')
        if actual is not None:
            check_value(self.findings, 'dynamic.decision_count', len(self.expected_decisions), len(actual))
            for expected, observed in zip(self.expected_decisions, actual):
                event = {'time': expected['time'], 'sequence': expected['last_event_sequence']}
                if 'policy_input' not in observed:
                    raise EvidenceProblem('Missing dynamic policy input')
                for wanted, seen in zip(expected['policy_input']['vehicles'], observed['policy_input']['vehicles']):
                    check_value(self.findings, 'dynamic.replan_position', wanted['position'], seen.get('position'), event, vehicle_id=wanted['id'])
                    check_value(self.findings, 'dynamic.replan_range', wanted['remaining_range'], seen.get('remaining_range'), event, vehicle_id=wanted['id'])
                if self.scenario['policy'] == 'dynamic_quality_aware':
                    audit_dynamic_quality_records(expected, observed, self.findings, event)
                for field in ('trigger', 'policy_input', 'requests', 'proposals'):
                    if field not in observed:
                        raise EvidenceProblem(f'Dynamic decision lacks {field}')
                    check_value(self.findings, f'dynamic.{field}', expected[field], observed[field], event)
        return super().reconcile(evidence)


def verify_run(run_directory: Path) -> Record:
    """Read, independently replay and report one run; return the report with its exit code."""
    findings: list[Record] = []
    evidence = read_evidence(run_directory, findings)
    summary = None
    replay_complete = False
    scenario = evidence.get("scenario.json")
    manifest = evidence.get("run_manifest.json")
    if manifest is not None and not isinstance(manifest.get("run_id"), str):
        add_finding(findings, "incomplete", "manifest.run_id", "Run ID string", manifest.get("run_id"))
    if scenario is not None and manifest is not None:
        if manifest.get("input_format_version") not in (1, 2, 3) or manifest.get("output_format_version") != (3 if manifest.get("policy") in ("dynamic_quality_unaware", "dynamic_quality_aware") else manifest.get("input_format_version")) or manifest.get("input_format_version") != scenario.get("format_version"):
            add_finding(findings, "incomplete", "format.version", "Supported input/output formats matching the execution policy", manifest)
        else:
            for key, expected in {"scenario_sha256": hashlib.sha256((run_directory / "scenario.json").read_bytes()).hexdigest(),
                                  "seed_provenance": scenario.get("seed_provenance")}.items():
                if key not in manifest:
                    add_finding(findings, "incomplete", "manifest.evidence", key, "Missing field")
                else:
                    check_value(findings, f"manifest.{key}", expected, manifest[key])
    execution_policy = scenario.get('policy') if scenario else None
    if scenario is not None and manifest is not None:
        if 'policy' not in manifest:
            add_finding(findings, 'incomplete', 'manifest.policy', 'Selected run policy', 'Missing field')
        elif scenario.get('format_version') in (2, 3):
            check_value(findings, 'manifest.policy', True, manifest['policy'] in ('static_quality_unaware', 'static_quality_aware', 'dynamic_quality_unaware', 'dynamic_quality_aware'))
            execution_policy = manifest['policy']
        else:
            check_value(findings, 'manifest.policy', scenario.get('policy'), manifest['policy'])
    if manifest is not None and manifest.get("status") != "completed":
        add_finding(findings, "incomplete", "manifest.status", "completed", manifest.get("status"))
    if scenario is not None and (scenario.get('format_version') == 2 or
            scenario.get('seed_provenance', {}).get('source') == 'seeded_network_configuration'):
        try:
            audit_seeded_generation(scenario, findings)
        except (EvidenceProblem, KeyError, TypeError, ValueError, IndexError) as error:
            add_finding(findings, 'incomplete', 'generation.evidence', 'Complete supported generation evidence', str(error))
    supported = False
    if scenario is not None:
        try:
            validate_supported_scenario(scenario)
            supported = manifest is None or (manifest.get("input_format_version") == scenario["format_version"] and manifest.get("output_format_version") == (3 if execution_policy in ("dynamic_quality_unaware", "dynamic_quality_aware") else scenario["format_version"]))
        except (EvidenceProblem, KeyError, TypeError, ValueError, IndexError) as error:
            add_finding(findings, "incomplete", "scenario.scope", "Supported checkpoint-one scenario", str(error))
    if scenario is not None and scenario.get('format_version') in (2, 3) and (manifest is None or 'policy' not in manifest):
        supported = False  # A run override cannot be inferred from frozen inputs alone.
    if supported and "events.jsonl" in evidence:
        try:
            if execution_policy in ("dynamic_quality_unaware", "dynamic_quality_aware"):
                execution_scenario = audit_dynamic_shape(scenario)
                execution_scenario['policy'] = execution_policy
                inventory_audit = DynamicAudit(execution_scenario, findings)
                state = inventory_audit.replay(evidence["events.jsonl"])
                replay_complete = True
                summary = inventory_audit.reconcile(evidence)
            elif scenario["format_version"] == 2:
                execution_scenario = copy.deepcopy(scenario)
                execution_scenario['policy'] = execution_policy
                inventory_audit = InventoryAudit(execution_scenario, findings)
                state = inventory_audit.replay(evidence["events.jsonl"])
                replay_complete = True
                summary = inventory_audit.reconcile(evidence)
            else:
                state = replay_events(scenario, evidence["events.jsonl"], findings)
                replay_complete = True
                summary = reconcile_exports(state, evidence, findings)
        except ReplayViolation:
            pass
        except (EvidenceProblem, KeyError, TypeError, ValueError, IndexError) as error:
            add_finding(findings, "incomplete", "replay.evidence", "Complete replayable records", str(error))
        if scenario["format_version"] == 1 and "dispatch_decisions.jsonl" in evidence:
            try:
                audit_dispatch(scenario, evidence["events.jsonl"], evidence["dispatch_decisions.jsonl"], findings)
            except (EvidenceProblem, KeyError, TypeError, ValueError, IndexError) as error:
                add_finding(findings, "incomplete", "policy.evidence", "Complete dispatch audit", str(error))
    exit_code = 1 if any(f["category"] == "violation" for f in findings) else 2 if findings else 0
    report = {"status": {0: "PASS", 1: "FAIL", 2: "INCOMPLETE"}[exit_code], "exit_code": exit_code,
        "scope": "Checkpoint 4c: dynamic quality-aware/unaware insertion and plant forecasts" if execution_policy in ("dynamic_quality_unaware", "dynamic_quality_aware") else "Checkpoint 4a: static quality-aware/unaware selection with fixed physical trips" if scenario and scenario.get("format_version") == 2 else "Checkpoint 1: one fixed quality-unaware remanufacturing journey",
        "numeric_tolerance": {"absolute": TOLERANCE, "relative": TOLERANCE},
        "replay_complete": replay_complete, "recalculated_summary": summary, "findings": findings,
        "unchecked_scope": (["Custom starts away from facilities, insertions with more than two extra reset visits, experiments, full-program metrics and statistical distribution validation"] if execution_policy in ("dynamic_quality_unaware", "dynamic_quality_aware") else ["Multiple plants/collection points, custom starts away from facilities, repeated trips per AV, dynamic policies/replanning, experiments, full-program metrics and statistical distribution validation remain outside this verifier scope"])}
    if not replay_complete:
        report["unchecked_scope"].append("Replay and final totals could not be fully checked")
    write_reports(run_directory, report)
    return report


def main() -> int:
    """Parse a saved-run path, write verification reports and return the report's exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    arguments = parser.parse_args()
    if not arguments.run_directory.is_dir():
        print(f"INCOMPLETE: run directory does not exist: {arguments.run_directory}")
        return 2
    try:
        report = verify_run(arguments.run_directory)
    except OSError as error:
        print(f"INCOMPLETE: unable to read evidence or write reports: {error}")
        return 2
    print(f"{report['status']} (exit {report['exit_code']})")
    print(json.dumps(report["recalculated_summary"], indent=2))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
