"""Load resolved inputs and validate legacy or inventory-run format versions."""
import json
import math
from pathlib import Path
from .models import Record


def load_scenario(path: Path) -> tuple[Record, bytes]:
    """Read original bytes and validate the selected resolved scenario format."""
    scenario_bytes = path.read_bytes()
    scenario = json.loads(scenario_bytes)
    validate_scenario(scenario)
    return scenario, scenario_bytes


def validate_scenario(scenario: Record) -> None:
    """Reject unsupported scope and invalid numerical or route inputs; change nothing."""
    if scenario["format_version"] == 3:
        from .dynamic_config import validate_dynamic_scenario
        validate_dynamic_scenario(scenario)
        return
    if scenario["format_version"] == 2:
        from .inventory_config import validate_inventory_scenario
        validate_inventory_scenario(scenario)
        return
    if scenario["format_version"] != 1 or scenario["policy"] != "static_quality_unaware":
        raise ValueError("Checkpoint 1 supports format 1 and static_quality_unaware only")
    plant, point, vehicle, product, trip = (scenario[key] for key in
        ("plant", "collection_point", "vehicle", "product", "trip"))
    if plant["inventory_capacity"] is not None or vehicle["maximum_range"] is not None:
        raise ValueError("Checkpoint 1 requires unlimited inventory and range")
    if scenario["recycling_facilities"] or product["disposition"] != "remanufacture":
        raise ValueError("Checkpoint 1 supports remanufacturing only")
    identifiers = [plant["id"], point["id"], vehicle["id"], product["id"]]
    if len(set(identifiers)) != 4 or not all(isinstance(i, str) and i for i in identifiers):
        raise ValueError("Entity IDs must be distinct nonempty strings")
    if vehicle["initial_site"] != plant["id"] or trip["stops"] != [point["id"], plant["id"]]:
        raise ValueError("The single trip must go from the plant to collection and back")
    if product["origin"] != point["id"]:
        raise ValueError("Product origin must reference the configured collection point")
    workstation_ids = [station["id"] for station in plant["workstations"]]
    if len(set(workstation_ids)) != len(workstation_ids) or any(
        not isinstance(identifier, str) or not identifier for identifier in workstation_ids
    ):
        raise ValueError("Workstation IDs must be unique nonempty strings")
    if vehicle["payload_capacity"] != 1:
        raise ValueError("Checkpoint 1 requires payload capacity one")
    numbers = [scenario["horizon"], product["release_time"], trip["departure_time"], vehicle["speed"]]
    for site in (plant, point):
        if len(site["coordinates"]) != 2:
            raise ValueError("Coordinates must have two components")
        numbers.extend(site["coordinates"])
    operations = [station["operation_id"] for station in plant["workstations"]]
    if len(set(operations)) != len(operations) or not product["operations"] or any(
        operation not in operations for operation in product["operations"]
    ):
        raise ValueError("Operations require one compatible workstation each")
    for station in plant["workstations"]:
        numbers.append(station["duration"])
        previous_end = -math.inf
        for start, end in station["windows"]:
            numbers.extend((start, end))
            if start < 0 or start >= end or start <= previous_end:
                raise ValueError("Windows must be positive-length, ordered and separated")
            previous_end = end
        if station["duration"] <= 0:
            raise ValueError("Processing durations must be positive")
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in numbers):
        raise ValueError("All times, coordinates and speeds must be finite numbers")
    if vehicle["speed"] <= 0 or not 0 <= product["release_time"] <= trip["departure_time"] <= scenario["horizon"]:
        raise ValueError("Require positive speed and release <= departure <= horizon")
