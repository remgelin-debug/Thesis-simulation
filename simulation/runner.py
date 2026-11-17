"""Single-run orchestration and versioned audit artifact serialization."""
from dataclasses import asdict
import copy
import hashlib
import json
from pathlib import Path
from .config import load_scenario
from .engine import run_simulation
from .models import Record, SimulationState
from .inventory_engine import run_inventory_simulation, inventory_summary
from .return_generation import prepare_scenario
from .dynamic_config import as_dynamic_scenario
from .dynamic_engine import run_dynamic_simulation


def write_json(path: Path, value: object) -> None:
    """Write one readable UTF-8 JSON file, replacing its previous contents."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_once(scenario_path: Path, output_directory: Path, *, seed: int | None = None, policy: str | None = None) -> Record:
    """Resolve or load one scenario, execute it, save a new run directory and return summary."""
    scenario, scenario_bytes = load_scenario(scenario_path) if seed is None else prepare_scenario(scenario_path, seed)
    selected_policy = scenario['policy'] if policy is None else policy
    if selected_policy not in ('static_quality_unaware', 'static_quality_aware', 'dynamic_quality_unaware', 'dynamic_quality_aware') or (scenario['format_version'] == 1 and selected_policy != 'static_quality_unaware'):
        raise ValueError('Selected policy is unsupported for this scenario format')
    if scenario['format_version'] == 3 and selected_policy not in ('dynamic_quality_unaware', 'dynamic_quality_aware'):
        raise ValueError('Format three requires a supported dynamic policy')
    execution_scenario = as_dynamic_scenario(scenario) if selected_policy in ('dynamic_quality_unaware', 'dynamic_quality_aware') else copy.deepcopy(scenario)
    execution_scenario['policy'] = selected_policy
    output_directory.mkdir(parents=True, exist_ok=False)
    fingerprint = hashlib.sha256(scenario_bytes).hexdigest()
    manifest = {"run_id": output_directory.name, "policy": selected_policy,
        "input_format_version": scenario["format_version"], "output_format_version": execution_scenario["format_version"],
        "scenario_sha256": fingerprint, "seed_provenance": scenario["seed_provenance"],
        "status": "running"}
    (output_directory / "scenario.json").write_bytes(scenario_bytes)
    write_json(output_directory / "run_manifest.json", manifest)
    if selected_policy in ('dynamic_quality_unaware', 'dynamic_quality_aware'):
        state = run_dynamic_simulation(execution_scenario)
    else:
        state = run_inventory_simulation(execution_scenario) if scenario["format_version"] == 2 else run_simulation(execution_scenario)
    for filename, records in (("events.jsonl", state.events), ("dispatch_decisions.jsonl", state.decisions)):
        with (output_directory / filename).open("w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
    final_state = asdict(state)
    del final_state["events"], final_state["decisions"]
    if not final_state.get('plant_states'):
        final_state.pop('plant_states', None)
    if scenario["format_version"] in (2, 3):
        summary = inventory_summary(state)
    else:
        summary = legacy_summary(state)
    if scenario["format_version"] == 1:
        final_state["product_id"] = scenario["product"]["id"]
    final_state["summary"] = summary
    write_json(output_directory / "final_state.json", final_state)
    write_json(output_directory / "summary.json", summary)
    manifest["status"] = "completed"
    write_json(output_directory / "run_manifest.json", manifest)
    return summary


def legacy_summary(state: SimulationState) -> Record:
    """Derive unchanged version-one summary fields from its single-product state."""
    completed = int(state.product_state == "remanufactured")
    summary = {"generated": int(state.product_state != "unreleased"), "remanufactured": completed,
        "recycled": 0, "unfinished": int(state.product_state not in ("unreleased", "remanufactured")),
        "vehicle_distance": state.vehicle_distance, "plant_inventory_wait": state.inventory_wait}
    return summary
