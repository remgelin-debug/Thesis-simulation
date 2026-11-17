# Thesis simulation — Checkpoint 6

The three-site pilot models three collection points, three autonomous vehicles,
and three remanufacturing facilities with three workstations each.

Run and verify the pilot from the repository folder:

```powershell
python -m simulation --config configs/three_site_pilot.json --seed 42 --policy dynamic_quality_aware --output results/pilot-42
python verify_simulation.py results/pilot-42
```

Choose a different seed and output folder for another run. The simulation
writes its results locally under `results/`.
