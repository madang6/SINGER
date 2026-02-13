# SSV Multi-3DGS Campaign Notebooks

## Simulating a Single Trajectory

### Command

```bash
python notebooks/ssv_multi3dgs_campaign.py simulate --config-file configs/experiment/ssv_multi3dgs.yml
```

Add `--verbose` for detailed simulation output.

### What controls "single trajectory"

The `simulate` command calls `simulate_roster()` from `sousvide.flight.deploy_ssv`, which loops over three nested dimensions:

1. **Flights** — `(scene_name, course_name)` pairs from the experiment config
2. **Objectives** — `queries` list from the scene's YAML (e.g., `["microwave", "ladder", "armchair"]`)
3. **Pilots** — `["expert"] + roster` (expert is always prepended)

Each combination gets simulated `test_set.reps` times (from the method JSON config).

### To get exactly one trajectory

Edit the experiment config (`configs/experiment/ssv_multi3dgs.yml`):

```yaml
cohort: "ssv_CLIPSEG_NORMAL"
method: "rrt"
review: false

flights:
  - ["sv_917_3_left_gemsplat", "sv_917_3_left_gemsplat"]  # single flight

roster: []  # empty = expert pilot only
```

Then trim the scene config (`configs/scenes/sv_917_3_left_gemsplat.yml`) to one objective:

```yaml
queries: ["microwave"]       # just one query
radii:
  - [1.75, 0.4]
altitudes:
  - -1.0
nbranches:
  - 110
```

This gives you: **1 flight x 1 objective x 1 pilot (expert) x 1 rep = 1 trajectory**.

### Using a pre-generated trajectory

Set `review: true` in the experiment config to skip RRT generation and load a saved `.pkl` trajectory file from disk instead of generating a new one.

### Key config files

| File | Controls |
|------|----------|
| `configs/experiment/ssv_multi3dgs.yml` | `cohort`, `method`, `flights`, `roster`, `review` |
| `configs/scenes/{scene}.yml` | `queries`, `radii`, `altitudes`, `nbranches` |
| `configs/method/rrt.json` | `test_set.reps`, duration, rate, policy |
