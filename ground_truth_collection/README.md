# Ground Truth Collection

Tools for generating randomized AIC trial configs and recording expert demonstrations into a LeRobot dataset.

## Prerequisites

- Running inside the `aic_eval` distrobox container
- Simulation environment built and workspace sourced (`/ws_aic/install/setup.bash`)
- `HF_USER` environment variable set to your HuggingFace username

## Workflow

### 1. Generate scenario configs

```bash
# Default: 3 trials (sfp, sfp, sc pattern)
pixi run python ground_truth_collection/generate_scenarios.py \
    -o ground_truth_collection/my_config.yaml --seed 42

# 10 trials with custom sequence
pixi run python ground_truth_collection/generate_scenarios.py \
    -n 10 --trial-types sfp sc sfp sc sfp sc sfp sc sfp sc \
    -o ground_truth_collection/my_config.yaml
```

The generated YAML follows the `aic_engine` config format (see `aic_engine/config/sample_config.yaml`). Task board pose, component placement on rails, and grasp offsets are randomized within documented limits.

#### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-o, --output` | stdout | Output YAML file path |
| `-n, --num-trials` | 3 | Number of trials to generate |
| `--trial-types` | sfp, sfp, sc (repeating) | Per-trial type: `sfp` or `sc` |
| `--seed` | None | Random seed for reproducibility |

### 2. Start the simulation

In a separate terminal, start the sim with an **empty world** (no task board or cable -- `aic_engine` will spawn them per trial):

```bash
/entrypoint.sh spawn_task_board:=false spawn_cable:=false \
    ground_truth:=true start_aic_engine:=false
```

All four flags are required:
- `spawn_task_board:=false` -- entities are spawned per trial by the engine
- `spawn_cable:=false` -- same
- `ground_truth:=true` -- publishes ground truth TF frames needed by the expert policy
- `start_aic_engine:=false` -- the recording script launches the engine itself

### 3. Record demonstrations

```bash
HF_USER=youruser pixi run python ground_truth_collection/record_trials.py \
    --config ground_truth_collection/my_config.yaml \
    --policy my_policy.ExpertCollector
```

The script:
1. Launches `aic_model` with the specified policy
2. Launches `aic_engine` with the config (handles spawning, lifecycle, goal dispatch, cleanup)
3. Records observations and actions at 4 Hz into a LeRobot dataset
4. Detects episode boundaries when `aic_engine` resets joints to home between trials
5. Stops when `aic_engine` finishes all trials

#### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | (required) | Path to generated YAML config |
| `--policy` | (required) | Policy module for aic_model (e.g. `my_policy.ExpertCollector`) |
| `--dataset-repo-id` | `$HF_USER/aic_expert_demos` | LeRobot dataset repo id |
| `--resume` | false | Resume recording into an existing dataset |
| `--push-to-hub` | false | Push dataset to HuggingFace Hub when done |

#### Logs

Process output is written to `recording_logs/`:
- `aic_model.log` -- policy node output
- `aic_engine.log` -- trial orchestration output

### 4. (Optional) Push to Hub

Either pass `--push-to-hub` during recording, or push manually afterward:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("youruser/aic_expert_demos")
ds.push_to_hub(private=True)
```

## Dataset format

Each episode corresponds to one trial. Frames are recorded at 4 Hz and contain:

- **Observations**: TCP pose (7), TCP velocity (6), TCP error (6), joint positions (7), 3 camera images (downscaled to 288x256)
- **Actions**: Cartesian velocity commands (linear xyz + angular xyz = 6)
- **Task**: `"insert cable into port"`
