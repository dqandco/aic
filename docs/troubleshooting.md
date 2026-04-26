# Troubleshooting

## Low real-time factor on Gazebo

The simulation is configured to run at **1.0 RTF (100% real-time factor)**, meaning simulation time should match wall-clock time. If you're experiencing lower RTF, the following sections may help diagnose and resolve the issue.

### Gazebo not using the dedicated GPU

If your machine has two GPUs (or a CPU with an integrated GPU), OpenGL may be using the *integrated* GPU for rendering, which causes RTF to be very low. To fix this, you may need to manually force it to use the *discrete* GPU.

To check if Open GL is using the discrete GPU, run `glxinfo -B`. The output should show the details of your discrete GPU. Additionally, you can verify GPU-specific process by running `nvidia-smi`. When the AIC sim is active, `gz sim` should appear in the process list.

If the wrong GPU is selected, run `sudo prime-select nvidia`.
**Note**: You must log out and log in again for the changes to take effect. Then, re-run `glxinfo -B` to verify that the discrete GPU is active.

You can also check out [Problems with dual Intel and Nvidia GPU systems](https://gazebosim.org/docs/latest/troubleshooting/#problems-with-dual-intel-and-nvidia-gpu-systems).

### No GPU Available

If your system doesn't have a dedicated GPU, you may experience poor real-time factor (RTF) performance. This is because Gazebo uses [GlobalIllumination (GI)](https://gazebosim.org/api/sim/9/global_illumination.html) based rendering for the AIC scene, which requires GPU acceleration for optimal performance.

**To improve simulation performance on systems without a GPU:**

You can disable GlobalIllumination by editing [`aic.sdf`](../aic_description/world/aic.sdf) and setting `<enabled>` to `false` in the global illumination configuration [here](https://github.com/intrinsic-dev/aic/blob/c8aa4571d9dc4bd55bbefc02b0a160ba0e8e1e90/aic_description/world/aic.sdf#L39) and [here](https://github.com/intrinsic-dev/aic/blob/c8aa4571d9dc4bd55bbefc02b0a160ba0e8e1e90/aic_description/world/aic.sdf#L109). This will reduce rendering quality but may significantly improve RTF on CPU-only systems.

> [!WARNING]
> Disabling GI will change the visual appearance of the scene, which may affect vision-based policies.

## Zenoh Shared Memory Watchdog Warnings

When running the system, you may see warnings like:

```
WARN Watchdog Validator ThreadId(17) zenoh_shm::watchdog::periodic_task:
error setting scheduling priority for thread: OS(1), will run with priority 48.
This is not an hard error and it can be safely ignored under normal operating conditions.
```

**This warning is harmless and can be safely ignored.** It indicates that Zenoh's shared memory watchdog thread couldn't set a higher scheduling priority (which requires elevated privileges). The system will continue to work correctly.

**Why it happens:**
- The watchdog thread monitors shared memory health
- Setting higher priority requires `CAP_SYS_NICE` capability or root privileges
- Without it, the thread runs at default priority (48)

**When it might matter:**
- Under extremely high CPU load, the watchdog may occasionally miss its deadlines
- This could cause rare timeouts in shared memory operations
- In practice, this is almost never an issue for typical workloads

**To verify shared memory is working:**
```bash
# Check for Zenoh shared memory files
ls -lh /dev/shm | grep zenoh

# Monitor network traffic (should be minimal)
sudo tcpdump -i lo port 7447 -v
```

If you see Zenoh files in `/dev/shm` and minimal traffic on port 7447, shared memory is functioning correctly despite the warning.

## NVIDIA RTX 50xx cards not supported by PyTorch version locked in Pixi

```
UserWarning:
NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation.

The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.
If you want to use the NVIDIA GeForce RTX 5090 GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/
```

The `lerobot` version in `pixi.toml` depends on an older version of `pytorch` (built for an older version of cuda). 
`pixi install` will pull in that older version which does not support the newer sm_120 architecture for NVIDIA RTX 50xx cards.

We were able to run this policy on an Nvidia RTX 5090 by adding the following to `pixi.toml`:
```
[pypi-options.dependency-overrides]
torch = ">=2.7.1"
torchvision = ">=0.22.1"
```

See this [LeRobot issue](https://github.com/huggingface/lerobot/issues/2217) for details.

## Error: no such container aic_eval

when running `distrobox enter -r aic_eval`, you might encounter the following error:
```bash
Error: no such container aic_eval
```

By default, distrobox uses podman but we are using docker in our setup. Make sure to have set the default container manager by exporting the `DBX_CONTAINER_MANAGER` environment variable:
```bash
export DBX_CONTAINER_MANAGER=docker
```

## Stuck simulator processes ("multiple arms in Gazebo", "Another world of the same name is running", `aic_engine`: "Failed to find a valid clock")

If a previous `entrypoint.sh` was killed with `Ctrl+C`, gz often refuses to die — `gz sim` ignores `SIGTERM` for ~15 s and the actual world is hosted *inside* `component_container` (loaded as a composable node), which `pkill -f "gz sim"` does not match. As a result, the next `entrypoint.sh` cheerfully starts a second simulator alongside the zombie one, you see multiple arms in Gazebo, and `aic_engine` gives up with `Failed to find a valid clock` because two different `/clock` publishers are now fighting on the bus.

Symptoms:
- More than one robot arm in the Gazebo viewer
- `aic_engine` log ends with `Failed to find a valid clock` followed by `Engine failed to initialize`
- `[component_container-3] (...) [error] [SimulationRunner.cc:207] Another world of the same name is running` in the entrypoint log
- `run_gazebo_act_episode.py` returns `failure_reason: "aic_engine completed without producing scoring.yaml."`

Use the bundled cleanup script to kill **everything** the eval stack tends to leak (host-side ROS nodes, the in-container `component_container` / `gz_server` / `controller_manager` / `rmw_zenohd`, and orphan `podman` `conmon` processes whose container has already been deleted):

```bash
# Inspect what's still alive (exits non-zero if there is anything to kill).
docker/my_policy/scripts/clean_sim.sh --check

# Actually clean it up.
docker/my_policy/scripts/clean_sim.sh

# Same as above but also `docker restart aic_eval` — kicks any open
# `distrobox enter` shell, but guarantees a fresh container PID 1.
docker/my_policy/scripts/clean_sim.sh --hard
```

Once `--check` reports `host=0, container=0, conmon=0`, re-run `entrypoint.sh` in your `aic_eval` shell **before** launching another episode:

```bash
distrobox enter aic_eval
/entrypoint.sh ground_truth:=false start_aic_engine:=false
```

`run_gazebo_act_episode.py` and `train_gazebo_residual_act.py` now also fail fast (in ~5 s) with the message `No running sim detected (...)` instead of waiting for `aic_engine` to time out, so if you forget to start the entrypoint you'll get a clear error pointing back here.

## `aic_engine`: "Failed to find a valid clock" with a healthy sim (RMW mismatch)

Symptoms — a single, clean simulator is running (so the multi-arm / `Another world` advice above does not apply), `clean_sim.sh --check` reports `host=0, container=0, conmon=0` after you `Ctrl+C` everything, but `run_gazebo_act_episode.py` still fails after ~17–20 s with:

```
"failure_reason": "aic_engine completed without producing scoring.yaml.",
"engine_return_code": 1,
```

and `outputs/<run>/logs/aic_engine.log` ends with:

```
[INFO] [aic_engine]: Waiting for clock
[ERROR] [aic_engine]: Failed to find a valid clock
[ERROR] [aic_engine]: Engine failed to initialize
```

Cause: `/entrypoint.sh` exports `RMW_IMPLEMENTATION=rmw_zenoh_cpp` only inside *its own* process tree, so a fresh `distrobox enter aic_eval -- bash -lc 'source setup.bash && ros2 run aic_engine ...'` shell falls back to the default RMW (FastDDS) and cannot see `/clock` (or any other topic) the sim is publishing over Zenoh. You can confirm by running, inside the container:

```bash
docker exec aic_eval bash -lc 'source /ws_aic/install/setup.bash && \
  ros2 service list | grep gz_server'      # empty == default RMW, broken
docker exec aic_eval bash -lc 'source /ws_aic/install/setup.bash && \
  export RMW_IMPLEMENTATION=rmw_zenoh_cpp && ros2 service list | grep gz_server'
                                            # 18 services == sim is fine
```

The runner now exports `RMW_IMPLEMENTATION=rmw_zenoh_cpp` and `ZENOH_ROUTER_CONFIG_URI=/aic_zenoh_config.json5` for the `aic_engine` command, and a second preflight check verifies `/clock` actually has a publisher visible from inside the container before launching the engine. If you ever hit this manually (e.g. running `aic_engine` by hand in a one-off shell), make sure to set both env vars first.

## `aic_engine`: "Participant model is not ready" / `aic_engine exited with a non-zero return code`

Symptoms — the engine *did* find the clock and started trial 1, then bailed out almost immediately:

```
"failure_reason": "aic_engine exited with a non-zero return code.",
"total_score": 0.0,
"score_lines": [
    "trial_1: total=0.000 (tier1=0.000, tier2=0.000, tier3=0.000)"
]
```

`outputs/<run>/logs/aic_engine.log`:

```
[INFO]  Service '/aic_model/get_state' is available. Participant model discovered.
[INFO]  Lifecycle node 'aic_model' is available. Checking if it is in 'unconfigured' state...
[ERROR] GetState service call timed out for node 'aic_model'
[ERROR]   ✗ Participant model is not ready for trial 'trial_1'
```

`outputs/<run>/logs/aic_model.log`:

```
[INFO] aic_model: Loading policy module: my_policy.ResidualACT
... 28 seconds later ...
[INFO] aic_model: Loaded policy module my_policy.ResidualACT
```

Cause: `aic_engine` only waits 5 s for `/aic_model/get_state` to answer before declaring the participant unready and aborting the trial. ResidualACT (and any policy that imports PyTorch + loads a checkpoint synchronously in `__init__`) routinely takes 25–35 s to construct, during which the lifecycle service is *advertised* on the ROS graph but the rclpy executor is busy and never dispatches the request.

The runner now polls `/aic_model/get_state` itself (with a generous default timeout of 90 s — see `EpisodeSpec.model_ready_timeout`) before launching the engine, so the engine only ever sees a fully-spun-up model. If your policy takes longer than 90 s to construct, bump the timeout:

```python
EpisodeSpec(..., model_ready_timeout=180.0)
```

If you hit `aic_model did not answer /aic_model/get_state within Ns` instead of the engine error, the policy is either still loading (raise the timeout) or it crashed during import (check `aic_model.log` for a Python traceback).
