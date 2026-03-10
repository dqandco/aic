# HPC Setup with Singularity

This guide is for running the AI for Industry Challenge on an HPC cluster where Docker is not available and only Singularity (or Apptainer) can be used for containers. You can skip the Distrobox step from the main [Getting Started](./getting_started.md) guide.

## Prerequisites

- **Singularity** (or Apptainer) installed on the cluster
- **Pixi** installed in your user space (no root required): `curl -fsSL https://pixi.sh/install.sh | sh`
- Access to a compute node with GPU (optional but recommended)

---

## Quick Start

### 1. Clone and install dependencies

```bash
mkdir -p ~/ws_aic/src
cd ~/ws_aic/src
git clone https://github.com/intrinsic-dev/aic
cd aic
pixi install
```

### 2. Pull the evaluation container

```bash
singularity pull aic_eval.sif docker://ghcr.io/intrinsic-dev/aic/aic_eval:latest
```

### 3. Run the evaluation container (headless)

```bash
singularity run --nv aic_eval.sif ground_truth:=false start_aic_engine:=true gazebo_gui:=false launch_rviz:=false
```

- `--nv` — Enable NVIDIA GPU support (omit if no GPU)
- `gazebo_gui:=false launch_rviz:=false` — Headless mode for HPC (no display required)

### 4. Run policy in a separate terminal

```bash
cd ~/ws_aic/src/aic
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.WaveArm
```

---

## Why This Works

- **Singularity**: Uses the host network by default. The Zenoh router listens on port 7447. The policy on the host connects to `localhost:7447`.
- **Headless mode**: Gazebo and RViz can run without a display (GUI disabled).
- **Pixi**: Runs entirely in user space; no root required.

---

## Troubleshooting

### Policy cannot connect to Zenoh router

Set the Zenoh connection explicitly before running the policy:

```bash
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=false;connect/endpoints=["tcp/localhost:7447"]'
export ZENOH_ROUTER_CHECK_ATTEMPTS=-1
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.WaveArm
```

### Job scheduling (SLURM, etc.)

For the eval container and policy to communicate, they must run on the same node or in the same allocation. Start the eval container first, then run the policy in the same allocation (e.g., in a second terminal after `srun` or in a separate job that shares the node).

### No GPU

Omit the `--nv` flag when running Singularity:

```bash
singularity run aic_eval.sif ground_truth:=false start_aic_engine:=true gazebo_gui:=false launch_rviz:=false
```

---

## Build from Source

The [Building from Source](./build_eval.md) guide requires `sudo` and system-wide package installation. It is not suitable for HPC without root. Use the pre-built Singularity image instead.
