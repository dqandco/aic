#!/usr/bin/env python3
"""Generate 250 distinct training scenarios for the AIC simulation.

Each scenario randomises the task-board pose, component presence/placement,
cable type, and NIC card slot — producing a launch-parameter string that can
be passed directly to the entrypoint or ``ros2 launch``.

Usage::

    # Generate scenarios (writes to scenarios/ directory)
    python docker/my_policy/scripts/generate_scenarios.py

    # Generate to a custom directory
    python docker/my_policy/scripts/generate_scenarios.py --output-dir ~/training_scenarios

    # Run a single scenario (inside eval container)
    /entrypoint.sh $(cat scenarios/scenario_042.args) ground_truth:=true start_aic_engine:=false

    # Run from source build
    ros2 launch aic_bringup aic_gz_bringup.launch.py $(cat scenarios/scenario_042.args)

    # Batch: launch each scenario, collect expert demos, save SDF
    for f in scenarios/scenario_*.args; do
        idx=$(basename "$f" .args | sed 's/scenario_//')
        /entrypoint.sh $(cat "$f") &
        PID=$!
        sleep 15  # wait for sim to settle
        cp /tmp/aic.sdf "scenarios/scenario_${idx}.sdf"
        kill $PID 2>/dev/null; wait $PID 2>/dev/null
    done
"""

import argparse
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Parameter ranges (from aic_gz_bringup.launch.py and task_board.urdf.xacro)
# ---------------------------------------------------------------------------

# Task board position — keep within reachable workspace of UR5e
TASK_BOARD_X = (0.10, 0.35)  # metres
TASK_BOARD_Y = (-0.30, 0.05)
TASK_BOARD_Z = (1.10, 1.20)
TASK_BOARD_YAW = (2.5, 3.8)  # radians (roughly facing the robot)

# Mount rail translation limits (Y-axis slide)
MOUNT_RAIL_RANGE = (-0.09, 0.09)  # m, spec is ±0.09625 — stay inside

# SC port translation limits (X-axis slide)
SC_PORT_RANGE = (-0.05, 0.05)  # m, spec is ±0.055

# NIC card mount translation limits (X-axis slide)
NIC_MOUNT_RANGE = (-0.045, 0.035)  # m, spec is -0.048 to 0.036

CABLE_TYPES = ["sfp_sc_cable", "sfp_sc_cable_reversed"]

# Component types that can appear on mount rails
MOUNT_TYPES = ["lc", "sfp", "sc"]

# Number of NIC card mounts (0–4)
NIC_MOUNTS = range(5)

# Number of SC port rails (0–1)
SC_PORTS = range(2)

# Number of mount rails per type (0–1)
MOUNT_RAILS = range(2)


@dataclass
class Task:
    """Which plug to insert into which port."""

    cable_name: str = "cable_0"
    cable_type_short: str = "sfp_sc"
    plug_type: str = "sfp"
    plug_name: str = "sfp_tip"
    port_type: str = "sfp"
    port_name: str = "sfp_port_0"
    target_module_name: str = "nic_card_mount_0"

    def to_dict(self) -> dict:
        return {
            "cable_name": self.cable_name,
            "cable_type": self.cable_type_short,
            "plug_type": self.plug_type,
            "plug_name": self.plug_name,
            "port_type": self.port_type,
            "port_name": self.port_name,
            "target_module_name": self.target_module_name,
        }


@dataclass
class Scenario:
    """One complete simulation configuration."""

    idx: int
    seed: int

    # Task board pose
    tb_x: float = 0.15
    tb_y: float = -0.2
    tb_z: float = 1.14
    tb_yaw: float = 3.1415

    # Cable
    cable_type: str = "sfp_sc_cable"

    # Components: list of (kind, rail_idx, translation) tuples
    mount_rails: list = field(default_factory=list)
    sc_ports: list = field(default_factory=list)
    nic_mounts: list = field(default_factory=list)

    # Task: which plug goes into which port
    task: Task = field(default_factory=Task)

    def to_args(self) -> str:
        """Return a single-line launch parameter string."""
        parts = [
            "spawn_task_board:=true",
            f"task_board_x:={self.tb_x:.4f}",
            f"task_board_y:={self.tb_y:.4f}",
            f"task_board_z:={self.tb_z:.4f}",
            "task_board_roll:=0.0",
            "task_board_pitch:=0.0",
            f"task_board_yaw:={self.tb_yaw:.4f}",
            "spawn_cable:=true",
            f"cable_type:={self.cable_type}",
            "attach_cable_to_gripper:=true",
        ]

        # Mount rails
        for kind, rail_idx, trans in self.mount_rails:
            prefix = f"{kind}_mount_rail_{rail_idx}"
            parts.append(f"{prefix}_present:=true")
            parts.append(f"{prefix}_translation:={trans:.5f}")

        # SC ports
        for port_idx, trans in self.sc_ports:
            prefix = f"sc_port_{port_idx}"
            parts.append(f"{prefix}_present:=true")
            parts.append(f"{prefix}_translation:={trans:.5f}")

        # NIC card mounts
        for mount_idx, trans in self.nic_mounts:
            prefix = f"nic_card_mount_{mount_idx}"
            parts.append(f"{prefix}_present:=true")
            parts.append(f"{prefix}_translation:={trans:.5f}")

        return " ".join(parts)

    def to_dict(self) -> dict:
        return {
            "idx": self.idx,
            "seed": self.seed,
            "task_board": {
                "x": self.tb_x,
                "y": self.tb_y,
                "z": self.tb_z,
                "yaw": self.tb_yaw,
            },
            "cable_type": self.cable_type,
            "mount_rails": [
                {"type": k, "rail": r, "translation": t}
                for k, r, t in self.mount_rails
            ],
            "sc_ports": [
                {"port": p, "translation": t} for p, t in self.sc_ports
            ],
            "nic_mounts": [
                {"mount": m, "translation": t} for m, t in self.nic_mounts
            ],
            "task": self.task.to_dict(),
        }


def generate_scenario(idx: int, rng: random.Random) -> Scenario:
    """Generate one randomised scenario."""
    seed = rng.randint(0, 2**31)
    s = Scenario(idx=idx, seed=seed)

    # --- Task board pose ---
    s.tb_x = round(rng.uniform(*TASK_BOARD_X), 4)
    s.tb_y = round(rng.uniform(*TASK_BOARD_Y), 4)
    s.tb_z = round(rng.uniform(*TASK_BOARD_Z), 4)
    s.tb_yaw = round(rng.uniform(*TASK_BOARD_YAW), 4)

    # --- Cable type ---
    s.cable_type = rng.choice(CABLE_TYPES)

    # --- Mount rails: pick 1-4 mounts randomly ---
    # Each mount type (lc, sfp, sc) has 2 rails (0, 1).
    # We pick a random subset of (type, rail) combos.
    all_slots = [(t, r) for t in MOUNT_TYPES for r in MOUNT_RAILS]
    n_mounts = rng.randint(1, min(4, len(all_slots)))
    chosen_slots = rng.sample(all_slots, n_mounts)
    s.mount_rails = [
        (kind, rail, round(rng.uniform(*MOUNT_RAIL_RANGE), 5))
        for kind, rail in chosen_slots
    ]

    # --- SC ports: 0-2 ports ---
    n_sc = rng.randint(0, 2)
    if n_sc > 0:
        chosen_ports = rng.sample(list(SC_PORTS), n_sc)
        s.sc_ports = [
            (p, round(rng.uniform(*SC_PORT_RANGE), 5)) for p in chosen_ports
        ]

    # --- NIC card mounts: 1-3 mounts ---
    n_nic = rng.randint(1, 3)
    chosen_nics = rng.sample(list(NIC_MOUNTS), n_nic)
    s.nic_mounts = [
        (m, round(rng.uniform(*NIC_MOUNT_RANGE), 5)) for m in chosen_nics
    ]

    # --- Task: derive from cable type and available components ---
    if s.cable_type == "sfp_sc_cable":
        # SFP end inserts into NIC card SFP port
        target_nic = rng.choice(chosen_nics)
        s.task = Task(
            cable_name="cable_0",
            cable_type_short="sfp_sc",
            plug_type="sfp",
            plug_name="sfp_tip",
            port_type="sfp",
            port_name="sfp_port_0",
            target_module_name=f"nic_card_mount_{target_nic}",
        )
    else:
        # sfp_sc_cable_reversed: SC end inserts into SC port
        if s.sc_ports:
            target_port = rng.choice(s.sc_ports)[0]
        else:
            # Ensure at least one SC port exists
            target_port = rng.choice(list(SC_PORTS))
            s.sc_ports.append(
                (target_port, round(rng.uniform(*SC_PORT_RANGE), 5))
            )
        s.task = Task(
            cable_name="cable_0",
            cable_type_short="sfp_sc",
            plug_type="sc",
            plug_name="sc_tip",
            port_type="sc",
            port_name="sc_port_base",
            target_module_name=f"sc_port_{target_port}",
        )

    return s


def main():
    parser = argparse.ArgumentParser(
        description="Generate 250 distinct AIC training scenarios"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scenarios"),
        help="Directory to write scenario files (default: scenarios/)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=250,
        help="Number of scenarios to generate (default: 250)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    scenarios = []

    # Track uniqueness via the args string
    seen = set()
    attempts = 0
    while len(scenarios) < args.count and attempts < args.count * 10:
        attempts += 1
        s = generate_scenario(len(scenarios), rng)
        key = s.to_args()
        if key not in seen:
            seen.add(key)
            scenarios.append(s)

    # Write individual .args files (one line each)
    for s in scenarios:
        args_file = out / f"scenario_{s.idx:03d}.args"
        args_file.write_text(s.to_args() + "\n")

    # Write manifest JSON with all scenarios
    manifest = {
        "master_seed": args.seed,
        "count": len(scenarios),
        "scenarios": [s.to_dict() for s in scenarios],
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    # Write a convenience bash runner
    runner_path = out / "run_all.sh"
    runner_path.write_text(
        """\
#!/usr/bin/env bash
# Run all scenarios sequentially, saving the exported SDF for each.
# Usage: bash scenarios/run_all.sh [extra launch args...]
#
# Example:
#   bash scenarios/run_all.sh ground_truth:=true start_aic_engine:=false

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXTRA_ARGS="${*:-ground_truth:=true start_aic_engine:=false}"
SETTLE_TIME="${SETTLE_TIME:-20}"

echo "Running $(ls "$SCRIPT_DIR"/scenario_*.args | wc -l) scenarios"
echo "Extra args: $EXTRA_ARGS"
echo "Settle time: ${SETTLE_TIME}s"
echo ""

for ARGS_FILE in "$SCRIPT_DIR"/scenario_*.args; do
    IDX=$(basename "$ARGS_FILE" .args | sed 's/scenario_//')
    echo "=== Scenario $IDX ==="

    SCENARIO_ARGS=$(cat "$ARGS_FILE")
    /entrypoint.sh $SCENARIO_ARGS $EXTRA_ARGS &
    SIM_PID=$!

    sleep "$SETTLE_TIME"

    if [ -f /tmp/aic.sdf ]; then
        cp /tmp/aic.sdf "$SCRIPT_DIR/scenario_${IDX}.sdf"
        echo "  Saved SDF -> scenario_${IDX}.sdf"
    fi

    kill $SIM_PID 2>/dev/null || true
    wait $SIM_PID 2>/dev/null || true
    sleep 2
done

echo ""
echo "Done. $(ls "$SCRIPT_DIR"/scenario_*.sdf 2>/dev/null | wc -l) SDFs saved."
"""
    )
    runner_path.chmod(0o755)

    print(f"Generated {len(scenarios)} scenarios in {out}/")
    print(f"  - {len(scenarios)} .args files  (launch parameters)")
    print(f"  - manifest.json          (structured metadata)")
    print(f"  - run_all.sh             (batch runner)")
    print()
    print("Quick start:")
    print(f"  # Preview a scenario")
    print(f"  cat {out}/scenario_000.args")
    print()
    print(f"  # Launch one scenario")
    print(
        f"  /entrypoint.sh $(cat {out}/scenario_042.args) "
        "ground_truth:=true start_aic_engine:=false"
    )


if __name__ == "__main__":
    main()
