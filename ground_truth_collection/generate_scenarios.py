#!/usr/bin/env python3
"""Generate randomized AIC engine trial configurations.

Produces YAML config files compatible with `aic_engine` by randomizing task board
pose, component placement on rails, and cable/task assignments within the
documented limits from the AIC task board specification.

Usage:
    python3 generate_scenarios.py --output config.yaml --num-trials 5
    python3 generate_scenarios.py --output config.yaml --num-trials 3 --seed 42
    python3 generate_scenarios.py --output config.yaml --trial-types sfp sfp sc
"""

import argparse
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# Limits from docs/task_board_description.md and aic_engine/config/sample_config.yaml
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RailLimits:
    min_translation: float
    max_translation: float


NIC_RAIL_LIMITS = RailLimits(min_translation=-0.0215, max_translation=0.0234)
SC_RAIL_LIMITS = RailLimits(min_translation=-0.06, max_translation=0.055)
MOUNT_RAIL_LIMITS = RailLimits(min_translation=-0.09425, max_translation=0.09425)

NIC_YAW_LIMITS = (-0.1745, 0.1745)  # ±10 degrees in radians
MOUNT_YAW_LIMITS = (-1.0472, 1.0472)  # ±60 degrees in radians

NUM_NIC_RAILS = 5
NUM_SC_RAILS = 2

# Task board pose ranges (based on sample_config values with reasonable variation)
TASK_BOARD_POSE_RANGES = {
    "x": (0.10, 0.25),
    "y": (-0.25, 0.05),
    "z": (1.10, 1.18),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (2.8, 3.5),
}

# Grasp offsets from qualification_phase.md
SFP_GRASP = {
    "x": 0.0, "y": 0.015385, "z": 0.04245,
    "roll": 0.4432, "pitch": -0.4838, "yaw": 1.3303,
}
SC_GRASP = {
    "x": 0.0, "y": 0.015385, "z": 0.04045,
    "roll": 0.4432, "pitch": -0.4838, "yaw": 1.3303,
}

# Small grasp deviation (~2mm, ~0.04 rad) per qualification_phase.md
GRASP_POS_DEVIATION = 0.002
GRASP_ORI_DEVIATION = 0.04

HOME_JOINT_POSITIONS = {
    "shoulder_pan_joint": -0.1597,
    "shoulder_lift_joint": -1.3542,
    "elbow_joint": -1.6648,
    "wrist_1_joint": -1.6933,
    "wrist_2_joint": 1.5710,
    "wrist_3_joint": 1.4110,
}

# Default scoring topics (from sample_config.yaml)
SCORING_TOPICS = [
    {"topic": {"name": "/joint_states", "type": "sensor_msgs/msg/JointState"}},
    {"topic": {"name": "/tf", "type": "tf2_msgs/msg/TFMessage"}},
    {"topic": {"name": "/tf_static", "type": "tf2_msgs/msg/TFMessage", "latched": True}},
    {"topic": {"name": "/scoring/tf", "type": "tf2_msgs/msg/TFMessage"}},
    {"topic": {"name": "/aic/gazebo/contacts/off_limit", "type": "ros_gz_interfaces/msg/Contacts"}},
    {"topic": {"name": "/fts_broadcaster/wrench", "type": "geometry_msgs/msg/WrenchStamped"}},
    {"topic": {"name": "/aic_controller/joint_commands", "type": "aic_control_interfaces/msg/JointMotionUpdate"}},
    {"topic": {"name": "/aic_controller/pose_commands", "type": "aic_control_interfaces/msg/MotionUpdate"}},
    {"topic": {"name": "/scoring/insertion_event", "type": "std_msgs/msg/String"}},
    {"topic": {"name": "/aic_controller/controller_state", "type": "aic_control_interfaces/msg/ControllerState"}},
]

TASK_BOARD_LIMITS = {
    "nic_rail": {"min_translation": NIC_RAIL_LIMITS.min_translation, "max_translation": NIC_RAIL_LIMITS.max_translation},
    "sc_rail": {"min_translation": SC_RAIL_LIMITS.min_translation, "max_translation": SC_RAIL_LIMITS.max_translation},
    "mount_rail": {"min_translation": MOUNT_RAIL_LIMITS.min_translation, "max_translation": MOUNT_RAIL_LIMITS.max_translation},
}


def rand_in(lo: float, hi: float) -> float:
    return round(random.uniform(lo, hi), 6)


def perturb_grasp(base: dict) -> dict:
    """Apply small random deviations to a grasp offset."""
    return {
        "x": round(base["x"] + random.uniform(-GRASP_POS_DEVIATION, GRASP_POS_DEVIATION), 6),
        "y": round(base["y"] + random.uniform(-GRASP_POS_DEVIATION, GRASP_POS_DEVIATION), 6),
        "z": round(base["z"] + random.uniform(-GRASP_POS_DEVIATION, GRASP_POS_DEVIATION), 6),
    }


def perturb_grasp_orientation(base: dict) -> dict:
    return {
        "roll": round(base["roll"] + random.uniform(-GRASP_ORI_DEVIATION, GRASP_ORI_DEVIATION), 4),
        "pitch": round(base["pitch"] + random.uniform(-GRASP_ORI_DEVIATION, GRASP_ORI_DEVIATION), 4),
        "yaw": round(base["yaw"] + random.uniform(-GRASP_ORI_DEVIATION, GRASP_ORI_DEVIATION), 4),
    }


def random_task_board_pose() -> dict:
    return {k: rand_in(*v) for k, v in TASK_BOARD_POSE_RANGES.items()}


def make_nic_rail(idx: int, present: bool) -> dict:
    if not present:
        return {"entity_present": False}
    return {
        "entity_present": True,
        "entity_name": f"nic_card_{idx}",
        "entity_pose": {
            "translation": rand_in(NIC_RAIL_LIMITS.min_translation, NIC_RAIL_LIMITS.max_translation),
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": round(random.uniform(*NIC_YAW_LIMITS), 4),
        },
    }


def make_sc_rail(idx: int, present: bool, name_suffix: Optional[int] = None) -> dict:
    if not present:
        return {"entity_present": False}
    suffix = name_suffix if name_suffix is not None else idx
    return {
        "entity_present": True,
        "entity_name": f"sc_mount_{suffix}",
        "entity_pose": {
            "translation": rand_in(SC_RAIL_LIMITS.min_translation, SC_RAIL_LIMITS.max_translation),
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": round(random.uniform(-0.15, 0.15), 4),
        },
    }


def make_mount_rail(mount_type: str, idx: int, present: bool, name_suffix: Optional[int] = None) -> dict:
    """mount_type is one of 'lc', 'sfp', 'sc'."""
    if not present:
        return {"entity_present": False}
    suffix = name_suffix if name_suffix is not None else idx
    return {
        "entity_present": True,
        "entity_name": f"{mount_type}_mount_{suffix}",
        "entity_pose": {
            "translation": rand_in(MOUNT_RAIL_LIMITS.min_translation, MOUNT_RAIL_LIMITS.max_translation),
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
        },
    }


def generate_sfp_trial(trial_idx: int) -> dict:
    """Generate a trial for SFP module insertion into a NIC card."""
    # Pick which NIC rail to use (1 or more, but exactly 1 is the target)
    target_nic_rail = random.randint(0, NUM_NIC_RAILS - 1)

    # Optionally add extra NIC cards on other rails
    extra_nic_rails = set()
    for i in range(NUM_NIC_RAILS):
        if i != target_nic_rail and random.random() < 0.25:
            extra_nic_rails.add(i)

    nic_rails = {}
    nic_name_counter = 0
    target_nic_name = None
    for i in range(NUM_NIC_RAILS):
        present = (i == target_nic_rail) or (i in extra_nic_rails)
        nic_rails[f"nic_rail_{i}"] = make_nic_rail(nic_name_counter, present)
        if present:
            if i == target_nic_rail:
                target_nic_name = f"nic_card_mount_{nic_name_counter}"
            nic_name_counter += 1

    # SC ports — random presence
    sc_rails = {}
    sc_counter = 0
    for i in range(NUM_SC_RAILS):
        present = random.random() < 0.5
        sc_rails[f"sc_rail_{i}"] = make_sc_rail(i, present, name_suffix=sc_counter)
        if present:
            sc_counter += 1

    # Mount rails — at least the sfp mount should exist for the pick location
    mount_rails = {
        "lc_mount_rail_0": make_mount_rail("lc", 0, random.random() < 0.6),
        "sfp_mount_rail_0": make_mount_rail("sfp", 0, True),
        "sc_mount_rail_0": make_mount_rail("sc", 0, random.random() < 0.5),
        "lc_mount_rail_1": make_mount_rail("lc", 1, random.random() < 0.5),
        "sfp_mount_rail_1": make_mount_rail("sfp", 1, random.random() < 0.3, name_suffix=1),
        "sc_mount_rail_1": make_mount_rail("sc", 1, random.random() < 0.3, name_suffix=1),
    }

    # Target SFP port: either port 0 or port 1 on the NIC card
    target_sfp_port = random.choice(["sfp_port_0", "sfp_port_1"])

    # Cable
    grasp_offset = perturb_grasp(SFP_GRASP)
    grasp_ori = perturb_grasp_orientation(SFP_GRASP)

    scene = {
        "task_board": {
            "pose": random_task_board_pose(),
            **nic_rails,
            **sc_rails,
            **mount_rails,
        },
        "cables": {
            "cable_0": {
                "pose": {
                    "gripper_offset": grasp_offset,
                    **grasp_ori,
                },
                "attach_cable_to_gripper": True,
                "cable_type": "sfp_sc_cable",
            },
        },
    }

    tasks = {
        "task_1": {
            "cable_type": "sfp_sc",
            "cable_name": "cable_0",
            "plug_type": "sfp",
            "plug_name": "sfp_tip",
            "port_type": "sfp",
            "port_name": target_sfp_port,
            "target_module_name": target_nic_name,
            "time_limit": 180,
        },
    }

    return {"scene": scene, "tasks": tasks}


def generate_sc_trial(trial_idx: int) -> dict:
    """Generate a trial for SC plug insertion into an SC port."""
    # Pick which SC rail to use as target
    target_sc_rail = random.randint(0, NUM_SC_RAILS - 1)

    # NIC rails — random presence (not the target, but adds clutter)
    nic_rails = {}
    nic_counter = 0
    for i in range(NUM_NIC_RAILS):
        present = random.random() < 0.2
        nic_rails[f"nic_rail_{i}"] = make_nic_rail(nic_counter, present)
        if present:
            nic_counter += 1

    # SC rails
    sc_rails = {}
    sc_counter = 0
    target_sc_name = None
    for i in range(NUM_SC_RAILS):
        present = (i == target_sc_rail) or (random.random() < 0.4)
        sc_rails[f"sc_rail_{i}"] = make_sc_rail(i, present, name_suffix=sc_counter)
        if present:
            if i == target_sc_rail:
                target_sc_name = f"sc_port_{sc_counter}"
            sc_counter += 1

    # Mount rails
    mount_rails = {
        "lc_mount_rail_0": make_mount_rail("lc", 0, random.random() < 0.5),
        "sfp_mount_rail_0": make_mount_rail("sfp", 0, random.random() < 0.6),
        "sc_mount_rail_0": make_mount_rail("sc", 0, True),
        "lc_mount_rail_1": make_mount_rail("lc", 1, random.random() < 0.5),
        "sfp_mount_rail_1": make_mount_rail("sfp", 1, random.random() < 0.3, name_suffix=1),
        "sc_mount_rail_1": make_mount_rail("sc", 1, random.random() < 0.3, name_suffix=1),
    }

    # Cable — reversed so SC plug end is grasped
    grasp_offset = perturb_grasp(SC_GRASP)
    grasp_ori = perturb_grasp_orientation(SC_GRASP)

    scene = {
        "task_board": {
            "pose": random_task_board_pose(),
            **nic_rails,
            **sc_rails,
            **mount_rails,
        },
        "cables": {
            "cable_0": {
                "pose": {
                    "gripper_offset": grasp_offset,
                    **grasp_ori,
                },
                "attach_cable_to_gripper": True,
                "cable_type": "sfp_sc_cable_reversed",
            },
        },
    }

    tasks = {
        "task_1": {
            "cable_type": "sfp_sc",
            "cable_name": "cable_0",
            "plug_type": "sc",
            "plug_name": "sc_tip",
            "port_type": "sc",
            "port_name": "sc_port_base",
            "target_module_name": target_sc_name,
            "time_limit": 180,
        },
    }

    return {"scene": scene, "tasks": tasks}


def generate_config(
    num_trials: int,
    trial_types: Optional[list[str]] = None,
    seed: Optional[int] = None,
) -> dict:
    """Generate a complete aic_engine config with the given number of trials.

    Args:
        num_trials: Number of trials to generate.
        trial_types: List of trial types ('sfp' or 'sc') per trial.
            If None, defaults to the qualification pattern: sfp, sfp, sc, repeating.
        seed: Random seed for reproducibility.
    """
    if seed is not None:
        random.seed(seed)

    if trial_types is None:
        # Default: follow qualification pattern (sfp, sfp, sc, ...)
        pattern = ["sfp", "sfp", "sc"]
        trial_types = [pattern[i % len(pattern)] for i in range(num_trials)]
    elif len(trial_types) != num_trials:
        raise ValueError(
            f"Length of trial_types ({len(trial_types)}) must match num_trials ({num_trials})"
        )

    generators = {
        "sfp": generate_sfp_trial,
        "sc": generate_sc_trial,
    }

    trials = {}
    for i, tt in enumerate(trial_types):
        gen = generators.get(tt)
        if gen is None:
            raise ValueError(f"Unknown trial type '{tt}'. Must be 'sfp' or 'sc'.")
        trials[f"trial_{i + 1}"] = gen(i)

    config = {
        "scoring": {"topics": SCORING_TOPICS},
        "task_board_limits": TASK_BOARD_LIMITS,
        "trials": trials,
        "robot": {"home_joint_positions": HOME_JOINT_POSITIONS},
    }

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Generate randomized AIC engine trial configurations."
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output YAML file path. Defaults to stdout.",
    )
    parser.add_argument(
        "-n", "--num-trials",
        type=int,
        default=3,
        help="Number of trials to generate (default: 3).",
    )
    parser.add_argument(
        "--trial-types",
        nargs="+",
        choices=["sfp", "sc"],
        default=None,
        help="Specify type for each trial (e.g., sfp sfp sc). "
             "Must match --num-trials count. "
             "Default: sfp, sfp, sc pattern.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()

    config = generate_config(
        num_trials=args.num_trials,
        trial_types=args.trial_types,
        seed=args.seed,
    )

    yaml_str = yaml.dump(config, default_flow_style=False, sort_keys=False)

    if args.output:
        with open(args.output, "w") as f:
            f.write(yaml_str)
        print(f"Generated {args.num_trials} trials -> {args.output}", file=sys.stderr)
    else:
        print(yaml_str)


if __name__ == "__main__":
    main()
