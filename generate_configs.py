#!/usr/bin/env python3
"""
Randomized config generator for AIC Challenge training data collection.

Generates YAML trial configs that the aic_engine can consume, with randomized:
  - Task board pose (x, y, yaw)
  - NIC card rail selection, translation, and yaw offset
  - SC port rail selection and translation
  - Mount rail fixtures (optional distractors)
  - Grasp pose perturbation (~2mm, ~0.04 rad)
  - Task type balancing (SFP_PORT_0, SFP_PORT_1, SC_PORT)

Usage:
    python generate_configs.py --num-trials 100 --output-dir ./configs --seed 42
    python generate_configs.py --num-trials 50 --task-type sfp_port_0 --output-dir ./configs
    python generate_configs.py --num-trials 90 --trials-per-config 3 --output-dir ./configs
"""

import argparse
import os
import random
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Documented limits
# ---------------------------------------------------------------------------

TASK_BOARD_POSE = {
    "x": (0.12, 0.22),
    "y": (-0.25, 0.10),
    "z": 1.14,
    "roll": 0.0,
    "pitch": 0.0,
    "yaw": (2.7, 3.3),
}

NIC_RAIL = {
    "count": 5,
    "translation": (-0.0215, 0.0234),
    "yaw": (-0.1745, 0.1745),  # +/-10 deg
}

SC_RAIL = {
    "count": 2,
    "translation": (-0.06, 0.055),
}

MOUNT_RAIL = {
    "translation": (-0.09425, 0.09425),
    "yaw": (-1.0472, 1.0472),  # +/-60 deg
}

GRASP_NOISE = {
    "position": 0.002,   # metres
    "orientation": 0.04,  # radians
}

NOMINAL_GRASP = {
    "sfp": {"x": 0.0, "y": 0.015385, "z": 0.04245,
            "roll": 0.4432, "pitch": -0.4838, "yaw": 1.3303},
    "sc":  {"x": 0.0, "y": 0.015385, "z": 0.04045,
            "roll": 0.4432, "pitch": -0.4838, "yaw": 1.3303},
}

ROBOT_HOME = {
    "shoulder_pan_joint":  -0.1597,
    "shoulder_lift_joint": -1.3542,
    "elbow_joint":         -1.6648,
    "wrist_1_joint":       -1.6933,
    "wrist_2_joint":        1.5710,
    "wrist_3_joint":        1.4110,
}

SCORING_TOPICS = [
    {"topic": {"name": "/joint_states",
               "type": "sensor_msgs/msg/JointState"}},
    {"topic": {"name": "/tf",
               "type": "tf2_msgs/msg/TFMessage"}},
    {"topic": {"name": "/tf_static",
               "type": "tf2_msgs/msg/TFMessage",
               "latched": True}},
    {"topic": {"name": "/scoring/tf",
               "type": "tf2_msgs/msg/TFMessage"}},
    {"topic": {"name": "/aic/gazebo/contacts/off_limit",
               "type": "ros_gz_interfaces/msg/Contacts"}},
    {"topic": {"name": "/fts_broadcaster/wrench",
               "type": "geometry_msgs/msg/WrenchStamped"}},
    {"topic": {"name": "/aic_controller/joint_commands",
               "type": "aic_control_interfaces/msg/JointMotionUpdate"}},
    {"topic": {"name": "/aic_controller/pose_commands",
               "type": "aic_control_interfaces/msg/MotionUpdate"}},
    {"topic": {"name": "/scoring/insertion_event",
               "type": "std_msgs/msg/String"}},
    {"topic": {"name": "/aic_controller/controller_state",
               "type": "aic_control_interfaces/msg/ControllerState"}},
]

TASK_BOARD_LIMITS = {
    "nic_rail": {"min_translation": -0.0215, "max_translation": 0.0234},
    "sc_rail":  {"min_translation": -0.06,   "max_translation": 0.055},
    "mount_rail": {"min_translation": -0.09425, "max_translation": 0.09425},
}

TASK_TYPES = ["sfp_port_0", "sfp_port_1", "sc_port"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform(rng: random.Random, lo: float, hi: float) -> float:
    return round(rng.uniform(lo, hi), 6)


def _perturb(rng: random.Random, nominal: float, noise: float) -> float:
    return round(nominal + rng.uniform(-noise, noise), 6)


def _make_task_board_pose(rng: random.Random) -> dict:
    return {
        "x": _uniform(rng, *TASK_BOARD_POSE["x"]),
        "y": _uniform(rng, *TASK_BOARD_POSE["y"]),
        "z": TASK_BOARD_POSE["z"],
        "roll": TASK_BOARD_POSE["roll"],
        "pitch": TASK_BOARD_POSE["pitch"],
        "yaw": _uniform(rng, *TASK_BOARD_POSE["yaw"]),
    }


def _make_nic_rails(rng: random.Random, target_rail: int | None,
                    num_extra: int = 0) -> dict:
    """Generate NIC rail entries. target_rail gets a card; optionally add
    extra distractor NIC cards on other rails."""
    rails = {}
    occupied = set()
    if target_rail is not None:
        occupied.add(target_rail)

    extra_candidates = [i for i in range(NIC_RAIL["count"])
                        if i != target_rail]
    extras = rng.sample(extra_candidates,
                        min(num_extra, len(extra_candidates)))
    occupied.update(extras)

    card_counter = 0
    for i in range(NIC_RAIL["count"]):
        key = f"nic_rail_{i}"
        if i in occupied:
            rails[key] = {
                "entity_present": True,
                "entity_name": f"nic_card_{i}",
                "entity_pose": {
                    "translation": _uniform(rng, *NIC_RAIL["translation"]),
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": _uniform(rng, *NIC_RAIL["yaw"]),
                },
            }
            card_counter += 1
        else:
            rails[key] = {"entity_present": False}
    return rails


def _make_sc_rails(rng: random.Random, target_rail: int | None,
                   extra_rails: list[int] | None = None) -> dict:
    """Generate SC rail entries. target_rail gets a port; extra_rails
    adds distractor SC ports on the listed rail indices."""
    occupied = set()
    if target_rail is not None:
        occupied.add(target_rail)
    if extra_rails:
        occupied.update(extra_rails)

    rails = {}
    for i in range(SC_RAIL["count"]):
        key = f"sc_rail_{i}"
        if i in occupied:
            rails[key] = {
                "entity_present": True,
                "entity_name": f"sc_mount_{i}",
                "entity_pose": {
                    "translation": _uniform(rng, *SC_RAIL["translation"]),
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0,
                },
            }
        else:
            rails[key] = {"entity_present": False}
    return rails


def _make_mount_rails(rng: random.Random,
                      include_distractors: bool = False) -> dict:
    """Generate the 6 mount rail entries (lc/sfp/sc x 0/1).
    Optionally populate some with fixtures as distractors."""
    rail_names = [
        "lc_mount_rail_0", "sfp_mount_rail_0", "sc_mount_rail_0",
        "lc_mount_rail_1", "sfp_mount_rail_1", "sc_mount_rail_1",
    ]
    entity_names = [
        "lc_mount_0", "sfp_mount_0", "sc_mount_0",
        "lc_mount_1", "sfp_mount_1", "sc_mount_1",
    ]
    rails = {}
    for name, ename in zip(rail_names, entity_names):
        present = include_distractors and rng.random() < 0.4
        if present:
            rails[name] = {
                "entity_present": True,
                "entity_name": ename,
                "entity_pose": {
                    "translation": _uniform(rng, *MOUNT_RAIL["translation"]),
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": _uniform(rng, *MOUNT_RAIL["yaw"]),
                },
            }
        else:
            rails[name] = {"entity_present": False}
    return rails


def _make_cable(rng: random.Random, plug_type: str) -> tuple[str, dict]:
    """Return (cable_key, cable_config) for the given plug type."""
    nominal = NOMINAL_GRASP[plug_type]
    noise_p = GRASP_NOISE["position"]
    noise_o = GRASP_NOISE["orientation"]

    if plug_type == "sfp":
        cable_key = "cable_0"
        cable_type = "sfp_sc_cable"
    else:
        cable_key = "cable_1"
        cable_type = "sfp_sc_cable_reversed"

    cable = {
        "pose": {
            "gripper_offset": {
                "x": _perturb(rng, nominal["x"], noise_p),
                "y": _perturb(rng, nominal["y"], noise_p),
                "z": _perturb(rng, nominal["z"], noise_p),
            },
            "roll": _perturb(rng, nominal["roll"], noise_o),
            "pitch": _perturb(rng, nominal["pitch"], noise_o),
            "yaw": _perturb(rng, nominal["yaw"], noise_o),
        },
        "attach_cable_to_gripper": True,
        "cable_type": cable_type,
    }
    return cable_key, cable


# ---------------------------------------------------------------------------
# Trial generators per task type
# ---------------------------------------------------------------------------

def generate_sfp_trial(rng: random.Random, trial_id: str,
                       port_name: str = "sfp_port_0",
                       include_distractors: bool = True) -> dict:
    """SFP module insertion into an SFP port on a NIC card."""
    target_rail = rng.randint(0, NIC_RAIL["count"] - 1)
    num_extra_nic = rng.randint(0, 4) if include_distractors else 0

    nic_rails = _make_nic_rails(rng, target_rail, num_extra=num_extra_nic)

    num_sc_distractors = rng.randint(0, 2) if include_distractors else 0
    distractor_sc_rails = rng.sample(range(SC_RAIL["count"]),
                                     min(num_sc_distractors, SC_RAIL["count"]))
    sc_rails = _make_sc_rails(rng, target_rail=None,
                              extra_rails=distractor_sc_rails)

    mount_rails = _make_mount_rails(rng, include_distractors=include_distractors)
    cable_key, cable = _make_cable(rng, "sfp")

    scene = {"task_board": {"pose": _make_task_board_pose(rng)}}
    scene["task_board"].update(nic_rails)
    scene["task_board"].update(sc_rails)
    scene["task_board"].update(mount_rails)
    scene["cables"] = {cable_key: cable}

    task = {
        "task_1": {
            "cable_type": "sfp_sc",
            "cable_name": cable_key,
            "plug_type": "sfp",
            "plug_name": "sfp_tip",
            "port_type": "sfp",
            "port_name": port_name,
            "target_module_name": f"nic_card_mount_{target_rail}",
            "time_limit": 180,
        }
    }

    return {"scene": scene, "tasks": task}


def generate_sc_trial(rng: random.Random, trial_id: str,
                      include_distractors: bool = True) -> dict:
    """SC plug insertion into an SC port."""
    target_rail = rng.randint(0, SC_RAIL["count"] - 1)

    other_sc = [i for i in range(SC_RAIL["count"]) if i != target_rail]
    extra_sc = other_sc if (include_distractors and rng.random() < 0.5) else []
    sc_rails = _make_sc_rails(rng, target_rail, extra_rails=extra_sc)

    num_nic = rng.randint(0, 5) if include_distractors else 0
    nic_rails = _make_nic_rails(rng, target_rail=None,
                                num_extra=num_nic)

    mount_rails = _make_mount_rails(rng, include_distractors=include_distractors)
    cable_key, cable = _make_cable(rng, "sc")

    scene = {"task_board": {"pose": _make_task_board_pose(rng)}}
    scene["task_board"].update(nic_rails)
    scene["task_board"].update(sc_rails)
    scene["task_board"].update(mount_rails)
    scene["cables"] = {cable_key: cable}

    task = {
        "task_1": {
            "cable_type": "sfp_sc",
            "cable_name": cable_key,
            "plug_type": "sc",
            "plug_name": "sc_tip",
            "port_type": "sc",
            "port_name": "sc_port_base",
            "target_module_name": f"sc_port_{target_rail}",
            "time_limit": 180,
        }
    }

    return {"scene": scene, "tasks": task}


# ---------------------------------------------------------------------------
# Top-level config assembly
# ---------------------------------------------------------------------------

def _generate_single_trial(rng: random.Random, task_type: str,
                           trial_id: str,
                           include_distractors: bool = True) -> dict:
    """Generate one trial dict for the given task type."""
    if task_type == "sfp_port_0":
        return generate_sfp_trial(rng, trial_id, port_name="sfp_port_0",
                                  include_distractors=include_distractors)
    elif task_type == "sfp_port_1":
        return generate_sfp_trial(rng, trial_id, port_name="sfp_port_1",
                                  include_distractors=include_distractors)
    elif task_type == "sc_port":
        return generate_sc_trial(rng, trial_id,
                                 include_distractors=include_distractors)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")


def generate_config(rng: random.Random,
                    trial_task_types: list[str],
                    include_distractors: bool = True) -> dict:
    """Build a complete aic_engine config dict with one or more trials.

    Each trial gets its own independently randomized scene (board pose,
    rail placements, cable, grasp perturbation).
    """
    trials = {}
    for idx, task_type in enumerate(trial_task_types):
        trial_id = f"trial_{idx + 1}"
        trials[trial_id] = _generate_single_trial(
            rng, task_type, trial_id,
            include_distractors=include_distractors)

    return {
        "scoring": {"topics": SCORING_TOPICS},
        "task_board_limits": TASK_BOARD_LIMITS,
        "trials": trials,
        "robot": {"home_joint_positions": ROBOT_HOME},
    }


def generate_batch(num_trials: int,
                   trials_per_config: int = 1,
                   task_types: list[str] | None = None,
                   seed: int = 0,
                   include_distractors: bool = True) -> list[tuple[str, dict, list[str]]]:
    """Generate a batch of configs, balanced across task types.

    Args:
        num_trials: Total number of trials to generate.
        trials_per_config: How many trials to pack into each config file.
            Each trial still gets a fully independent randomized scene.
        task_types: Subset of TASK_TYPES to use, or None for all three.
        seed: Random seed.
        include_distractors: Whether to add distractor objects.

    Returns:
        List of (filename_stem, config_dict, task_type_list) tuples.
    """
    if task_types is None:
        task_types = TASK_TYPES

    rng = random.Random(seed)

    all_trial_types = [task_types[i % len(task_types)]
                       for i in range(num_trials)]

    configs = []
    for chunk_start in range(0, num_trials, trials_per_config):
        chunk = all_trial_types[chunk_start:chunk_start + trials_per_config]
        cfg = generate_config(rng, chunk,
                              include_distractors=include_distractors)
        config_idx = chunk_start // trials_per_config
        stem = f"config_{config_idx:05d}"
        configs.append((stem, cfg, chunk))

    return configs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate randomized AIC trial configs for training.")
    parser.add_argument("--num-trials", type=int, default=100,
                        help="Total number of trials to generate (default: 100)")
    parser.add_argument("--trials-per-config", type=int, default=1,
                        help="Trials per config file (default: 1). "
                             "Higher values save Gazebo restart time.")
    parser.add_argument("--output-dir", type=str, default="./configs",
                        help="Output directory (default: ./configs)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--task-type", type=str, default=None,
                        choices=TASK_TYPES,
                        help="Generate only this task type (default: balanced)")
    parser.add_argument("--no-distractors", action="store_true",
                        help="Omit distractor objects on the board")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print one config to stdout instead of writing files")
    args = parser.parse_args()

    task_types = [args.task_type] if args.task_type else None
    configs = generate_batch(
        num_trials=args.num_trials,
        trials_per_config=args.trials_per_config,
        task_types=task_types,
        seed=args.seed,
        include_distractors=not args.no_distractors,
    )

    if args.dry_run:
        stem, cfg, types = configs[0]
        print(f"# {stem}.yaml  (trials: {types})")
        print(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_trials = 0
    for stem, cfg, types in configs:
        path = out_dir / f"{stem}.yaml"
        with open(path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        total_trials += len(types)

    print(f"Generated {len(configs)} config files "
          f"({total_trials} total trials) in {out_dir.resolve()}")
    print(f"  Trials per config: {args.trials_per_config}")

    type_counts: dict[str, int] = {}
    for _, _, types in configs:
        for tt in types:
            type_counts[tt] = type_counts.get(tt, 0) + 1
    for tt, count in sorted(type_counts.items()):
        print(f"  {tt}: {count} trials")


if __name__ == "__main__":
    main()