# Generated Config Distribution Summary

Generated with `generate_configs.py --num-trials 5000 --trials-per-config 3 --seed 42`.

## Overview

| Metric | Value |
|--------|-------|
| Config files | 1667 |
| Total trials | 5000 |
| SFP trials | 3334 (66.7%) |
| SC trials | 1666 (33.3%) |
| Trials per config | 3 (one of each task type) |

## Task Type Balance

| Task Type | Count | % |
|-----------|-------|---|
| sfp_port_0 | 1667 | 33.3% |
| sfp_port_1 | 1667 | 33.3% |
| sc_port | 1666 | 33.3% |

## Board Pose (uniform per trial)

| Axis | Range | Mean | Std |
|------|-------|------|-----|
| x | [0.12, 0.22] m | 0.170 | 0.029 |
| y | [-0.25, 0.10] m | -0.076 | 0.101 |
| yaw | [2.70, 3.30] rad | 3.000 | 0.171 |
| z | 1.14 (fixed) | — | — |
| roll, pitch | 0.0 (fixed) | — | — |

## Target Rail Distribution

### NIC card targets (SFP trials, 5 rails)

| Target | Count | % |
|--------|-------|---|
| nic_card_mount_0 | 719 | 21.6% |
| nic_card_mount_1 | 635 | 19.0% |
| nic_card_mount_2 | 665 | 19.9% |
| nic_card_mount_3 | 666 | 20.0% |
| nic_card_mount_4 | 649 | 19.5% |

### SC port targets (SC trials, 2 rails)

| Target | Count | % |
|--------|-------|---|
| sc_port_0 | 857 | 51.4% |
| sc_port_1 | 809 | 48.6% |

## NIC Rail Parameters

| Param | Range | Mean | Std |
|-------|-------|------|-----|
| translation | [-0.0215, 0.0234] m | 0.001 | 0.013 |
| yaw | [-0.1744, 0.1745] rad | -0.001 | 0.101 |

## SC Rail Parameters

| Param | Range | Mean | Std |
|-------|-------|------|-----|
| translation | [-0.06, 0.055] m | -0.002 | 0.033 |

## Distractors

### Extra NIC cards in SFP trials (0–4)

| Extra NICs | Count | % |
|------------|-------|---|
| 0 | 672 | 20.2% |
| 1 | 705 | 21.1% |
| 2 | 672 | 20.2% |
| 3 | 656 | 19.7% |
| 4 | 629 | 18.9% |

### SC port distractors in SFP trials (0–2)

| SC ports | Count | % |
|----------|-------|---|
| 0 | 1141 | 34.2% |
| 1 | 1117 | 33.5% |
| 2 | 1076 | 32.3% |

### NIC card distractors in SC trials (0–5)

| NIC cards | Count | % |
|-----------|-------|---|
| 0 | 286 | 17.2% |
| 1 | 282 | 16.9% |
| 2 | 280 | 16.8% |
| 3 | 285 | 17.1% |
| 4 | 255 | 15.3% |
| 5 | 278 | 16.7% |

### Extra SC port in SC trials (0–1)

| Extra SC | Count | % |
|----------|-------|---|
| 0 | 864 | 51.9% |
| 1 | 802 | 48.1% |

### Mount rail fixtures (6 slots, ~40% each)

| Mounts present | Count | % |
|----------------|-------|---|
| 0 | 228 | 4.6% |
| 1 | 977 | 19.5% |
| 2 | 1518 | 30.4% |
| 3 | 1384 | 27.7% |
| 4 | 691 | 13.8% |
| 5 | 186 | 3.7% |
| 6 | 16 | 0.3% |

Mean mounts per trial: 2.39

## Grasp Perturbation

Noise applied to nominal gripper offset (~2 mm position, ~0.04 rad orientation).

| Param | Nominal | Mean | Std |
|-------|---------|------|-----|
| SFP gripper_offset.z | 0.04245 | 0.0424 | 0.0011 |
| SC gripper_offset.z | 0.04045 | 0.0405 | 0.0011 |
| roll | 0.4432 | 0.4431 | 0.023 |
| pitch | -0.4838 | -0.4839 | 0.023 |
| yaw | 1.3303 | 1.3309 | 0.023 |
