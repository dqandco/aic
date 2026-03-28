# CheatCode Test Setups

These configs are intended for quick `CheatCode` validation runs with
`ground_truth:=true`.

## Run The Policy

Start the policy in a separate terminal:

```bash
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.CheatCode
```

## Run Setup 1

One NIC card on `nic_rail_0`, one SC port on `sc_rail_1`, target is
`sfp_port_0` on `nic_card_mount_0`.

```bash
/entrypoint.sh ground_truth:=true start_aic_engine:=true shutdown_on_aic_engine_exit:=true aic_engine_config_file:=/home/cding/Downloads/aic/aic_engine/config/cheat_code_setup_1.yaml
```

## Run Setup 2

Three NIC cards on `nic_rail_0`, `nic_rail_2`, and `nic_rail_4`, two SC ports
present, target is `sfp_port_1` on `nic_card_mount_4`.

```bash
/entrypoint.sh ground_truth:=true start_aic_engine:=true shutdown_on_aic_engine_exit:=true aic_engine_config_file:=/home/cding/Downloads/aic/aic_engine/config/cheat_code_setup_2.yaml
```

## Run Setup 3

No NIC cards, two SC ports on `sc_rail_0` and `sc_rail_1`, target is
`sc_port_base` on `sc_port_0`.

```bash
/entrypoint.sh ground_truth:=true start_aic_engine:=true shutdown_on_aic_engine_exit:=true aic_engine_config_file:=/home/cding/Downloads/aic/aic_engine/config/cheat_code_setup_3.yaml
```
