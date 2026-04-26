# Postmortem: Gazebo CEM training infrastructure

Short writeup of the infrastructure bugs we fixed while getting
`docker/my_policy/scripts/train_gazebo_residual_act.py` to actually run a
clean Cross-Entropy-Method loop end-to-end against the `aic_eval` Gazebo
stack, plus a snapshot of the current working setup so future-you can
reproduce a healthy run.

For the user-facing recovery steps and detailed symptom-by-symptom guides,
see [`docs/troubleshooting.md`](../troubleshooting.md).

## Issues we hit (and how we fixed them)

### 1. Multi-arm Gazebo / "Another world of the same name is running"

* `Ctrl+C` on `entrypoint.sh` left a zombie sim alive. `gz_server` is
  loaded as a composable node inside `component_container`, so a naive
  `pkill -f "gz sim"` missed it; orphan podman `conmon` processes also
  outlived their container and kept ROS nodes alive.
* The next `entrypoint.sh` then spawned a *second* sim alongside the
  zombie. We saw 2-3 arms in Gazebo, fighting `/clock` publishers, and
  `aic_engine` would die with `Failed to find a valid clock`.
* **Fix:** `docker/my_policy/scripts/clean_sim.sh` (idempotent; supports
  `--check`, `--hard`, `--container`) kills host-side ROS nodes,
  in-container processes, and orphan conmons in the right order. The
  runner's `_check_existing_sim_alive` preflight aborts in ~5 s with a
  pointer back to `clean_sim.sh` instead of letting the engine time out.

### 2. RMW mismatch: `aic_engine` couldn't see the Zenoh-published `/clock`

* `entrypoint.sh` exports `RMW_IMPLEMENTATION=rmw_zenoh_cpp` only inside
  *its own* process tree.
* A fresh `distrobox enter aic_eval -- bash -lc 'ros2 run aic_engine ...'`
  shell spawned by the runner therefore fell back to FastDDS, never saw
  any Zenoh-published topic, and died after 10 s with `Waiting for
  clock` -> `Failed to find a valid clock` -> exit 1. The user-visible
  message was the very confusing `aic_engine completed without producing
  scoring.yaml.`
* **Fix:** the runner explicitly prepends
  `export RMW_IMPLEMENTATION=rmw_zenoh_cpp && export ZENOH_ROUTER_CONFIG_URI=/aic_zenoh_config.json5`
  to the engine command. A second preflight (`_check_clock_in_container`)
  runs `ros2 topic info /clock` *inside the container* with the same RMW
  exports and refuses to launch the engine if `Publisher count` is 0.

### 3. Race: `aic_engine` timed out waiting for the participant model

* `aic_engine` waits 5 s for `/aic_model/get_state` to answer.
* `ResidualACT.__init__` synchronously imports torch and loads a HF
  checkpoint, which takes ~25–35 s on a cold cache. The lifecycle
  service is *advertised* almost immediately (as soon as
  `create_service` runs), but the rclpy executor isn't spinning yet, so
  `GetState` requests time out without ever being dispatched.
* Engine bailed with `GetState service call timed out` -> `Participant
  model is not ready` -> exit 1.
* **Fix:** `wait_for_model_ready()` polls `/aic_model/get_state` from the
  host until it actually responds (default budget 90 s,
  `EpisodeSpec.model_ready_timeout`). The engine only launches once the
  model is responsive.

### 4. `sudo` password prompts in non-interactive subprocesses

* `distrobox enter -r` shells out to `sudo docker exec`. With sudo's
  default `tty_tickets`, cached credentials are scoped per-TTY, so a
  `sudo -v` in your tmux pane does *not* help the runner's child
  processes (they have no TTY).
* In tmux, training would either hang on the prompt or fail cryptically
  when the subprocess couldn't re-auth.
* **Fix (operator):** add `cding ALL=(ALL) NOPASSWD: /usr/bin/docker` to
  `/etc/sudoers.d/cding-docker`, so the docker CLI never prompts.
* **Fix (runner):** `_detect_distrobox_root` tries `distrobox enter`
  *without* `-r` first (works whenever the user is in the `docker`
  group, which is the recommended Docker post-install setup) and only
  falls back to `-r` if that fails. All `subprocess.run`/`Popen` calls
  use `stdin=DEVNULL` so a stuck `sudo` prompt fails immediately
  instead of hanging until the timeout.

### 5. CEM unfairness: each candidate saw a different config

* The first version of the CEM loop drew configs *per candidate*, so
  candidate A vs B comparison was confounded by task difficulty: an
  unlucky candidate could be ranked out by a hard task even though its
  parameters were better.
* **Fix:** `sample_iteration_configs` is called once per iteration and
  every candidate is scored on the same configs. The draw still rotates
  across iterations so the population sees a broad task distribution
  over the run. The chosen configs are recorded in
  `iter_NNN/summary.json` for audit.

## Current setup that produces successful CEM runs

### One-time host setup

* `export DBX_CONTAINER_MANAGER=docker` (also worth putting in `~/.bashrc`).
* User in the `docker` group (`groups | grep docker`).
* Passwordless sudo for docker only:
  ```bash
  echo "$USER ALL=(ALL) NOPASSWD: /usr/bin/docker" \
    | sudo tee /etc/sudoers.d/$USER-docker
  sudo chmod 440 /etc/sudoers.d/$USER-docker
  ```

### Per-session bring-up

In an `aic_eval` distrobox shell (kept alive in its own tmux pane):

```bash
distrobox enter aic_eval
/entrypoint.sh ground_truth:=false start_aic_engine:=false
```

This keeps Gazebo, RViz, the Zenoh router, the controller manager, and
the ROS graph alive but lets the runner own `aic_engine` per episode.

In a *separate* host tmux pane:

```bash
docker/my_policy/scripts/clean_sim.sh --check    # should print host=0, container=0, conmon=0 once entrypoint is up
```

### Training command

```bash
pixi run python docker/my_policy/scripts/train_gazebo_residual_act.py \
  --output-dir outputs/gazebo_score_rl \
  --init-checkpoint outputs/train_smoketest/best_adapter.pt \
  --init-sigma 0.03 \
  2>&1 | tee outputs/gazebo_score_rl.log
```

Re-running on the same `--output-dir` automatically warm-starts from
`<output-dir>/best_adapter.pt` and carries over its `best_reward` as a
floor. Use `--no-auto-resume` to deliberately start from zero.

### What the loop does each iteration

1. Sample `--episodes-per-candidate` configs once; reuse them across
   every candidate in this iteration.
2. For each of `--population-size` candidates:
   * Sample `θ ~ N(mean, σ²)`, save adapter checkpoint to
     `iter_NNN/candidate_MM/adapter.pt`.
   * For each rollout:
     * Preflight: distrobox can enter, `aic_controller` service is
       present on the host ROS graph, `/clock` has a publisher inside
       the container.
     * Launch `aic_model` on the host pixi env with `ACT_*` env vars
       pointing at this candidate's checkpoint.
     * Poll `/aic_model/get_state` until it actually answers (≤ 90 s).
     * Launch `aic_engine` inside the container with explicit Zenoh
       RMW + router config + `AIC_RESULTS_DIR`.
     * Wait ≤ `--episode-timeout` (default 240 s), parse
       `scoring.yaml`, convert to reward via `RewardWeights`.
   * Average the per-rollout rewards. If better than the running best,
     overwrite `<output-dir>/best_adapter.pt`.
3. Sort candidates by avg reward, take the top `--elite-count`, refit
   the Gaussian (`mean = elite_vectors.mean(0)`,
   `σ = elite_vectors.std(0).clamp(min=--min-sigma)`).
4. Write `iter_NNN/summary.json` with all candidate vectors, episode
   records, the iteration's config draw, and the new mean/σ.

### Retry policy for transient infra failures

* `is_infrastructure_failure` flags an episode as retryable when:
  * `aic_engine` exits non-zero, *or*
  * the produced `scoring.yaml` is all-trial validation-failed (total
    score 0 with `Validation failed` in every tier-1 message), *or*
  * the runner's own `failure_reason` matches a known marker
    (`/clock has no publisher`, `aic_model did not answer
    /aic_model/get_state`, `No running sim detected`, etc.), *or*
  * the engine log tail contains a known marker
    (`GetState service call timed out`, `Engine Stopped with Errors`,
    `Failed to spawn cable`, etc.).
* Retried up to `--episode-retries=2` times with `--retry-backoff-s=5.0`
  between attempts. Each attempt's artefacts go in
  `episode_NN/attempt_KK/` so retries are auditable.

## Known sharp edges still in the system

* The CEM search space is ~7,400 params (the residual MLP); pop=4 /
  elite=2 is wildly under-sampled for that. Expect slow / noisy
  convergence until we shrink the adapter or switch optimizer.
* `aic_model` reload on every candidate costs ~30 s of pure PyTorch
  import + checkpoint load. Each iteration is therefore dominated by
  model spin-up, not by sim time.
* `clean_sim.sh --hard` calls `docker restart aic_eval`, which kicks any
  open `distrobox enter` shell. Always re-launch the entrypoint
  afterwards.
