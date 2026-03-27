#!/usr/bin/env bash
#
# Batch-collect expert demonstrations across generated scenarios.
#
# Orchestrates three processes per episode:
#   1. Simulation (Gazebo + aic_controller, with ground_truth:=true)
#   2. aic_model running ExpertCollector (drives the robot using GT TF)
#   3. lerobot-record (captures observations + actions into a LeRobot dataset)
#
# The recorder stays alive across all episodes. Between scenarios, it
# receives a simulated Right-Arrow keypress (via xdotool) to advance
# to the next episode.
#
# Usage:
#   cd ~/ws_aic/src/aic
#
#   # 1. Generate scenarios (only needed once)
#   python3 docker/my_policy/scripts/generate_scenarios.py
#
#   # 2. Collect demonstrations
#   HF_USER=youruser bash docker/my_policy/scripts/batch_collect.sh
#
# Environment variables:
#   HF_USER          – HuggingFace username (required)
#   SCENARIO_DIR     – scenarios directory (default: scenarios)
#   DATASET_REPO_ID  – dataset name (default: ${HF_USER}/aic_expert_demos)
#   START_IDX        – first scenario index to collect (default: 0)
#   END_IDX          – last scenario index (default: all)
#   SETTLE_TIME      – seconds to wait for sim startup (default: 25)
#   EPISODE_TIMEOUT  – max seconds per episode (default: 60)
#   SIM_LAUNCH_CMD   – command to launch the sim (default: distrobox enter -r aic_eval --)
#                      Set to "pixi run ros2 launch aic_bringup aic_gz_bringup.launch.py"
#                      if running from a full source build with aic_bringup available.

set -euo pipefail

: "${HF_USER:?Set HF_USER to your HuggingFace username}"
: "${SCENARIO_DIR:=scenarios}"
: "${DATASET_REPO_ID:=${HF_USER}/aic_expert_demos}"
: "${SETTLE_TIME:=25}"
: "${EPISODE_TIMEOUT:=60}"
: "${SIM_LAUNCH_CMD:=distrobox enter -r aic_eval -- /entrypoint.sh}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

MANIFEST="${SCENARIO_DIR}/manifest.json"
if [ ! -f "$MANIFEST" ]; then
    echo "Generating scenarios..."
    python3 docker/my_policy/scripts/generate_scenarios.py --output-dir "$SCENARIO_DIR"
fi

TOTAL=$(python3 -c "import json; print(json.load(open('$MANIFEST'))['count'])")
: "${START_IDX:=0}"
: "${END_IDX:=$((TOTAL - 1))}"

LOG_DIR="${SCENARIO_DIR}/logs"
mkdir -p "$LOG_DIR"

echo "============================================"
echo " Batch Expert Demo Collection"
echo "============================================"
echo " Scenarios:   ${START_IDX}..${END_IDX} of ${TOTAL}"
echo " Dataset:     ${DATASET_REPO_ID}"
echo " Settle time: ${SETTLE_TIME}s"
echo " Episode max: ${EPISODE_TIMEOUT}s"
echo " Sim launch:  ${SIM_LAUNCH_CMD}"
echo "============================================"
echo ""

# -----------------------------------------------------------------------
# Cleanup trap
# -----------------------------------------------------------------------
RECORDER_PID=""
SIM_PID=""
MODEL_PID=""

cleanup() {
    echo ""
    echo "Cleaning up..."
    [ -n "$MODEL_PID" ] && kill "$MODEL_PID" 2>/dev/null || true
    [ -n "$SIM_PID" ] && kill "$SIM_PID" 2>/dev/null || true
    [ -n "$RECORDER_PID" ] && kill "$RECORDER_PID" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT

# -----------------------------------------------------------------------
# Start the LeRobot recorder (stays alive across all episodes)
# -----------------------------------------------------------------------
echo "Starting LeRobot recorder..."
echo "(Press Ctrl-C at any time to stop collection)"
echo ""

pixi run lerobot-record \
    --robot.type=aic_controller --robot.id=aic \
    --teleop.type=aic_keyboard_ee --teleop.id=aic \
    --robot.teleop_target_mode=cartesian \
    --robot.teleop_frame_id=base_link \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.single_task="insert cable into port" \
    --dataset.push_to_hub=false \
    --dataset.private=true \
    --play_sounds=false \
    --display_data=false \
    > "${LOG_DIR}/recorder.log" 2>&1 &
RECORDER_PID=$!
sleep 8  # wait for recorder to initialise and connect

if ! kill -0 "$RECORDER_PID" 2>/dev/null; then
    echo "[FAIL] Recorder crashed. See ${LOG_DIR}/recorder.log"
    exit 1
fi
echo "Recorder started (PID ${RECORDER_PID})"

# -----------------------------------------------------------------------
# Helper: send Right-Arrow key to the recorder's terminal
# -----------------------------------------------------------------------
advance_episode() {
    # Method 1: xdotool (if available and recorder has a window)
    if command -v xdotool &>/dev/null; then
        # Send Right arrow key to the recorder process
        xdotool key --clearmodifiers Right 2>/dev/null && return 0
    fi
    # Method 2: send escape sequence to recorder's stdin via /proc
    if [ -d "/proc/${RECORDER_PID}/fd" ]; then
        # Right arrow: ESC [ C
        printf '\033[C' > "/proc/${RECORDER_PID}/fd/0" 2>/dev/null && return 0
    fi
    echo "  [WARN] Could not send next-episode signal to recorder."
    echo "         Press Right Arrow manually, or install xdotool."
    sleep 3
}

# -----------------------------------------------------------------------
# Collect one episode
# -----------------------------------------------------------------------
collect_episode() {
    local IDX=$1
    local IDX_FMT
    IDX_FMT=$(printf "%03d" "$IDX")
    local ARGS_FILE="${SCENARIO_DIR}/scenario_${IDX_FMT}.args"

    if [ ! -f "$ARGS_FILE" ]; then
        echo "  [SKIP] ${ARGS_FILE} not found"
        return 1
    fi

    local SCENARIO_ARGS
    SCENARIO_ARGS=$(cat "$ARGS_FILE")

    # Extract task info from manifest
    local TASK_JSON
    TASK_JSON=$(python3 -c "
import json
m = json.load(open('$MANIFEST'))
t = m['scenarios'][$IDX]['task']
print(json.dumps(t))
")
    local CABLE_NAME PLUG_NAME PORT_NAME TARGET_MODULE CABLE_TYPE_SHORT
    CABLE_NAME=$(echo "$TASK_JSON" | python3 -c "import json,sys; t=json.load(sys.stdin); print(t['cable_name'])")
    PLUG_NAME=$(echo "$TASK_JSON" | python3 -c "import json,sys; t=json.load(sys.stdin); print(t['plug_name'])")
    PORT_NAME=$(echo "$TASK_JSON" | python3 -c "import json,sys; t=json.load(sys.stdin); print(t['port_name'])")
    TARGET_MODULE=$(echo "$TASK_JSON" | python3 -c "import json,sys; t=json.load(sys.stdin); print(t['target_module_name'])")
    CABLE_TYPE_SHORT=$(echo "$TASK_JSON" | python3 -c "import json,sys; t=json.load(sys.stdin); print(t['cable_type'])")
    local PLUG_TYPE PORT_TYPE
    PLUG_TYPE=$(echo "$TASK_JSON" | python3 -c "import json,sys; t=json.load(sys.stdin); print(t['plug_type'])")
    PORT_TYPE=$(echo "$TASK_JSON" | python3 -c "import json,sys; t=json.load(sys.stdin); print(t['port_type'])")

    echo "=== Scenario ${IDX_FMT} ==="
    echo "  Task: ${PLUG_NAME} -> ${PORT_NAME} @ ${TARGET_MODULE}"

    # --- 1. Launch simulation (via eval container) ---
    echo "  Starting sim..."
    $SIM_LAUNCH_CMD \
        $SCENARIO_ARGS \
        ground_truth:=true \
        start_aic_engine:=false \
        gazebo_gui:=false \
        launch_rviz:=false \
        > "${LOG_DIR}/sim_${IDX_FMT}.log" 2>&1 &
    SIM_PID=$!

    sleep "$SETTLE_TIME"

    if ! kill -0 "$SIM_PID" 2>/dev/null; then
        echo "  [FAIL] Sim crashed. See ${LOG_DIR}/sim_${IDX_FMT}.log"
        SIM_PID=""
        return 1
    fi

    # --- 2. Tare F/T sensor ---
    pixi run ros2 service call \
        /aic_controller/tare_force_torque_sensor std_srvs/srv/Trigger \
        > /dev/null 2>&1 || true
    sleep 1

    # --- 3. Start ExpertCollector ---
    echo "  Starting ExpertCollector..."
    pixi run ros2 run aic_model aic_model --ros-args \
        -p use_sim_time:=true \
        -p policy:=my_policy.ExpertCollector \
        > "${LOG_DIR}/expert_${IDX_FMT}.log" 2>&1 &
    MODEL_PID=$!
    sleep 5

    if ! kill -0 "$MODEL_PID" 2>/dev/null; then
        echo "  [FAIL] ExpertCollector crashed. See ${LOG_DIR}/expert_${IDX_FMT}.log"
        kill "$SIM_PID" 2>/dev/null || true; wait "$SIM_PID" 2>/dev/null || true
        SIM_PID=""; MODEL_PID=""
        return 1
    fi

    # --- 4. Send insert_cable action goal ---
    echo "  Sending insert_cable goal..."
    timeout "$EPISODE_TIMEOUT" pixi run ros2 action send_goal \
        /insert_cable aic_task_interfaces/action/InsertCable \
        "{task: {id: 'scenario_${IDX_FMT}', cable_type: '${CABLE_TYPE_SHORT}', cable_name: '${CABLE_NAME}', plug_type: '${PLUG_TYPE}', plug_name: '${PLUG_NAME}', port_type: '${PORT_TYPE}', port_name: '${PORT_NAME}', target_module_name: '${TARGET_MODULE}', time_limit: ${EPISODE_TIMEOUT}}}" \
        > "${LOG_DIR}/goal_${IDX_FMT}.log" 2>&1 \
        && echo "  [OK] Episode done" \
        || echo "  [WARN] Episode timed out or failed"

    # --- 5. Signal recorder: advance to next episode ---
    echo "  Advancing recorder to next episode..."
    advance_episode

    # --- 6. Tear down sim + expert for this scenario ---
    kill "$MODEL_PID" 2>/dev/null || true; wait "$MODEL_PID" 2>/dev/null || true
    MODEL_PID=""
    kill "$SIM_PID" 2>/dev/null || true; wait "$SIM_PID" 2>/dev/null || true
    SIM_PID=""
    sleep 3  # let ports release

    return 0
}

# -----------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------
SUCCEEDED=0
FAILED=0

for IDX in $(seq "$START_IDX" "$END_IDX"); do
    if collect_episode "$IDX"; then
        SUCCEEDED=$((SUCCEEDED + 1))
    else
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

# Stop the recorder (send ESC)
echo "Stopping recorder..."
if command -v xdotool &>/dev/null; then
    xdotool key Escape 2>/dev/null || true
elif [ -d "/proc/${RECORDER_PID}/fd" ]; then
    printf '\033' > "/proc/${RECORDER_PID}/fd/0" 2>/dev/null || true
fi
sleep 5
kill "$RECORDER_PID" 2>/dev/null || true
wait "$RECORDER_PID" 2>/dev/null || true
RECORDER_PID=""

echo ""
echo "============================================"
echo " Collection complete"
echo "  Succeeded: ${SUCCEEDED}"
echo "  Failed:    ${FAILED}"
echo "  Dataset:   ${DATASET_REPO_ID}"
echo "  Logs:      ${LOG_DIR}/"
echo "============================================"
echo ""
echo "Next steps:"
echo "  # Fine-tune ACT on the collected data"
echo "  HF_USER=${HF_USER} bash docker/my_policy/scripts/finetune_act.sh"
echo ""
echo "  # Run the fine-tuned policy"
echo "  ACT_MODEL_REPO=${HF_USER}/aic_act_finetuned \\"
echo "    pixi run ros2 run aic_model aic_model --ros-args \\"
echo "    -p policy:=my_policy.HybridACTInsertion"
