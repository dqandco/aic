#!/usr/bin/env bash
#
# Run a single scenario end-to-end with a configurable policy.
# Useful for debugging whether issues lie in the scenarios or the policy.
#
# Usage (inside aic_eval container):
#   cd ~/ws_aic/src/aic
#
#   # Test with CheatCode (the known-good reference policy):
#   bash docker/my_policy/scripts/test_scenario.sh 0
#
#   # Test with ExpertCollector:
#   POLICY=my_policy.ExpertCollector bash docker/my_policy/scripts/test_scenario.sh 0
#
#   # Test scenario 42 with CheatCode, longer timeout:
#   EPISODE_TIMEOUT=90 bash docker/my_policy/scripts/test_scenario.sh 42
#
# Environment variables:
#   POLICY          – policy class (default: aic_example_policies.ros.CheatCode)
#   SCENARIO_DIR    – scenarios directory (default: scenarios)
#   SETTLE_TIME     – seconds to wait for sim startup (default: 25)
#   EPISODE_TIMEOUT – max seconds per episode (default: 60)

set -euo pipefail

IDX="${1:?Usage: $0 <scenario_index>}"
: "${POLICY:=aic_example_policies.ros.CheatCode}"
: "${SCENARIO_DIR:=scenarios}"
: "${SETTLE_TIME:=25}"
: "${EPISODE_TIMEOUT:=60}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

IDX_FMT=$(printf "%03d" "$IDX")
ARGS_FILE="${SCENARIO_DIR}/scenario_${IDX_FMT}.args"
MANIFEST="${SCENARIO_DIR}/manifest.json"

if [ ! -f "$ARGS_FILE" ]; then
    echo "ERROR: $ARGS_FILE not found"
    exit 1
fi
if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: $MANIFEST not found. Run generate_scenarios.py first."
    exit 1
fi

SCENARIO_ARGS=$(cat "$ARGS_FILE")

# Extract task info from manifest
TASK_JSON=$(python3 -c "
import json, sys
m = json.load(open('$MANIFEST'))
t = m['scenarios'][$IDX]['task']
print(json.dumps(t))
")
CABLE_NAME=$(echo "$TASK_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['cable_name'])")
CABLE_TYPE=$(echo "$TASK_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['cable_type'])")
PLUG_TYPE=$(echo "$TASK_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['plug_type'])")
PLUG_NAME=$(echo "$TASK_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['plug_name'])")
PORT_TYPE=$(echo "$TASK_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['port_type'])")
PORT_NAME=$(echo "$TASK_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['port_name'])")
TARGET_MODULE=$(echo "$TASK_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['target_module_name'])")

echo "============================================"
echo " Test Scenario ${IDX_FMT}"
echo "============================================"
echo " Policy:  ${POLICY}"
echo " Task:    ${PLUG_NAME} -> ${PORT_NAME} @ ${TARGET_MODULE}"
echo " Cable:   ${CABLE_TYPE} (${CABLE_NAME})"
echo " Timeout: ${EPISODE_TIMEOUT}s"
echo "============================================"
echo ""

# -----------------------------------------------------------------------
# Cleanup trap
# -----------------------------------------------------------------------
SIM_PID=""
MODEL_PID=""

cleanup() {
    echo ""
    echo "Cleaning up..."
    [ -n "$MODEL_PID" ] && kill "$MODEL_PID" 2>/dev/null || true
    [ -n "$SIM_PID" ]   && kill "$SIM_PID"   2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT

# -----------------------------------------------------------------------
# 1. Launch simulation
# -----------------------------------------------------------------------
echo "[1/5] Starting sim..."
/entrypoint.sh \
    $SCENARIO_ARGS \
    ground_truth:=true \
    start_aic_engine:=false \
    gazebo_gui:=true \
    launch_rviz:=false \
    &
SIM_PID=$!

echo "  Waiting ${SETTLE_TIME}s for sim to settle..."
sleep "$SETTLE_TIME"

if ! kill -0 "$SIM_PID" 2>/dev/null; then
    echo "  FAIL: Sim crashed."
    exit 1
fi

# -----------------------------------------------------------------------
# 2. Tare F/T sensor
# -----------------------------------------------------------------------
echo "[2/5] Taring F/T sensor..."
pixi run ros2 service call \
    /aic_controller/tare_force_torque_sensor std_srvs/srv/Trigger \
    > /dev/null 2>&1 || true
sleep 1

# -----------------------------------------------------------------------
# 3. Start policy
# -----------------------------------------------------------------------
echo "[3/5] Starting policy: ${POLICY}"
pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p "policy:=${POLICY}" &
MODEL_PID=$!
sleep 5

if ! kill -0 "$MODEL_PID" 2>/dev/null; then
    echo "  FAIL: Policy crashed."
    exit 1
fi

# -----------------------------------------------------------------------
# 4. Activate aic_model lifecycle
# -----------------------------------------------------------------------
echo "[4/5] Activating aic_model lifecycle..."
pixi run ros2 lifecycle set /aic_model configure
pixi run ros2 lifecycle set /aic_model activate
sleep 1

# -----------------------------------------------------------------------
# 5. Send insert_cable goal
# -----------------------------------------------------------------------
echo "[5/5] Sending insert_cable goal..."
echo ""

GOAL_YAML="{task: {id: 'scenario_${IDX_FMT}', cable_type: '${CABLE_TYPE}', cable_name: '${CABLE_NAME}', plug_type: '${PLUG_TYPE}', plug_name: '${PLUG_NAME}', port_type: '${PORT_TYPE}', port_name: '${PORT_NAME}', target_module_name: '${TARGET_MODULE}', time_limit: ${EPISODE_TIMEOUT}}}"

timeout "$EPISODE_TIMEOUT" pixi run ros2 action send_goal \
    --feedback \
    /insert_cable \
    aic_task_interfaces/action/InsertCable \
    "$GOAL_YAML" \
    && echo "" && echo "=== RESULT: Goal succeeded ===" \
    || echo "" && echo "=== RESULT: Goal timed out or failed ==="

echo ""
echo "Done."
