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
