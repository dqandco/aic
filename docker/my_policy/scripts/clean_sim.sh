#!/usr/bin/env bash
# Clean up every leftover process the AIC Gazebo eval stack tends to leave
# behind: stale ros2 launch nodes on the host, the in-container ROS graph
# (gz_server, robot_state_publisher, controller_manager, ros_gz_bridge,
# rmw_zenohd, ...), and orphan podman ``conmon`` processes whose container
# was deleted but whose children kept running.
#
# Usage:
#   docker/my_policy/scripts/clean_sim.sh            # SIGTERM, then SIGKILL
#   docker/my_policy/scripts/clean_sim.sh --check    # list only, no kill
#   docker/my_policy/scripts/clean_sim.sh --hard     # also `docker restart`
#   docker/my_policy/scripts/clean_sim.sh --container my_eval
#
# Safe to run more than once. Exits non-zero only when *--check* finds
# leftovers (so it can be wired into preflight checks / CI).

set -u
set -o pipefail

CONTAINER="${AIC_EVAL_CONTAINER:-aic_eval}"
CHECK_ONLY=0
HARD=0
QUIET=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check)        CHECK_ONLY=1; shift ;;
    --hard)         HARD=1; shift ;;
    --quiet|-q)     QUIET=1; shift ;;
    --container|-c) CONTAINER="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,16p' "$0"
      exit 0
      ;;
    *)
      echo "clean_sim.sh: unknown argument '$1'" >&2
      exit 2
      ;;
  esac
done

log() { [[ $QUIET -eq 1 ]] || echo "[clean_sim] $*"; }

# Patterns that should never survive a clean shutdown. ``component_container``
# is the most-missed one: it hosts ``gz_server`` as a composable node, which
# is why grepping for "gz sim" alone leaves the actual world running.
HOST_PATTERNS=(
  "ros2 launch aic_bringup"
  "/opt/ros/.*/lib/rclcpp_components/component_container.*ros_gz_container"
  "/opt/ros/.*/lib/robot_state_publisher/robot_state_publisher"
  "/ws_aic/install/lib/aic_adapter/aic_adapter"
  "/ws_aic/install/lib/aic_engine/aic_engine"
  "/ws_aic/install/lib/controller_manager/spawner"
  "ros2_control_node"
  "controller_manager"
  "ros_gz_bridge"
  "gz_server"
  "gzserver"
  "gz sim"
  "rmw_zenohd"
  "rviz2"
)

CONTAINER_PATTERNS=(
  "ros2 launch aic_bringup"
  "rclcpp_components/component_container"
  "robot_state_publisher"
  "aic_adapter"
  "aic_engine"
  "controller_manager"
  "ros_gz_bridge"
  "gz_server"
  "gzserver"
  "gz sim"
  "rmw_zenohd"
  "rviz2"
)

list_host_pids() {
  local pattern_re
  pattern_re=$(IFS='|'; echo "${HOST_PATTERNS[*]}")
  ps -eo pid,user,etime,pcpu,args --no-headers \
    | awk -v re="$pattern_re" 'BEGIN{IGNORECASE=0} $0 ~ re {print}' \
    | grep -v -F 'clean_sim.sh' \
    | grep -v -F ' awk '
}

list_orphan_conmons() {
  ps -eo pid,user,etime,args --no-headers \
    | awk -v c="$CONTAINER" '$0 ~ ("conmon.*-n " c "( |$)") {print}' \
    | grep -v -F ' awk '
}

container_alive() {
  docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null \
    | grep -qx true
}

list_container_pids() {
  container_alive || return 0
  local pattern_re
  pattern_re=$(IFS='|'; echo "${CONTAINER_PATTERNS[*]}")
  docker exec "$CONTAINER" bash -lc \
    "ps -eo pid,user,etime,pcpu,args --no-headers \
       | awk -v re='$pattern_re' '\$0 ~ re && \$0 !~ /awk/' || true" 2>/dev/null
}

if [[ $CHECK_ONLY -eq 1 ]]; then
  found=0

  log "Host processes matching the eval stack:"
  host_hits=$(list_host_pids || true)
  if [[ -n "$host_hits" ]]; then
    echo "$host_hits"
    found=1
  else
    log "  (none)"
  fi

  log "In-container ($CONTAINER) processes:"
  if container_alive; then
    container_hits=$(list_container_pids || true)
    if [[ -n "$container_hits" ]]; then
      echo "$container_hits"
      found=1
    else
      log "  (none)"
    fi
  else
    log "  (container not running)"
  fi

  log "Orphan podman conmon processes for '$CONTAINER':"
  conmon_hits=$(list_orphan_conmons || true)
  if [[ -n "$conmon_hits" ]]; then
    echo "$conmon_hits"
    found=1
  else
    log "  (none)"
  fi

  exit $found
fi

# 1) In-container TERM -> KILL. Do this FIRST: if we kill the host
#    distrobox shell while the in-container nodes are still alive, gz keeps
#    running with no controlling terminal.
if container_alive; then
  log "Stopping ROS nodes inside container '$CONTAINER'..."
  for sig in TERM KILL; do
    for pat in "${CONTAINER_PATTERNS[@]}"; do
      docker exec "$CONTAINER" bash -lc "pkill -$sig -f '$pat' 2>/dev/null || true" >/dev/null 2>&1 || true
    done
    [[ "$sig" == "TERM" ]] && sleep 2
  done
else
  log "Container '$CONTAINER' is not running; skipping in-container kill."
fi

# 2) Host TERM -> KILL.
log "Stopping host-side eval stack processes..."
host_pids=$(list_host_pids | awk '{print $1}' | sort -u | tr '\n' ' ')
if [[ -n "${host_pids// /}" ]]; then
  # Try our own UID first; fall back to sudo for processes owned by root
  # (the container's own entrypoint runs as root and shows up on the host).
  kill -TERM $host_pids 2>/dev/null || true
  sleep 2
  still=$(list_host_pids | awk '{print $1}' | sort -u | tr '\n' ' ')
  if [[ -n "${still// /}" ]]; then
    kill -KILL $still 2>/dev/null || true
    sleep 1
    still=$(list_host_pids | awk '{print $1}' | sort -u | tr '\n' ' ')
    if [[ -n "${still// /}" ]]; then
      log "Some processes need sudo to kill: $still"
      sudo kill -KILL $still 2>/dev/null || true
    fi
  fi
fi

# 3) Reap orphan podman conmons. These are visible on the host even after
#    ``podman ps -a`` shows nothing because conmon outlives its container
#    when podman crashes. They keep child ROS processes alive forever.
log "Reaping orphan podman conmon for '$CONTAINER'..."
conmon_pids=$(list_orphan_conmons | awk '{print $1}' | tr '\n' ' ')
if [[ -n "${conmon_pids// /}" ]]; then
  sudo kill -KILL $conmon_pids 2>/dev/null || true
fi

# 4) Optional nuke: restart the docker container. The container CMD is
#    ``entrypoint.sh`` which auto-restarts the sim as root, so callers who
#    use --hard should normally re-launch entrypoint themselves afterwards.
if [[ $HARD -eq 1 ]] && container_alive; then
  log "Restarting container '$CONTAINER' (kicks any open distrobox shells)..."
  docker restart "$CONTAINER" >/dev/null
fi

# 5) Verify.
remaining_host=$(list_host_pids | wc -l | tr -d ' ')
remaining_container=$(list_container_pids 2>/dev/null | wc -l | tr -d ' ')
remaining_conmon=$(list_orphan_conmons | wc -l | tr -d ' ')

log "After cleanup: host=$remaining_host, container=$remaining_container, conmon=$remaining_conmon"

if [[ "$remaining_host" -ne 0 || "$remaining_container" -ne 0 || "$remaining_conmon" -ne 0 ]]; then
  log "Some processes are still alive. Run with --check to inspect."
  exit 1
fi

log "Clean."
exit 0
