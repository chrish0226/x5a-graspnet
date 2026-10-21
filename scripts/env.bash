# Source this file instead of the original arm workspace environment.
export PATH="/usr/bin:/bin:$PATH"
unset PYTHONHOME
WRIST_WS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export WRIST_WS_ROOT
# Remove overlays from previous terminals so imports cannot silently select the
# old arm implementation. Preserve system and vendor dependencies.
for wrist_var in AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH LD_LIBRARY_PATH; do
  wrist_value="${!wrist_var-}"
  wrist_filtered=""
  IFS=: read -r -a wrist_parts <<< "$wrist_value"
  for wrist_part in "${wrist_parts[@]}"; do
    case "$wrist_part" in
      /home/quella/arx/arm/*|"$WRIST_WS_ROOT"/install/*|"$WRIST_WS_ROOT"/build/*) continue ;;
    esac
    if [[ -n "$wrist_part" ]]; then wrist_filtered="${wrist_filtered:+$wrist_filtered:}$wrist_part"; fi
  done
  export "$wrist_var=$wrist_filtered"
done
source /opt/ros/humble/setup.bash
source /home/quella/repos/arx/ARX_X5/ROS2/X5_ws/install/local_setup.bash
if [[ -f "$WRIST_WS_ROOT/.deps/trac_ik_ws/install/local_setup.bash" ]]; then
  source "$WRIST_WS_ROOT/.deps/trac_ik_ws/install/local_setup.bash"
fi
if [[ -f "$WRIST_WS_ROOT/install/local_setup.bash" ]]; then
  source "$WRIST_WS_ROOT/install/local_setup.bash"
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export X5A_MQTT_ENABLED=0 X5A_TEXT_FORWARD=0 X5A_CLOUD_PARSE=0 X5A_GRASP_ROUTE=rule
unset wrist_var wrist_value wrist_filtered wrist_parts wrist_part
