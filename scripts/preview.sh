#!/usr/bin/env bash
set -e
source "$(dirname -- "${BASH_SOURCE[0]}")/env.bash"
exec ros2 run rqt_image_view rqt_image_view /wrist_preview/debug_image
