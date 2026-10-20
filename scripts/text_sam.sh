#!/usr/bin/env bash
set -e
source "$(dirname -- "${BASH_SOURCE[0]}")/env.bash"
exec ros2 run x5a_web_agent text_cli --url http://127.0.0.1:8000 "$@"
