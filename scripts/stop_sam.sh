#!/usr/bin/env bash
set -e
exec /usr/bin/python3 "$(dirname -- "${BASH_SOURCE[0]}")/stop_sam.py" "$@"
