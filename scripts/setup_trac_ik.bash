#!/usr/bin/env bash
# Build the optional plugin locally; no sudo or system package replacement.
set -euo pipefail
trac_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
trac_ws="$trac_root/.deps/trac_ik_ws"
trac_commit=ac8ad6ef9f8dddd04524545af79b86c4a7df1c18
mkdir -p "$trac_ws/src" "$trac_ws/sysroot"
if [[ ! -d "$trac_ws/src/trac_ik/.git" ]]; then
  git clone --no-checkout https://github.com/traclabs/trac_ik.git "$trac_ws/src/trac_ik"
  git -C "$trac_ws/src/trac_ik" checkout --detach "$trac_commit"
fi
if [[ "$(git -C "$trac_ws/src/trac_ik" rev-parse HEAD)" != "$trac_commit" ]]; then
  echo 'Unexpected TRAC-IK checkout; preserve it and inspect before rebuilding.' >&2
  exit 1
fi
if ! git -C "$trac_ws/src/trac_ik" apply --reverse --check "$trac_root/config/third_party/trac_ik_humble.patch" 2>/dev/null; then
  git -C "$trac_ws/src/trac_ik" apply --check "$trac_root/config/third_party/trac_ik_humble.patch"
  git -C "$trac_ws/src/trac_ik" apply "$trac_root/config/third_party/trac_ik_humble.patch"
fi
cd "$trac_ws/sysroot"
apt-get download libnlopt-dev libnlopt-cxx-dev libnlopt0 libnlopt-cxx0
for trac_deb in ./*.deb; do dpkg-deb -x "$trac_deb" .; done
cd "$trac_ws"
# ROS setup scripts may read unset variables.
set +u
source "$trac_root/scripts/env.bash"
set -u
export PKG_CONFIG_PATH="$trac_ws/sysroot/usr/lib/x86_64-linux-gnu/pkgconfig:${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="$trac_ws/sysroot/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
colcon build --packages-select trac_ik_lib trac_ik_kinematics_plugin --cmake-args \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
  -DNLopt_DIR="$trac_ws/sysroot/usr/lib/x86_64-linux-gnu/cmake/nlopt" \
  -DCMAKE_INSTALL_RPATH="$trac_ws/sysroot/usr/lib/x86_64-linux-gnu" \
  -DPython3_EXECUTABLE=/usr/bin/python3
