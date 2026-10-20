#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.bash"
mkdir -p "$WRIST_WS_ROOT/vendor/asr" "$WRIST_WS_ROOT/models"
# Install only the ASR runtime, keeping the existing ROS/NumPy packages intact.
python3 -m pip install --no-deps --only-binary=:all: --target "$WRIST_WS_ROOT/vendor/asr" \
  sherpa-onnx==1.13.8 sherpa-onnx-core==1.13.8
wrist_asr_name=sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17
if [[ ! -f "$WRIST_WS_ROOT/models/$wrist_asr_name/model.int8.onnx" ]]; then
  curl -fL --retry 2 --connect-timeout 10 \
    -o "$WRIST_WS_ROOT/models/sensevoice-int8.tar.bz2" \
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$wrist_asr_name.tar.bz2"
  python3 - <<'PY'
import os, tarfile
from pathlib import Path
root = Path(os.environ['WRIST_WS_ROOT'])/'models'
with tarfile.open(root/'sensevoice-int8.tar.bz2', 'r:bz2') as archive:
    for member in archive.getmembers():
        p = Path(member.name)
        if p.is_absolute() or '..' in p.parts or not (member.isfile() or member.isdir()):
            raise RuntimeError('Unexpected archive member')
    archive.extractall(root)
PY
fi
python3 - <<'PY'
from x5a_web_agent.asr import SenseVoiceSpeechToText
asr = SenseVoiceSpeechToText()
asr.preload()
print('SenseVoiceSmall INT8 已就绪，CPU 线程数：', asr.num_threads)
PY
