"""Speech-to-text backends.

Default is local SenseVoiceSmall INT8 on the robot host. The phone only records
audio and uploads it. Browser Web Speech API remains as an unused fallback.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import threading
from abc import ABC, abstractmethod
from pathlib import Path


def _disable_proxy_for_local_model() -> None:
    """Clash/Meta sets socks:// proxies that huggingface_hub cannot parse."""
    for key in (
        "ALL_PROXY",
        "all_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "SOCKS_PROXY",
        "socks_proxy",
    ):
        os.environ.pop(key, None)
    os.environ.setdefault("NO_PROXY", "*")
    os.environ.setdefault("no_proxy", "*")


class SpeechToText(ABC):
    name = "base"
    ready = True

    @abstractmethod
    def transcribe(
        self,
        audio_bytes: bytes,
        mime_type: str = "audio/webm",
        language: str = "zh",
    ) -> str:
        raise NotImplementedError


class BrowserSpeechToText(SpeechToText):
    name = "browser_web_speech"

    def transcribe(
        self,
        audio_bytes: bytes,
        mime_type: str = "audio/webm",
        language: str = "zh",
    ) -> str:
        raise RuntimeError(
            "BrowserSpeechToText runs on the phone. Send the transcript as text."
        )


class WhisperSpeechToText(SpeechToText):
    name = "faster_whisper"

    def __init__(self, model_size: str = "small") -> None:
        self.model_size = model_size
        self._model = None
        self._lock = threading.Lock()
        self.ready = False
        self.error = ""

    def preload(self) -> None:
        self._ensure_model()

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            _disable_proxy_for_local_model()
            from faster_whisper import WhisperModel

            try:
                self._model = WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="int8",
                    local_files_only=True,
                )
            except Exception:
                self._model = WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="int8",
                )
            self.ready = True
            self.error = ""
            return self._model

    def transcribe(
        self,
        audio_bytes: bytes,
        mime_type: str = "audio/webm",
        language: str = "zh",
    ) -> str:
        if not audio_bytes:
            return ""
        _disable_proxy_for_local_model()
        model = self._ensure_model()
        suffix = _suffix_for_mime(mime_type)
        with tempfile.TemporaryDirectory(prefix="x5a-asr-") as tmp:
            src = Path(tmp) / f"clip{suffix}"
            wav = Path(tmp) / "clip.wav"
            src.write_bytes(audio_bytes)
            if suffix != ".wav":
                proc = subprocess.run(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-i",
                        str(src),
                        "-ac",
                        "1",
                        "-ar",
                        "16000",
                        str(wav),
                    ],
                    capture_output=True,
                    text=True,
                )
                if proc.returncode != 0 or not wav.is_file():
                    raise RuntimeError(
                        proc.stderr.strip() or "ffmpeg could not decode uploaded audio"
                    )
            else:
                wav = src
            segments, _info = model.transcribe(
                str(wav),
                language=language or "zh",
                vad_filter=True,
                beam_size=5,
            )
            return "".join(segment.text for segment in segments).strip()


class SenseVoiceSpeechToText(SpeechToText):
    name = 'sensevoice_small_int8'

    def __init__(self, model_dir=None, num_threads=None):
        root = Path(os.environ.get('WRIST_WS_ROOT', str(Path.home()/'arx/wrist_ws')))
        self.model_dir = Path(model_dir or os.environ.get('X5A_SENSEVOICE_MODEL_DIR', str(
            root/'models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17')))
        self.runtime_dir = Path(os.environ.get('X5A_SENSEVOICE_RUNTIME_DIR', str(root/'vendor/asr')))
        self.num_threads = int(num_threads if num_threads is not None else os.environ.get('X5A_ASR_THREADS', '2'))
        if not 1 <= self.num_threads <= 8:
            raise ValueError('X5A_ASR_THREADS must be between 1 and 8')
        self._model = None
        self._lock = threading.Lock()
        self.ready = False
        self.error = ''

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            model, tokens = self.model_dir/'model.int8.onnx', self.model_dir/'tokens.txt'
            if not model.is_file() or not tokens.is_file():
                raise RuntimeError(f'SenseVoice 模型未安装：{self.model_dir}；请运行 scripts/install_asr.sh')
            # Keep the new runtime separate from ROS and the existing Python packages.
            if self.runtime_dir.is_dir() and str(self.runtime_dir) not in sys.path:
                sys.path.insert(0, str(self.runtime_dir))
            import sherpa_onnx
            self._model = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=str(model), tokens=str(tokens), num_threads=self.num_threads,
                language='zh', use_itn=True, provider='cpu', debug=False)
            self.ready, self.error = True, ''
            return self._model

    def preload(self):
        self._ensure_model()

    def transcribe(self, audio_bytes, mime_type='audio/webm', language='zh'):
        if not audio_bytes:
            return ''
        model = self._ensure_model()
        import numpy as np
        # Always normalize WAV as well: Android/browser sample rates and
        # channel counts differ. A seekable file also handles M4A moov-at-end.
        with tempfile.TemporaryDirectory(prefix='x5a-sensevoice-') as tmp:
            src = Path(tmp)/('clip'+_suffix_for_mime(mime_type))
            src.write_bytes(audio_bytes)
            proc = subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error',
                '-threads', '1', '-i', str(src), '-t', '31', '-vn', '-ac', '1', '-ar', '16000',
                '-f', 'f32le', 'pipe:1'], capture_output=True, timeout=20)
            if proc.returncode:
                raise RuntimeError(proc.stderr.decode(errors='replace').strip() or '无法解码录音')
            samples = np.frombuffer(proc.stdout, dtype='<f4').copy()
        if len(samples) > 30*16000:
            raise ValueError('录音超过 30 秒，请缩短指令后重试')
        if len(samples) < 1600 or not np.isfinite(samples).all() or np.max(np.abs(samples)) < .002:
            return ''
        np.clip(samples, -1., 1., out=samples)
        with self._lock:
            stream = model.create_stream()
            stream.accept_waveform(16000, samples)
            model.decode_stream(stream)
            text = stream.result.text
        return re.sub(r'<\|[^|]*\|>', '', text).strip()


def _suffix_for_mime(mime_type: str) -> str:
    name = (mime_type or "").lower()
    if "wav" in name:
        return ".wav"
    if "mp4" in name or "m4a" in name or "aac" in name:
        return ".m4a"
    if "ogg" in name or "opus" in name:
        return ".ogg"
    if "mpeg" in name or "mp3" in name:
        return ".mp3"
    return ".webm"


def create_asr(backend: str = 'sensevoice') -> SpeechToText:
    name = (backend or 'sensevoice').strip().lower()
    if name in ('sensevoice', 'sensevoice_small', 'sensevoice_small_int8'):
        return SenseVoiceSpeechToText()
    if name in {"browser", "web_speech", "browser_web_speech"}:
        return BrowserSpeechToText()
    if name in {"whisper", "faster-whisper", "faster_whisper", "local"}:
        return WhisperSpeechToText()
    raise ValueError(f"unknown ASR backend: {backend!r}")
