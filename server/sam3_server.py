"""SAM 3.1 (Object Multiplex) HTTP service.

Text-prompt concept segmentation over HTTP for LAN clients.

Run:
    CUDA_VISIBLE_DEVICES=1 /home/gjj/uavseg-sam3-pyenv/bin/python -m uvicorn \
        sam3_server:app --host 0.0.0.0 --port 8300
"""
from __future__ import annotations

import base64
import hashlib
from collections import OrderedDict
from fastapi.concurrency import run_in_threadpool
import io
import logging
import os
import sys
import threading
import time

import numpy as np
import torch

SAM3_REPO = os.environ.get("SAM3_REPO", "/home/gjj/uavseg-vfm/vendor/sam3")
if SAM3_REPO not in sys.path:
    sys.path.insert(0, SAM3_REPO)

from fastapi import FastAPI, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image

import sam3  # noqa: E402  (needs sys.path set up first)
from sam3 import build_sam3_image_model  # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402

CKPT = os.environ.get("SAM3_CHECKPOINT", "/mnt/data/gjj/sam3-weights/sam3.1_multiplex_fp32.pt")
BPE = os.environ.get("SAM3_BPE", os.path.join(SAM3_REPO, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz"))
MAX_UPLOAD_BYTES = 64 * 1024 * 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sam3_server")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

app = FastAPI(title="SAM 3.1 service", version="1.0")
from llm_gateway import register_llm_routes  # noqa: E402
register_llm_routes(app)
_model = None
_processor = None
_lock = threading.Lock()
# Only exact RGB pixels share an encoder result, across prompts on one frame.
# Keep at most two recent images; never share mutable prompt/results state.
_image_cache = OrderedDict()
_IMAGE_CACHE_TTL = 15.0
_IMAGE_CACHE_SIZE = 2


def image_state(processor, image):
    now = time.monotonic()
    key = (image.size, hashlib.sha256(image.tobytes()).digest())
    for old in list(_image_cache):
        if now-_image_cache[old][0] > _IMAGE_CACHE_TTL:
            del _image_cache[old]
    hit = key in _image_cache
    if hit:
        _, encoded = _image_cache.pop(key)
    else:
        encoded = processor.set_image(image)
    _image_cache[key] = (now, encoded)
    while len(_image_cache) > _IMAGE_CACHE_SIZE:
        _image_cache.popitem(last=False)
    state = dict(encoded)
    state['backbone_out'] = dict(encoded['backbone_out'])
    return state, hit
API_KEY = os.environ.get("SAM3_API_KEY", "").strip()


def _check_api_key(request: Request):
    if API_KEY and request.headers.get("X-API-Key") != API_KEY:
        raise HTTPException(401, "invalid or missing X-API-Key header")


def get_model():
    global _model, _processor
    if _model is None:
        t0 = time.time()
        log.info("loading SAM3.1 from %s on %s", CKPT, DEVICE)
        _model = build_sam3_image_model(
            bpe_path=BPE,
            device=DEVICE,
            checkpoint_path=CKPT,
            load_from_HF=False,
            enable_segmentation=True,
            enable_inst_interactivity=False,
        )
        _processor = Sam3Processor(_model, confidence_threshold=0.5, device=DEVICE)
        log.info("model ready in %.1fs", time.time() - t0)
    return _model, _processor


def encode_mask_png(mask) -> str:
    """bool HxW tensor -> base64 single-channel PNG (0/255)."""
    arr = mask.squeeze(0).detach().cpu().numpy()
    img = Image.fromarray((arr * 255).astype("uint8"), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def overlay_viz(image: Image.Image, masks, colors) -> str:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    for mask, color in zip(masks, colors):
        m = mask.squeeze(0).detach().cpu().numpy()
        base[m] = base[m] * 0.55 + np.asarray(color, dtype=np.float32) * 0.45
    combined = Image.fromarray(base.astype(np.uint8))
    buf = io.BytesIO()
    combined.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


PALETTE = [
    (255, 56, 56), (56, 156, 255), (80, 220, 100), (255, 200, 50),
    (200, 80, 255), (0, 220, 220), (255, 120, 200), (160, 160, 160),
]


@app.post("/segment_text")
async def segment_text(
    request: Request,
    file: bytes = File(...),
    prompt: str = Form(..., description="text concept, e.g. 'car'"),
    threshold: float = Form(0.5, ge=0.01, le=1.0),
    viz: bool = Form(False, description="also return overlay jpeg base64"),
):
    if len(file) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "image too large")
    _check_api_key(request)
    try:
        image = Image.open(io.BytesIO(file)).convert("RGB")
    except Exception:
        raise HTTPException(400, "cannot decode image")

    # GPU work runs in a worker, keeping the gateway/health event loop free.
    return await run_in_threadpool(_segment_image, image, prompt, threshold, viz)


def _segment_image(image, prompt, threshold, viz):
    t0 = time.monotonic()
    with _lock, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        acquired = time.monotonic()
        model, processor = get_model()
        state, cache_hit = image_state(processor, image)
        encoded = time.monotonic()
        processor.set_confidence_threshold(threshold)
        state = processor.set_text_prompt(prompt=prompt.strip(), state=state)

        masks = state["masks"]
        scores = state["scores"].detach().cpu().tolist()
        boxes_xyxy = state["boxes"].detach().cpu().tolist()

        objects = []
        for i in range(len(scores)):
            x0, y0, x1, y1 = [round(v, 1) for v in boxes_xyxy[i]]
            objects.append(
                {
                    "score": round(scores[i], 4),
                    "box_xyxy": [x0, y0, x1, y1],
                    "box_xywh": [x0, y0, round(x1 - x0, 1), round(y1 - y0, 1)],
                    "area_px": int(masks[i].sum().item()),
                    "mask_png_b64": encode_mask_png(masks[i]),
                }
            )
        viz_b64 = overlay_viz(image, masks, PALETTE) if viz and objects else None
    latency = round(time.monotonic() - t0, 3)

    log.info("prompt=%r n=%d threshold=%.2f %.2fs", prompt, len(objects), threshold, latency)
    return JSONResponse(
        {
            "model": "sam3.1_multiplex",
            "prompt": prompt,
            "threshold": threshold,
            "width": image.width,
            "height": image.height,
            "num_objects": len(objects),
            "objects": objects,
            "viz_jpeg_b64": viz_b64,
            "latency_s": latency,
            "timing": dict(image_cache_hit=cache_hit, queue_s=round(acquired-t0, 4),
                           image_encode_s=round(encoded-acquired, 4),
                           prompt_and_output_s=round(time.monotonic()-encoded, 4)),
        }
    )


@app.get("/health")
def health(request: Request):
    from llm_gateway import build_health_response
    base = {"model_loaded": _model is not None, "device": DEVICE, "image_cache_capacity": _IMAGE_CACHE_SIZE}
    out = build_health_response(request, sam_ready=(_model is not None))
    out.update(base)
    return out


@app.on_event("startup")
def warmup():
    try:
        _, processor = get_model()
        with _lock, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            # Load CUDA kernels as part of startup, not the first robot request.
            state = processor.set_image(Image.new("RGB", (640, 480)))
            processor.set_text_prompt(prompt="object", state=state)
        log.info("SAM model and CUDA inference warmed")
    except Exception as exc:
        log.exception("model load failed: %s", exc)
        raise
