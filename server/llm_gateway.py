"""LLM + grasp gateway routes appended to the SAM3.1 service.

POST /parse     - Chinese instruction -> structured JSON via local Ollama (qwen3:4b)
POST /grasp     - multipart passthrough to GraspNet service (127.0.0.1:8301/grasp)
POST /v1/grasp  - authenticated passthrough to 127.0.0.1:8301/v1/grasp, the
                  contract expected by the ROS x5a_remote_grasp package.
                  Requires header X-API-Key matching env GRASP_API_KEY when set.
Only stdlib urllib is used, no new dependencies.
"""
import json
import os
import time
import urllib.error
import urllib.request

from fastapi import Depends, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:4b")
GRASP_URL = "http://127.0.0.1:8301/grasp"
GRASP_V1_URL = "http://127.0.0.1:8301/v1/grasp"
LOCATE_URL = "http://127.0.0.1:8301/v1/locate"
GRASP_API_KEY = os.environ.get("GRASP_API_KEY", "").strip()
if not GRASP_API_KEY:
    # read from gateway.env directly so any restart path (watchdog, manual,
    # cron) gets the key without needing the env wired through
    try:
        with open(os.environ.get("GATEWAY_ENV_FILE", os.path.join(os.path.dirname(__file__), "gateway.env")), encoding="utf-8") as fh:
            for line in fh:
                if line.strip().startswith("GRASP_API_KEY="):
                    GRASP_API_KEY = line.split("=", 1)[1].strip()
                    break
    except OSError:
        pass

SCHEMA = {
    "type": "object",
    "properties": {
        "object": {"type": "string"},
        "color": {"type": ["string", "null"]},
        "position": {"type": ["string", "null"]},
        "sam_prompt": {"type": "string"},
        "reference_prompt": {"type": ["string", "null"]},
        "relation": {"type": ["string", "null"]},
        "place_prompt": {"type": ["string", "null"]},
        "place_rel": {"type": ["string", "null"]},
        "action": {"type": "string"},
    },
    "required": ["object", "color", "position", "sam_prompt", "reference_prompt",
                 "relation", "place_prompt", "place_rel", "action"],
}

SYSTEM_PROMPT = (
    "你是机械臂抓取系统的指令解析模块。把用户的中文指令解析成JSON。字段规则：\n"
    "object: 目标物体英文名，单数名词(如 cube/bottle/cup/box/can/tape)；用户只说了颜色没说物体时，object 默认为 cube\n"
    "color: 目标物体自身颜色英文小写；若颜色属于参照物，color 为 null\n"
    "position: 方位(left/right/top/bottom/center)，没提方位则为 null\n"
    "sam_prompt: 目标物体本身的简短英文短语，只含颜色和名词，不要冠词和关系词，如 'black tape'、'orange cube'\n"
    "reference_prompt: 仅当【抓取目标本身】需要参照物才能定位时填写"
    "(如'红色方块旁边的方块'→'red cube')；注意'把X放到Y上面'中的Y是放置目标，"
    "填 place_prompt/place_rel，此时 reference_prompt 为 null\n"
    "relation: 抓取目标与参照物的空间关系：'on'(在参照物上面/里面) 或 "
    "'next_to'(在旁边)；没有抓取参照物则为 null\n"
    "place_prompt: 指令包含放置意图(放到/放进/放在)时，放置目标的简短英文短语"
    "(如 'black box')；纯抓取指令则为 null\n"
    "place_rel: 放置目标与被抓物体的关系：'in'(放进/放到里面) 或 'on'(放到上面)；"
    "无放置意图则为 null\n"
    "action: 用户要做的动作(pick/place/handover)，默认 pick"
)


class ParseRequest(BaseModel):
    text: str


def _ollama_chat(payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        raise HTTPException(503, f"ollama unreachable: {e}")


def _verify_key(request: Request):
    """Dependency: runs before body validation so a missing key short-circuits 401."""
    if GRASP_API_KEY and request.headers.get("X-API-Key") != GRASP_API_KEY:
        raise HTTPException(401, "invalid or missing X-API-Key header")


def build_health_response(request: Request, sam_ready: bool) -> dict:
    """Combined health for the x5a_remote_grasp health check:
    401 without a valid X-API-Key, then {"status": "ready"|"loading",
    "sam": {"ready": bool}, "graspnet": {"ready": bool}}."""
    _verify_key(request)
    graspnet_ready = False
    try:
        with urllib.request.urlopen("http://127.0.0.1:8301/grasp_health", timeout=3) as resp:
            g = json.loads(resp.read())
            graspnet_ready = bool(g.get("status") == "ok" and g.get("model_loaded"))
    except (urllib.error.URLError, ValueError, OSError):
        pass
    ready = bool(sam_ready and graspnet_ready)
    return {"status": "ready" if ready else "loading",
            "sam": {"ready": bool(sam_ready)},
            "graspnet": {"ready": graspnet_ready},
            "latency_info": None}


def register_llm_routes(app):
    @app.post("/parse", summary="中文指令 -> 抓取目标 JSON (qwen3:4b)")
    def parse(req: ParseRequest):
        payload = {
            "model": MODEL,
            "think": False,
            "stream": False,
            "format": SCHEMA,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": req.text},
            ],
            "options": {"temperature": 0.3, "num_predict": 120},
        }
        t0 = time.time()
        out = _ollama_chat(payload)
        latency = round(time.time() - t0, 3)
        content = out.get("message", {}).get("content", "")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            raise HTTPException(500, f"model returned invalid JSON: {content[:200]}")
        return {"latency_s": latency, "result": parsed, "eval_count": out.get("eval_count")}

    async def _proxy_grasp(request: Request, url: str, auth: bool):
        if auth:
            if GRASP_API_KEY and request.headers.get("X-API-Key") != GRASP_API_KEY:
                raise HTTPException(401, "invalid or missing X-API-Key header")
        body = await request.body()
        ctype = request.headers.get("content-type", "application/octet-stream")

        def forward():
            req = urllib.request.Request(
                url, data=body,
                headers={"Content-Type": ctype}, method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    return resp.status, resp.read()
            except urllib.error.HTTPError as e:
                return e.code, e.read()
            except urllib.error.URLError as e:
                raise HTTPException(503, f"grasp service unreachable: {e}")

        status, data = await run_in_threadpool(forward)
        return Response(content=data, status_code=status, media_type="application/json")

    @app.post("/grasp", summary="抓取位姿代理 -> GraspNet 服务 (8301, 无鉴权)")
    async def grasp_proxy(request: Request):
        return await _proxy_grasp(request, GRASP_URL, auth=False)

    @app.post("/v1/grasp", summary="x5a_remote_grasp 契约接口 (需 X-API-Key)",
              dependencies=[Depends(_verify_key)])
    async def grasp_v1_proxy(request: Request):
        return await _proxy_grasp(request, GRASP_V1_URL, auth=False)

    @app.post("/v1/locate", summary="只定位不抓取 (需 X-API-Key)",
              dependencies=[Depends(_verify_key)])
    async def locate_proxy(request: Request):
        return await _proxy_grasp(request, LOCATE_URL, auth=False)
