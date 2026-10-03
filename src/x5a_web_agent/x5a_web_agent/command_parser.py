#!/usr/bin/env python3
"""Natural-language command parser for X5A voice pick-place.

First version is rule/keyword based. Later swap in LLMCommandParser.
The parser may only emit the whitelist below. It never emits joint
angles, Cartesian trajectories, /arm_cmd, shell, or Python code.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence


ALLOWED_ACTIONS = frozenset({"pick_place", "unknown", "cancel", "sequence"})
ALLOWED_COLORS = frozenset({"red", "white", "orange"})
ALLOWED_SELECTORS = frozenset({"nearest", "farthest"})
ALLOWED_TARGETS = ALLOWED_COLORS | ALLOWED_SELECTORS

COLOR_ALIASES = {
    "red": ("red", "红", "红色", "红的", "红方块", "红色方块", "红色的"),
    "white": ("white", "白", "白色", "白的", "白方块", "白色方块", "白色的"),
    "orange": (
        "orange",
        "橙",
        "橙色",
        "橘色",
        "桔色",
        "橙的",
        "橘的",
        "橙色方块",
        "橘色方块",
        "橙色的",
        "橘色的",
    ),
}

PICK_HINTS = (
    "抓",
    "拿",
    "取",
    "夹",
    "捡",
    "挑",
    "搬走",
    "拿走",
    "抓取",
    "帮我",
    "请",
    "pick",
    "grasp",
    "grab",
)

STOP_HINTS = ("停止", "停下", "取消", "别抓", "不要动", "急停", "stop", "cancel")

# ---------------------------------------------------------------------------
# 后续指令（多轮交互）：云端 /parse 是无状态的，"放到白色方块上" 在那里和
# "抓白色方块" 长得一样，所以放置/放下意图必须在本地判定。
# ---------------------------------------------------------------------------
PLACE_VERBS = (
    "放到", "放在", "放进", "搁在", "搁到", "摆在", "摆到", "扔进", "扔到",
    "塞进", "装进", "移到", "挪到", "put", "place",
)
GRASP_VERBS = (
    "抓", "拿", "取", "夹", "捡", "挑", "搬", "拾", "握", "捏",
    "pick", "grasp", "grab",
)
RELEASE_VERBS = (
    "放下", "松开", "松手", "撒手", "放手", "丢掉", "扔掉", "放掉", "释放",
    "release", "drop",
)
RELATION_IN_HINTS = ("里面", "里边", "里头", "内部", "里", "中", "内")
RELATION_ON_HINTS = ("上面", "上边", "上头", "顶部", "顶上", "上", "on", "onto")
# 抽取名词短语时要去掉的尾缀（关系词、语气词、标点）
_TRAILING_NOISE = (
    "上面", "上边", "上头", "顶部", "顶上", "里面", "里边", "里头", "内部",
    "一下", "那儿", "那里", "这里", "去", "吧", "呢", "啊", "上", "里", "中",
    "内", "的",
    "。", "，", "！", "？", "!", "?", ".", ",",
)


def _tail_after(text: str, markers: Sequence[str]) -> Optional[str]:
    """第一个 marker 之后的部分（用于取放置目标名词短语）。"""
    position = -1
    for marker in markers:
        found = text.find(marker)
        if found >= 0 and (position < 0 or found + len(marker) < position):
            position = found + len(marker)
    return None if position < 0 else text[position:]


def _head_before(text: str, markers: Sequence[str]) -> str:
    """第一个 marker 之前的部分（动词前的宾语短语）。"""
    position = -1
    for marker in markers:
        found = text.find(marker)
        if found >= 0 and (position < 0 or found < position):
            position = found
    return "" if position < 0 else text[:position]


# 动词前只出现这些词时，说明那句话没有点名「要抓什么」，是接着上一步的后续指令
_PRE_VERB_FUNCTION_WORDS = (
    "把那个", "把那", "把这个", "把这", "帮我", "帮忙", "给我", "请", "把",
    "将", "你", "它", "他", "那个", "这个", "再", "接着", "然后", "先", "给",
)


def _names_a_grasp_target(pre_verb: str) -> bool:
    """动词前是否点了具体物体名（点了 = 完整的一步任务，不是后续指令）。"""
    cleaned = pre_verb.strip()
    changed = True
    while changed and cleaned:
        changed = False
        for word in _PRE_VERB_FUNCTION_WORDS:
            if cleaned.startswith(word):
                cleaned = cleaned[len(word):].strip()
                changed = True
                break
            if cleaned.endswith(word):
                cleaned = cleaned[: -len(word)].strip()
                changed = True
                break
    return len(cleaned) >= 2


def _strip_trailing_noise(text: str) -> str:
    cleaned = text.strip()
    changed = True
    while changed and cleaned:
        changed = False
        for noise in _TRAILING_NOISE:
            if cleaned.endswith(noise) and len(cleaned) > len(noise):
                cleaned = cleaned[: -len(noise)].strip()
                changed = True
                break
    return cleaned.strip()


def follow_up_intent(text: str) -> Optional[Dict[str, Any]]:
    """判定「接着上一步动作」的后续指令；不是后续指令返回 None。

    返回 {"action": "release"}，或
         {"action": "remote_place", "target_text": 名词短语, "relation": "on"|"in"}。

    规则刻意做成纯本地、可预测：含放置动词且不含抓取动词、且动词前没有点名
    要抓的物体 = 放置后续指令（否则云端会把「放到白色方块上」理解成「抓白色
    方块」）。像「把红色方块放到盒子里」那样点名了物体的，仍交给云端当一步
    任务处理；此时若手上还拿着东西，app 会拒绝并提示先说「放下」。
    """
    compact = _normalize(text)
    if not compact:
        return None
    has_place_verb = any(hint in compact for hint in PLACE_VERBS)
    if any(hint in compact for hint in RELEASE_VERBS) and not has_place_verb:
        return {"action": "release", "source": "rule"}
    if not has_place_verb:
        return None
    if any(hint in compact for hint in GRASP_VERBS):
        return None
    if _names_a_grasp_target(_head_before(compact, PLACE_VERBS)):
        return None
    tail = _tail_after(compact, PLACE_VERBS) or ""
    relation = "on"
    if any(hint in tail for hint in RELATION_IN_HINTS):
        relation = "in"
    elif any(hint in tail for hint in RELATION_ON_HINTS):
        relation = "on"
    return {
        "action": "remote_place",
        "target_text": _strip_trailing_noise(tail),
        "relation": relation,
        "source": "rule",
    }

SEQUENCE_HINTS = (
    "先", "再", "然后", "接着", "之后", "和", "跟", "以及", "还有", "与",
    "都", "全部", "两个", "三个", "and then", "then", " and ",
)
NEAREST_HINTS = ("最近", "最近的", "离得最近", "离盒子最近", "nearest", "closest")
FARTHEST_HINTS = ("最远", "最远的", "离得最远", "离盒子最远", "farthest", "furthest")

UNKNOWN_RESULT = {
    "action": "unknown",
    "message": "暂时无法理解该指令",
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip().lower())


def _find_colors_in_order(text: str) -> List[str]:
    """Return colors in the order they appear in the utterance."""
    hits: List[tuple[int, str]] = []
    for color, aliases in COLOR_ALIASES.items():
        for alias in sorted(aliases, key=len, reverse=True):
            idx = text.find(alias.lower() if alias.isascii() else alias)
            # Chinese aliases are matched against the original mixed string.
            idx = text.find(alias) if idx < 0 else idx
            if idx < 0:
                idx = text.find(alias.lower())
            if idx >= 0:
                hits.append((idx, color))
                break
    hits.sort(key=lambda item: item[0])
    seen = set()
    ordered: List[str] = []
    for _, color in hits:
        if color not in seen:
            seen.add(color)
            ordered.append(color)
    return ordered


def sanitize_task(task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Reject anything outside the first-version whitelist."""
    if not isinstance(task, dict):
        return None
    action = str(task.get("action", "")).strip()
    if action not in ALLOWED_ACTIONS:
        return None
    if action == "pick_place":
        color = str(task.get("target_color", "")).strip().lower()
        if color not in ALLOWED_TARGETS:
            return None
        return {"action": "pick_place", "target_color": color}
    if action == "cancel":
        return {"action": "cancel"}
    if action == "sequence":
        tasks = []
        for item in task.get("tasks") or []:
            cleaned = sanitize_task(item)
            if cleaned is None or cleaned.get("action") != "pick_place":
                return None
            tasks.append(cleaned)
        if not tasks:
            return None
        return {"action": "sequence", "tasks": tasks}
    return {
        "action": "unknown",
        "message": str(task.get("message") or UNKNOWN_RESULT["message"]),
    }


class CommandParser(ABC):
    @abstractmethod
    def parse_command(self, text: str) -> Dict[str, Any]:
        raise NotImplementedError


class RuleBasedCommandParser(CommandParser):
    def parse_command(self, text: str) -> Dict[str, Any]:
        raw = (text or "").strip()
        if not raw:
            return dict(UNKNOWN_RESULT)
        compact = _normalize(raw)
        if any(hint in compact for hint in STOP_HINTS):
            return {"action": "cancel"}

        colors = _find_colors_in_order(compact)
        looks_like_sequence = (
            len(colors) >= 2 and any(hint in compact for hint in SEQUENCE_HINTS)
        )
        if looks_like_sequence:
            return sanitize_task(
                {
                    "action": "sequence",
                    "tasks": [
                        {"action": "pick_place", "target_color": color}
                        for color in colors
                    ],
                }
            ) or dict(UNKNOWN_RESULT)

        has_nearest = any(hint in compact for hint in NEAREST_HINTS)
        has_farthest = any(hint in compact for hint in FARTHEST_HINTS)
        if has_farthest and not colors:
            return {"action": "pick_place", "target_color": "farthest"}
        if has_nearest and not colors:
            return {"action": "pick_place", "target_color": "nearest"}

        if not colors:
            return dict(UNKNOWN_RESULT)

        # A lone color, or any utterance that names one allowed color, is pick_place.
        return {"action": "pick_place", "target_color": colors[0]}


class LLMCommandParser(CommandParser):
    """Reserved hook. An LLM may only fill this JSON schema.

    Allowed output:
      {"action":"pick_place","target_color":"red"|"white"|"orange"}
      {"action":"unknown","message":"..."}
      {"action":"cancel"}
      {"action":"sequence","tasks":[{"action":"pick_place","target_color":"..."}]}

    Forbidden output:
      joint angles, Cartesian trajectories, /arm_cmd, shell, ros2 topic pub,
      Python code, or any extra keys that would bypass MoveIt.
    """

    SCHEMA = {
        "type": "object",
        "required": ["action"],
        "additionalProperties": False,
        "properties": {
            "action": {
                "type": "string",
                "enum": sorted(ALLOWED_ACTIONS),
            },
            "target_color": {
                "type": "string",
                "enum": sorted(ALLOWED_TARGETS),
            },
            "message": {"type": "string"},
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["action", "target_color"],
                    "additionalProperties": False,
                    "properties": {
                        "action": {"type": "string", "enum": ["pick_place"]},
                        "target_color": {
                            "type": "string",
                            "enum": sorted(ALLOWED_TARGETS),
                        },
                    },
                },
            },
        },
    }

    def __init__(self, fallback: Optional[CommandParser] = None) -> None:
        self.fallback = fallback or RuleBasedCommandParser()

    def parse_command(self, text: str) -> Dict[str, Any]:
        # First version does not call a model. Keep the rule parser in charge.
        return self.fallback.parse_command(text)

    @staticmethod
    def validate_model_json(payload: Any) -> Dict[str, Any]:
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                return dict(UNKNOWN_RESULT)
        cleaned = sanitize_task(payload) if isinstance(payload, dict) else None
        return cleaned or dict(UNKNOWN_RESULT)


def parse_command(text: str) -> Dict[str, Any]:
    return RuleBasedCommandParser().parse_command(text)


def _self_test() -> None:
    parser = RuleBasedCommandParser()
    cases = {
        "抓红色": {"action": "pick_place", "target_color": "red"},
        "把红色方块拿走": {"action": "pick_place", "target_color": "red"},
        "帮我拿一下红色的": {"action": "pick_place", "target_color": "red"},
        "红色": {"action": "pick_place", "target_color": "red"},
        "抓一下那个红色方块": {"action": "pick_place", "target_color": "red"},
        "白色": {"action": "pick_place", "target_color": "white"},
        "橙色": {"action": "pick_place", "target_color": "orange"},
        "橘色": {"action": "pick_place", "target_color": "orange"},
        "orange": {"action": "pick_place", "target_color": "orange"},
        "停止": {"action": "cancel"},
        "先抓红色，再抓白色": {
            "action": "sequence",
            "tasks": [
                {"action": "pick_place", "target_color": "red"},
                {"action": "pick_place", "target_color": "white"},
            ],
        },
        "请把红色和白色的方块放入盒子里": {
            "action": "sequence",
            "tasks": [
                {"action": "pick_place", "target_color": "red"},
                {"action": "pick_place", "target_color": "white"},
            ],
        },
        "把最近的方块放入盒子": {"action": "pick_place", "target_color": "nearest"},
        "把最远的放入盒子里": {"action": "pick_place", "target_color": "farthest"},
        "跳舞": UNKNOWN_RESULT,
        "": UNKNOWN_RESULT,
    }
    failed = 0
    for text, expected in cases.items():
        got = parser.parse_command(text)
        if got != expected:
            failed += 1
            print(f"FAIL {text!r}: {got} != {expected}")
    if failed:
        raise SystemExit(failed)
    print("command_parser self-test: PASS")


if __name__ == "__main__":
    _self_test()
