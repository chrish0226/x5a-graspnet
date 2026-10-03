"""In-memory interaction session for multi-turn grasp/place.

The cloud /parse gateway is stateless (it only receives the utterance text), so
"放到白色方块上" cannot be resolved there: it looks exactly like "grasp the
white cube". The follow-up intent is therefore decided locally (see
command_parser.follow_up_intent) and this module remembers what the gripper is
actually holding between commands.

The ground truth for ``holding`` is the executor: /x5a_remote_grasp/execute
returns the real execution stage plus a ``holding`` flag, and
/x5a_remote_grasp/executor_state can be queried to re-sync after a web_agent
restart. This object is only the agent-side mirror of that state; it lives in
process memory and is intentionally not persisted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class HeldObject:
    """What the gripper is holding right now."""

    prompt: str = ""
    object: str = ""
    color: str = ""
    width_m: float = 0.0
    held_since: str = ""

    def label(self) -> str:
        """Chinese label for the phone UI, e.g. 「黑笔」/「black pen」."""
        parts = [part for part in (self.color, self.object) if part]
        if parts:
            if all(part.isascii() for part in parts):
                return " ".join(parts)
            return "".join(parts)
        return self.prompt or "物体"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "object": self.object,
            "color": self.color,
            "width_m": self.width_m,
            "held_since": self.held_since,
            "label": self.label(),
        }


@dataclass
class Session:
    held: Optional[HeldObject] = None
    history: list = field(default_factory=list)

    @property
    def holding(self) -> bool:
        return self.held is not None

    def hold(
        self,
        prompt: str,
        object_name: str = "",
        color: str = "",
        width_m: float = 0.0,
    ) -> HeldObject:
        self.held = HeldObject(
            prompt=str(prompt or ""),
            object=str(object_name or ""),
            color=str(color or ""),
            width_m=float(width_m or 0.0),
            held_since=datetime.now().isoformat(timespec="seconds"),
        )
        self._remember(f"hold {self.held.prompt}")
        return self.held

    def clear(self, reason: str = "") -> None:
        if self.held is not None:
            self._remember(f"clear {self.held.prompt} ({reason})")
        self.held = None

    def observe_executor(self, result: Dict[str, Any], prompt: str = "",
                         object_name: str = "", color: str = "",
                         width_m: float = 0.0) -> None:
        """Actual holding feedback applies on both successful and failed tasks.

        Retain known object identity when an operation fails. If a failed task
        discovers an already-held object, its attempted target is not evidence
        of that object's identity. Missing feedback leaves the state intact.
        """
        holding = result.get('holding')
        if holding is False:
            self.clear('executor says the gripper is empty')
        elif holding is True and self.held is None:
            if result.get('success'):
                self.hold(prompt, object_name, color, width_m)
            else:
                self.hold(prompt='物体')

    def _remember(self, note: str) -> None:
        self.history.append(note)
        del self.history[:-20]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "holding": self.holding,
            "held_object": None if self.held is None else self.held.as_dict(),
        }

    def held_label(self) -> str:
        return "" if self.held is None else self.held.label()
