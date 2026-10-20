"""Ordered language tasks and readable labels; contains no robot interfaces."""
from dataclasses import dataclass, field
import re
import threading
import time


MAX_SEQUENCE_STEPS = 8
# Split explicit order markers, rather than treating every mentioned colour as
# an object to pick. Spatial phrases such as “红色和白色方块之间” stay intact.
_SEPARATOR = re.compile(r'[，,。；;\s]*(?:再然后|然后|接着|随后|之后|再把|再将|再抓|再拿|再放|\band\s+then\b|\bthen\b)\s*', re.I)


def split_sequence(text):
    text = str(text or '').strip()
    pieces, start = [], 0
    for match in _SEPARATOR.finditer(text):
        clause = text[start:match.start()].strip(' ，,。；;')
        if clause:
            pieces.append(clause)
        # “再把/再抓/再放” also contains the verb that starts the next task.
        marker = match.group().strip(' ，,。；;\t\r\n')
        start = match.end()-1 if marker in {'再把', '再将', '再抓', '再拿', '再放'} else match.end()
    tail = text[start:].strip(' ，,。；;')
    if tail:
        pieces.append(tail)
    elif pieces:
        raise ValueError('顺序连接词后缺少下一步指令，整条指令未执行')
    if len(pieces) < 2:
        return None
    if len(pieces) > MAX_SEQUENCE_STEPS:
        raise ValueError(f'一次最多接受 {MAX_SEQUENCE_STEPS} 步，请分批下达')
    return pieces


def describe_task(task):
    text = str(task.get('source_text') or '').strip(' ，,。；;')
    if text:
        return text
    action = task.get('action')
    if action in ('sequence', 'task_sequence'):
        return '；然后 '.join(describe_task(item) for item in task.get('tasks', []))
    if action == 'remote_grasp':
        prompt = str(task.get('prompt') or '')
        grasp, _, place = prompt.partition(' | place: ')
        grasp = grasp.split(' | ref:', 1)[0]
        if place:
            target, _, relation = place.partition('; prel: ')
            return f'抓取 {grasp} → 放到 {target}' + ('上' if relation == 'on' else '里')
        return f'抓取 {grasp}'
    if action == 'remote_place':
        return f'放到 {task.get("prompt", "目标")}' + ('上' if task.get('relation', 'on') == 'on' else '里')
    labels = {'red': '红块', 'white': '白块', 'orange': '橙块', 'box': '盒子', 'hold': '夹持等待'}
    source = labels.get(task.get('target_color'), task.get('target_color', '方块'))
    destination = labels.get(task.get('destination_color', 'box'), '目标')
    if action == 'pick_place' and task.get('target_color') in labels:
        return f'{source} → {destination}'
    if action == 'pick':
        return f'抓起{source}，夹持等待'
    if action == 'place':
        return f'当前持物 → {destination}'
    return {'go_ready': '回到观察位', 'go_home': '回原点', 'release': '原地松开夹爪',
            'pick_place': f'抓放 {task.get("target_color", "")}', 'cancel': '取消'}.get(action, str(action or '未知任务'))


def sequence_state_error(tasks, holding):
    """Reject impossible hand-state transitions before starting the first step."""
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= MAX_SEQUENCE_STEPS:
        return f'任务序列必须包含 1～{MAX_SEQUENCE_STEPS} 步'
    for index, task in enumerate(tasks, 1):
        if not isinstance(task, dict):
            return f'第 {index} 步不是有效任务'
        action = task.get('action')
        if action == 'remote_grasp':
            if holding:
                return f'第 {index} 步开始时仍在持物，请先补充放置目标'
            if not str(task.get('prompt') or '').strip():
                return f'第 {index} 步缺少抓取目标'
            holding = ' | place: ' not in task['prompt']
        elif action == 'remote_place':
            if not holding:
                return f'第 {index} 步没有可放置的物体，请先抓取'
            if not str(task.get('prompt') or '').strip():
                return f'第 {index} 步缺少放置目标'
            holding = False
        elif action == 'release':
            holding = False
        elif action == 'pick_place':
            if holding:
                return f'第 {index} 步开始时仍在持物，请先放置'
            holding = False
        elif action not in {'go_ready', 'go_home'}:
            return f'第 {index} 步暂不支持：{action or "未知指令"}'
    return ''


@dataclass
class CommandRun:
    number: int
    parsed: dict
    started: float = field(default_factory=time.monotonic)
    cancelled: threading.Event = field(default_factory=threading.Event)
    scheduled: bool = False
    step_index: int = 1
    step_total: int = 1

    def context(self):
        return dict(command_id=self.number, step_index=self.step_index,
                    step_total=self.step_total, elapsed_s=round(time.monotonic()-self.started, 2))
