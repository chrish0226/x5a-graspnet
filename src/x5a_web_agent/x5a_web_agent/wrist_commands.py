"""Local grammar for prepare, folded home, box placement and two-cube stacking."""
import re
from x5a_web_agent.command_parser import COLOR_ALIASES
from x5a_web_agent.task_sequence import split_sequence
from x5a_pick_place.color_sequence import color_step_pairs

COLORS = ('red', 'white', 'orange')
_ALIASES = {alias: color for color in COLORS for alias in COLOR_ALIASES[color]}
_ALIASES.update({'橘': 'orange', '桔': 'orange'})
_COLOR = '(?:' + '|'.join(re.escape(a) for a in sorted(_ALIASES, key=len, reverse=True)) + ')'
_OBJECT = r'(?:的)?(?:那个|这个)?(?:小)?(?:方块|积木|正方体|立方体|木块|块)?'
_PREFIX = r'(?:请帮我|请|帮我|麻烦|帮忙)?(?:把|将|抓起|拿起|抓取|抓|拿|取|夹|pick|grab)?(?:一下)?(?:一个|一块|个)?(?:那个|这个)?'
_SUFFIX = r'(?:一下|吧|好吗|好么)?'
_PLACE = r'(?:放到|放在|摆到|摆在|放置到|叠到|叠在|放上|堆到)'
_STACK = re.compile(rf'{_PREFIX}(?P<source>{_COLOR}){_OBJECT}{_PLACE}(?P<destination>{_COLOR}){_OBJECT}(?:的)?(?:上面|上边|顶部|顶上|上)?{_SUFFIX}')
_BOX = re.compile(rf'{_PREFIX}(?P<source>{_COLOR}){_OBJECT}(?:放到|放进|放入|放在|扔进|放置到)(?:那个|这个)?(?:黑色|黑)?(?:盒子|盒|箱子)(?:里面|里|内|中)?{_SUFFIX}')
_PICK = re.compile(rf'{_PREFIX}(?P<source>{_COLOR}){_OBJECT}{_SUFFIX}')
_HELD_PREFIX = r'(?:请帮我|请|帮我|麻烦|帮忙)?(?:把|将)?(?:它|他|她|这个方块|这个|手里的方块|夹着的方块)?'
_HELD_STACK = re.compile(rf'{_HELD_PREFIX}{_PLACE}(?P<destination>{_COLOR}){_OBJECT}(?:的)?(?:上面|上边|顶部|顶上|上)?{_SUFFIX}')
_HELD_BOX = re.compile(rf'{_HELD_PREFIX}(?:放到|放进|放入|放在|放置到)(?:那个|这个)?(?:黑色|黑)?(?:盒子|盒|箱子)(?:里面|里|内|中)?{_SUFFIX}')
_BATCH_BOX = re.compile(
    rf'(?:按|按照)?{_PREFIX}(?P<colors>{_COLOR}{_OBJECT}'
    rf'(?:(?:和|跟|与|以及|还有|、)?{_COLOR}{_OBJECT}){{1,2}})'
    rf'(?:的?顺序|这三个方块|三个方块|这三块|三块)?'
    rf'(?:都|全部|一起|依次|按顺序)?(?:放到|放进|放入|放在|放置到)'
    rf'(?:那个|这个)?(?:黑色|黑)?(?:盒子|盒|箱子)(?:里面|里|内|中)?{_SUFFIX}')
_PREPARE = ('准备', '预备', '就位', 'ready', '开始准备', '移动到观察位', '到观察位', '回观察位', '回到观察位', '返回观察位')
_HOME = ('回原点', '回到原点', '归原位', '返回原位', '回原位', '回到原位', '归位', '回位', '归零', '回零', '回零位', '回到零位', '归零位', '收起', '收起机械臂', '回初始位置', '回到初始位置', '返回初始位置', 'home')


def _named_color(alias):
    return _ALIASES[alias]


def _parse_single(text):
    text = re.sub(r'[\s，。！？,.!?]+', '', str(text or '').lower())
    if any(word in text for word in ('取消', '停止', '停下', '别抓', '不要抓', '不要动', '别动', 'cancel', 'stop')):
        return {'action': 'cancel'}
    simple = re.sub(r'^(?:请帮我|帮我|麻烦|请)', '', text)
    simple = re.sub(r'(?:一下|吧)$', '', simple)
    if simple in ('镜像准备', '镜像侧准备', '右侧准备', '右边准备', '准备右侧', '准备右边', '到镜像观察位', '切换到镜像侧'):
        return {'action': 'prepare', 'observation_side': 'mirror'}
    if simple in ('当前侧准备', '原侧准备', '左侧准备', '左边准备', '准备左侧', '准备左边', '到当前观察位', '切换到当前侧'):
        return {'action': 'prepare', 'observation_side': 'current'}
    if simple in _PREPARE:
        return {'action': 'prepare'}
    if simple in _HOME:
        return {'action': 'go_home'}
    if simple in ('松开夹爪', '打开夹爪', '松开', '放下', 'release'):
        return {'action': 'release'}
    match = _STACK.fullmatch(text)
    if match:
        source, destination = (_named_color(match[name]) for name in ('source', 'destination'))
        if source == destination:
            return {'action': 'unknown', 'message': '叠放需要两种不同颜色的方块'}
        return {'action': 'pick_place', 'target_color': source, 'destination_color': destination}
    match = _BATCH_BOX.fullmatch(text)
    if match:
        colors = [_named_color(item.group()) for item in re.finditer(_COLOR, match['colors'])]
        if re.search(r'三(?:个|块)', text[match.end('colors'):]) and len(colors) != 3:
            return {'action': 'unknown', 'message': '请明确三种不同颜色，或直接说“把三个方块放进盒子”'}
        if len(set(colors)) != len(colors):
            return {'action': 'unknown', 'message': '每种颜色暂支持一个方块，请不要重复颜色'}
        return {'action': 'sequence', 'tasks': [{'action': 'pick_place', 'target_color': color} for color in colors]}
    match = _BOX.fullmatch(text)
    if match:
        return {'action': 'pick_place', 'target_color': _named_color(match['source'])}
    match = _HELD_STACK.fullmatch(text)
    if match:
        return {'action': 'place', 'destination_color': _named_color(match['destination'])}
    if _HELD_BOX.fullmatch(text):
        return {'action': 'place', 'destination_color': 'box'}
    match = _PICK.fullmatch(text)
    if match:
        return {'action': 'pick', 'target_color': _named_color(match['source'])}
    return {'action': 'unknown', 'message': '支持“准备”“抓白色方块”“放到红色方块上”“放进盒子里”和“把白色方块放到红色方块上”'}


def _normalize_spoken(text):
    text = re.sub(r'[\s，。！？,.!?；;]+', '', str(text or '').lower())
    text = re.sub(r'^(?:请你帮我|请帮我|麻烦你帮我|麻烦帮我|麻烦你|请你|能不能帮我|能不能|可以帮我|帮忙|给我|帮我|请)', '', text)
    text = re.sub(r'(?:一下|吧|好吗|好么|可以吗)$', '', text)
    text = re.sub(r'(?:抓起来|拿起来|夹起来|捡起来|取出来|拾起|捡起|拿走|夹起|夹住|抓住|抓一下|拿一下|夹一下)', '抓', text)
    text = re.sub(r'(?:堆叠在|叠放在|摆放在|放置在)', '放在', text)
    text = re.sub(r'(?:收纳盒|收纳箱|箱子)', '盒子', text)
    text = re.sub(r'(?:收进|收回|放回|装进)', '放进', text)
    # Only rewrite colour-adjacent object phrases; arbitrary spatial language
    # and unsupported colours must not silently become a grasp command.
    text = re.sub(r'([红白橙橘桔])(?:色)?(?:的)?(?:那(?:一)?个|这(?:一)?个)?(?:小)?(?:正方体|立方体|积木块|方木|木块|方块|积木|块)',
                  lambda m: ('橙' if m[1] in '橘桔' else m[1])+'色方块', text)
    text = re.sub(r'(?:放|叠)(?=[红白橙橘桔])', '放到', text)
    return text


def _unknown(message):
    return {'action': 'unknown', 'message': message}


def parse_wrist_command(text, context=None):
    raw = str(text or '').strip()
    # Cancellation wins over every other verb and every clause.
    if re.search(r'取消|停止|停下|(?:不要|别|先别|不许)(?:再)?(?:抓|拿|夹|放|动|执行)|急停|\bstop\b|\bcancel\b', raw, re.I):
        return {'action': 'cancel'}
    normalized = _normalize_spoken(raw)
    if normalized in ('帮助', '怎么用', '能做什么', '支持什么指令', '指令示例', 'help'):
        return {'action': 'help'}
    if normalized in ('拿着什么', '拿着什么颜色', '你拿着什么', '你拿着什么颜色', '现在拿着什么', '抓着什么', '夹着什么', '手里是什么'):
        return {'action': 'status', 'query': 'holding'}
    if normalized in ('看到了什么', '看到了哪些方块', '看到了什么颜色', '有哪些方块', '哪些方块能抓', '方块位置', '识别情况'):
        return {'action': 'status', 'query': 'vision'}
    if normalized in ('状态', '当前状态', '现在状态', '现在怎么样', '查看状态', '看一下状态', '任务进度', '完成了吗', '准备好了吗'):
        return {'action': 'status', 'query': 'all'}
    try:
        clauses = split_sequence(raw)
        if clauses:
            return _parse_program(clauses, context or {})
    except ValueError as exc:
        return _unknown(str(exc))
    if re.fullmatch(r'(?:把|将)?(?:这|那)?(?:三个|三块|所有|全部)(?:的)?(?:小)?(?:方块|积木|块)?(?:都|全部|一起|依次|按顺序)?放进(?:黑色|黑)?盒子(?:里面|里|内|中)?', normalized):
        return {'action': 'sequence', 'tasks': [{'action': 'pick_place', 'target_color': color} for color in COLORS]}
    task = _parse_single(normalized)
    if task['action'] == 'unknown':
        try:
            return _parse_program([raw], context or {})
        except ValueError as exc:
            return _unknown(str(exc))
    return _held_context(task, context or {})


def _held_context(task, context):
    if task.get('action') == 'pick_place' and context.get('holding'):
        source = context.get('holding_color')
        if source in COLORS and source == task['target_color']:
            return {'action': 'place', 'source_color': source,
                    'destination_color': task.get('destination_color', 'box')}
        if source in COLORS:
            return _unknown('当前夹着另一颜色，请先放置当前方块')
    return task


def _parse_program(clauses, context):
    held = context.get('holding_color') if context.get('holding') else None
    subject = held or context.get('last_moved_color')
    support = context.get('last_support_color')
    tasks = []
    for index, clause in enumerate(clauses, 1):
        text = _normalize_spoken(clause)
        text = re.sub(r'^(?:先|再)(?=把|将|抓|拿|夹|放)', '', text)
        task = _parse_single(text)
        if task['action'] == 'unknown':
            # Explicit re-pick with an omitted object, e.g. 再拿起来放到红块上.
            match = re.fullmatch(r'(?:把|将)?(?:它|他|她|这个方块)?(?:抓|拿|夹)(?:起来|起)?(?:它|他|她|这个方块)?(?:再)?(.*)', text)
            if match:
                tail = match[1]
                place = _parse_single(tail) if tail else {'action': 'pick'}
                if place['action'] in ('place', 'pick'):
                    if subject not in COLORS:
                        raise ValueError(f'第 {index} 步“再拿起来”缺少指代，请说出颜色')
                    destination = place.get('destination_color', 'hold')
                    if subject == destination:
                        label = {'red': '红', 'white': '白', 'orange': '橙'}
                        hint = f'；你是指拿{label[support]}块放到{label[destination]}块上吗？' if support in COLORS and support != destination else ''
                        raise ValueError(f'第 {index} 步默认指刚操作的{label[subject]}块，不能放到自身上'+hint)
                    task = {'action': 'pick', 'target_color': subject} if destination == 'hold' else {
                        'action': 'pick_place', 'target_color': subject, 'destination_color': destination}
        if task['action'] == 'place' and tasks and tasks[-1]['action'] == 'pick':
            previous = tasks.pop()
            task = {'action': 'pick_place', 'target_color': previous['target_color'],
                    'destination_color': task['destination_color']}
        if task['action'] == 'place' and not tasks and held in COLORS:
            task = {'action': 'pick_place', 'target_color': held, 'destination_color': task['destination_color']}
        if task['action'] not in ('pick_place', 'pick', 'place'):
            raise ValueError(f'第 {index} 步无法明确解析，整条指令未执行：{clause}')
        tasks.append(task)
        if task.get('target_color') in COLORS:
            subject = task['target_color']
        if task['action'] == 'pick_place':
            support = task.get('destination_color')
    color_step_pairs(tasks, held)
    if len(tasks) == 1:
        return _held_context(tasks[0], context)
    if (len(tasks) <= 3 and len({t.get('target_color') for t in tasks}) == len(tasks)
            and all(t['action'] == 'pick_place' and t.get('destination_color', 'box') == 'box' for t in tasks)):
        return {'action': 'sequence', 'tasks': tasks}
    return {'action': 'task_sequence', 'tasks': tasks}
