"""Validate an entire colour programme before its first robot operation."""
COLORS = ('red', 'white', 'orange')
MAX_COLOR_STEPS = 8


def color_step_pairs(tasks, held_color=None):
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= MAX_COLOR_STEPS:
        raise ValueError(f'一次支持 1～{MAX_COLOR_STEPS} 步颜色抓放')
    pairs = []
    support = {}
    holding = held_color
    for index, task in enumerate(tasks, 1):
        if not isinstance(task, dict):
            raise ValueError(f'第 {index} 步不是有效任务')
        action = task.get('action')
        source = task.get('target_color')
        if action == 'place':
            source = task.get('source_color') or holding
        destination = 'hold' if action == 'pick' else task.get('destination_color', 'box')
        if action not in ('pick_place', 'pick', 'place') or source not in COLORS or destination not in ('box', 'hold', *COLORS):
            raise ValueError(f'第 {index} 步缺少明确的颜色或放置目标')
        if destination == source:
            raise ValueError(f'第 {index} 步不能把同一个方块放到自身上，请明确另一颜色')
        if holding and holding != source:
            raise ValueError(f'第 {index} 步开始时仍夹着另一颜色，请先放置当前方块')
        if action == 'place' and not holding:
            raise ValueError(f'第 {index} 步没有可放置的持物')
        # Picking a known lower block invokes the existing clearing routine.
        # Its upper block is moved to an observed buffer before the pick.
        for upper, lower in list(support.items()):
            if lower == source:
                support[upper] = None
        support[source] = None
        if destination in COLORS and support.get(destination) is not None:
            raise ValueError(f'第 {index} 步会形成三层叠塔；当前颜色流程仅支持两层')
        if destination in COLORS and destination in support.values():
            raise ValueError(f'第 {index} 步的支撑方块上已有方块，请先移开上层方块')
        support[source] = destination if destination in COLORS else None
        holding = source if destination == 'hold' else None
        pairs.append((source, destination))
    return pairs
