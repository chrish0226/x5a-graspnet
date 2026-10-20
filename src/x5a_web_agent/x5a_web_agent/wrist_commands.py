"""Local grammar for prepare, folded home, box placement and two-cube stacking."""
import re
from x5a_web_agent.command_parser import COLOR_ALIASES

COLORS = ('red', 'white', 'orange')
_ALIASES = {alias: color for color in COLORS for alias in COLOR_ALIASES[color]}
_COLOR = '(?:' + '|'.join(re.escape(a) for a in sorted(_ALIASES, key=len, reverse=True)) + ')'
_OBJECT = r'(?:的)?(?:那个|这个)?(?:方块|积木|块)?'
_PREFIX = r'(?:请帮我|请|帮我|麻烦|帮忙)?(?:把|将|抓起|拿起|抓取|抓|拿|取|夹|pick|grab)?(?:一下)?(?:那个|这个)?'
_SUFFIX = r'(?:一下|吧|好吗|好么)?'
_PLACE = r'(?:放到|放在|摆到|摆在|放置到|叠到|叠在|放上|堆到)'
_STACK = re.compile(rf'{_PREFIX}(?P<source>{_COLOR}){_OBJECT}{_PLACE}(?P<destination>{_COLOR}){_OBJECT}(?:的)?(?:上面|上边|顶部|顶上|上){_SUFFIX}')
_BOX = re.compile(rf'{_PREFIX}(?P<source>{_COLOR}){_OBJECT}(?:放到|放进|放入|放在|扔进|放置到)(?:那个|这个)?(?:黑色|黑)?(?:盒子|盒|箱子)(?:里面|里|内|中)?{_SUFFIX}')
_PICK = re.compile(rf'{_PREFIX}(?P<source>{_COLOR}){_OBJECT}{_SUFFIX}')
_HELD_PREFIX = r'(?:请帮我|请|帮我|麻烦|帮忙)?(?:把|将)?(?:它|这个方块|这个|手里的方块|夹着的方块)?'
_HELD_STACK = re.compile(rf'{_HELD_PREFIX}{_PLACE}(?P<destination>{_COLOR}){_OBJECT}(?:的)?(?:上面|上边|顶部|顶上|上)?{_SUFFIX}')
_HELD_BOX = re.compile(rf'{_HELD_PREFIX}(?:放到|放进|放入|放在|放置到)(?:那个|这个)?(?:黑色|黑)?(?:盒子|盒|箱子)(?:里面|里|内|中)?{_SUFFIX}')
_BATCH_BOX = re.compile(rf'{_PREFIX}(?P<colors>{_COLOR}{_OBJECT}(?:(?:和|跟|与|以及|还有|、){_COLOR}{_OBJECT}){{1,2}})(?:都|全部|一起)?(?:放到|放进|放入|放在|放置到)(?:那个|这个)?(?:黑色|黑)?(?:盒子|盒|箱子)(?:里面|里|内|中)?{_SUFFIX}')
_PREPARE = ('准备', '预备', '就位', 'ready', '开始准备', '移动到观察位', '到观察位', '回观察位', '回到观察位', '返回观察位')
_HOME = ('回原点', '回到原点', '归原位', '返回原位', '回原位', '回到原位', '归位', '回位', '归零', '回零', '回零位', '回到零位', '归零位', '收起', '收起机械臂', '回初始位置', '回到初始位置', '返回初始位置', 'home')


def _named_color(alias):
    return _ALIASES[alias]


def parse_wrist_command(text):
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
