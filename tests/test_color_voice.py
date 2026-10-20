"""Spoken colour commands remain bounded to supported, explicit operations."""
import asyncio
from unittest.mock import Mock

import pytest
from x5a_web_agent.wrist_commands import parse_wrist_command


@pytest.mark.parametrize('colour,expected', [('红色','red'),('白色','white'),('橘色','orange'),('桔色','orange')])
@pytest.mark.parametrize('phrase', ['请你帮我夹起{}小方块', '麻烦你捡起{}的那个小积木',
                                   '帮我拿起来{}的立方体', '抓一下{}的小正方体'])
def test_spoken_pick_variants(colour, expected, phrase):
    assert parse_wrist_command(phrase.format(colour)) == {'action':'pick','target_color':expected}


@pytest.mark.parametrize('text,expected', [
    ('帮我把橘色的木块放到白色的那个小方块上面', {'action':'pick_place','target_color':'orange','destination_color':'white'}),
    ('把红块放白块上', {'action':'pick_place','target_color':'red','destination_color':'white'}),
    ('把白色叠橙色上面', {'action':'pick_place','target_color':'white','destination_color':'orange'}),
    ('把红色方块放回收纳盒里', {'action':'pick_place','target_color':'red'}),
    ('先抓红块，然后放到白块上', {'action':'pick_place','target_color':'red','destination_color':'white'}),
    ('先拿起来橘色小积木，再放进盒子里', {'action':'pick_place','target_color':'orange','destination_color':'box'}),
    ('把它放白色的小方块上', {'action':'place','destination_color':'white'}),
    ('把手里的方块收进收纳盒里', {'action':'place','destination_color':'box'}),
])
def test_spoken_place_and_atomic_pick_place(text, expected):
    assert parse_wrist_command(text) == expected


@pytest.mark.parametrize('text,colours', [
    ('把这三个小方块都收进盒子里', ['red','white','orange']),
    ('把红白橙三个方块依次放进盒子', ['red','white','orange']),
    ('按橙白红顺序放进盒子', ['orange','white','red']),
    ('按照白橘红的顺序放回收纳盒里', ['white','orange','red']),
    ('把白色方块放进盒子，然后把橘色积木放进盒子', ['white','orange']),
])
def test_spoken_batches_keep_order(text, colours):
    assert parse_wrist_command(text) == {'action':'sequence','tasks':[
        {'action':'pick_place','target_color':colour} for colour in colours]}


@pytest.mark.parametrize('text', ['先抓红块然后放到红色方块上','按红红白顺序放进盒子',
    '抓蓝色小方块','把红色方块放到白色方块旁边','三个方块叠起来',
    '先抓红块然后抓白块','把红块放进盒子然后', '准备然后抓红色',
    '抓红白橙', '把白色放橙色上然后把红色放白色上', '把红白三个方块都放进盒子'])
def test_ambiguous_or_unsupported_requests_do_not_partially_execute(text):
    assert parse_wrist_command(text)['action'] == 'unknown'


@pytest.mark.parametrize('text', ['不要抓红色','别拿白块然后拿橘色','不要放白块上','停止，然后抓红色','急停'])
def test_cancel_wins_over_other_verbs(text):
    assert parse_wrist_command(text) == {'action':'cancel'}


@pytest.mark.parametrize('text,query', [('你拿着什么颜色','holding'),('看到了哪些方块','vision'),('当前状态','all')])
def test_status_queries(text, query):
    assert parse_wrist_command(text) == {'action':'status','query':query}


def test_help_does_not_need_server_or_start_a_job(monkeypatch):
    import x5a_web_agent.wrist_app as web
    async def broadcast(payload): pass
    marker=object()
    monkeypatch.setattr(web,'bridge',None)
    monkeypatch.setattr(web,'_job',marker)
    monkeypatch.setattr(web,'broadcast',broadcast)
    reply=asyncio.run(web.dispatch(parse_wrist_command('支持什么指令')))
    assert reply['ok'] and '三个方块' in reply['message']
    assert web._job is marker


@pytest.mark.parametrize('query,part', [('holding','橙块'),('vision','白块：尚无定位'),('all','正在抬升')])
def test_queries_are_read_only_while_busy(monkeypatch, query, part):
    import x5a_web_agent.wrist_app as web
    async def refresh():
        return {'busy':True,'holding':True,'holding_color':'orange','stage_zh':'正在抬升','vision':{}}
    async def broadcast(payload):pass
    marker=object()
    monkeypatch.setattr(web,'bridge',Mock(side_effect=AssertionError('no robot operation')))
    monkeypatch.setattr(web,'refresh',refresh)
    monkeypatch.setattr(web,'broadcast',broadcast)
    monkeypatch.setattr(web,'_job',marker)
    reply=asyncio.run(web.dispatch({'action':'status','query':query}))
    assert reply['ok'] and part in reply['message'] and web._job is marker
