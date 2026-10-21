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
    '抓红白橙', '把红白三个方块都放进盒子'])
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


@pytest.mark.parametrize('pronoun', ['它', '他', '她'])
def test_asr_pronouns_and_omitted_on(pronoun):
    assert parse_wrist_command(f'把{pronoun}放到橙色方块') == {'action':'place','destination_color':'orange'}


def test_named_held_cube_is_placement_with_expected_source():
    text = '把白色方块放到橙色方块上'
    assert parse_wrist_command(text)['action'] == 'pick_place'
    assert parse_wrist_command(text, {'holding':True,'holding_color':'white'}) == {
        'action':'place','source_color':'white','destination_color':'orange'}
    assert parse_wrist_command(text, {'holding':True,'holding_color':'red'})['action'] == 'unknown'


@pytest.mark.parametrize('tail', ['然后再拿起来放到红色方块上', '然后把它拿起来放到红块上',
                                  '再拿起来放红块上', '接着拿起来再放到红色上'])
def test_repick_refers_to_last_manipulated_cube(tail):
    assert parse_wrist_command('把白色方块放到橙色方块上，'+tail) == {
        'action':'task_sequence','tasks':[
            {'action':'pick_place','target_color':'white','destination_color':'orange'},
            {'action':'pick_place','target_color':'white','destination_color':'red'}]}


def test_repick_across_utterances_uses_confirmed_robot_context():
    context = {'holding':False,'last_moved_color':'white','last_support_color':'orange'}
    assert parse_wrist_command('再拿起来放红块上', context) == {
        'action':'pick_place','target_color':'white','destination_color':'red'}
    assert parse_wrist_command('再拿起来放红块上')['action'] == 'unknown'
    result = parse_wrist_command('再拿起来放白块上', context)
    assert result['action'] == 'unknown' and '拿橙块' in result['message']


def test_literal_self_reference_requests_clarification_before_first_step():
    task = parse_wrist_command('把白色方块放到橙色方块上，然后再拿起来放到白色方块上')
    assert task['action'] == 'unknown' and '不能放到自身' in task['message']
    assert '拿橙块' in task['message']


def test_explicit_swap_uses_existing_clearing_instead_of_rejecting_repeat_colors():
    task = parse_wrist_command('把白块放到橙块上，然后把橙块放到白块上')
    assert task['action'] == 'task_sequence'
    assert [(t['target_color'],t['destination_color']) for t in task['tasks']] == [
        ('white','orange'),('orange','white')]


@pytest.mark.parametrize('text', ['把白块放到橙块上，然后把它拿起来放到蓝块上',
    '把白块放到橙块上，然后转一下', '把白块放到橙块上，然后再拿起来放到白块上',
    '把白块放到橙块上，然后把红块放到橙块上'])
def test_invalid_later_step_rejects_entire_program(text):
    assert parse_wrist_command(text)['action'] == 'unknown'



def test_terminal_describes_every_resolved_color_step():
    from x5a_web_agent.task_sequence import describe_task
    task=parse_wrist_command('把白块放橙块上，然后再拿起来放红块上')
    assert describe_task(task) == '白块 → 橙块；然后 白块 → 红块'


@pytest.mark.parametrize('pronoun', ['它','他','她','这个方块'])
@pytest.mark.parametrize('connector', ['再把','然后把','接着把','再将'])
def test_implicit_repick_after_placement(pronoun,connector):
    text=f'把白色方块放到红色方块上，{connector}{pronoun}放到橙色方块上。'
    assert parse_wrist_command(text) == {'action':'task_sequence','tasks':[
        {'action':'pick_place','target_color':'white','destination_color':'red'},
        {'action':'pick_place','target_color':'white','destination_color':'orange'}]}


@pytest.mark.parametrize('tail', ['再放到橙色方块上','然后放到橙色方块上','接着放橙块上'])
def test_omitted_subject_after_complete_placement(tail):
    task=parse_wrist_command('先抓白块，然后放红块上，'+tail)
    assert task == {'action':'task_sequence','tasks':[
        {'action':'pick_place','target_color':'white','destination_color':'red'},
        {'action':'pick_place','target_color':'white','destination_color':'orange'}]}


@pytest.mark.parametrize('source,first,second', [('red','white','orange'),('orange','red','white')])
def test_reference_uses_source_not_last_mentioned_support(source,first,second):
    labels={'red':'红','white':'白','orange':'橙'}
    text=f'把{labels[source]}块放{labels[first]}块上，再把它放{labels[second]}块上'
    tasks=parse_wrist_command(text)['tasks']
    assert [task['target_color'] for task in tasks] == [source,source]
    assert [task['destination_color'] for task in tasks] == [first,second]


@pytest.mark.parametrize('text', ['再把它放橙块上','然后把他放橙块上','接着放到橙块上'])
def test_cross_utterance_continuation_requires_confirmed_source(text):
    assert parse_wrist_command(text)['action'] == 'unknown'
    context={'holding':False,'last_moved_color':'white','last_support_color':'red'}
    assert parse_wrist_command(text,context) == {'action':'pick_place','target_color':'white','destination_color':'orange'}
    assert parse_wrist_command(text,{'holding':True,'holding_color':'white'}) == {
        'action':'place','source_color':'white','destination_color':'orange'}


def test_plain_place_without_continuation_still_requires_current_holding():
    context={'holding':False,'last_moved_color':'white','last_support_color':'red'}
    assert parse_wrist_command('把它放橙块上',context) == {'action':'place','destination_color':'orange'}


def test_repeated_implicit_placements_keep_the_same_cube():
    task=parse_wrist_command('把白块放红块上，再把它放橙块上，然后放红块上')
    assert task['action'] == 'task_sequence'
    assert [(t['target_color'],t['destination_color']) for t in task['tasks']] == [
        ('white','red'),('white','orange'),('white','red')]


@pytest.mark.parametrize('tail', ['再把它放白块上','再把它放蓝块上','再把手里的方块放橙块上'])
def test_impossible_or_invalid_implicit_tail_is_rejected_whole(tail):
    assert parse_wrist_command('把白块放红块上，'+tail)['action'] == 'unknown'


@pytest.mark.parametrize('text', ['把白块放橙块上，然后把红块放白块上',
    '把白色放橙色上然后把红色放白色上', '先抓白块，然后放橙块上，再抓红块，然后放白块上'])
def test_explicit_three_layer_program(text):
    assert parse_wrist_command(text) == {'action':'task_sequence','tasks':[
        {'action':'pick_place','target_color':'white','destination_color':'orange'},
        {'action':'pick_place','target_color':'red','destination_color':'white'}]}
