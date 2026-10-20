"""Phone voice/text UI for the independent wrist color workspace."""
from __future__ import annotations

import argparse
import asyncio
import os
import signal
import subprocess
import threading
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from x5a_web_agent.asr import SenseVoiceSpeechToText, WhisperSpeechToText, create_asr
from x5a_web_agent.terminal import TerminalBackend, TerminalConsole
from x5a_web_agent.wrist_bridge import WristBridge
from x5a_web_agent.wrist_commands import parse_wrist_command

STAGE_ZH = {
    'SELECTING_BUFFER': '正在为当前目标寻找空闲、可达的暂放位置',
    'CLEARING_BLOCKER': '正在抓取目标上方的方块',
    'BUFFER_PLACEMENT': '正在把上层方块放到暂放位',
    'HOLDING_ABOVE_SOURCE': '上层方块已抬升，正在继续转移到暂放位',
    'REOBSERVE_UNCOVERED': '上层方块已移开，正在重新定位目标',
    'IDLE': '请先说“准备”', 'ACCEPTED': '指令已接受', 'READY': '已到观察位，可以抓取',
    'MOVE_INITIAL_OBSERVATION': '正在移动到观察位', 'WAITING_VISION': '正在定位方块和盒子',
    'TARGET_FOUND': '已定位目标', 'MOVE_PRE_GRASP': '正在移动到方块上方',
    'REOBSERVING': '正在二次定位', 'CORRECTING_PRE_GRASP': '正在纠偏',
    'VERIFYING_CORRECTION': '正在复核位置', 'APPROACH': '正在接近方块',
    'GRASPING': '正在夹取', 'LIFTING': '正在抬升', 'MOVE_PRE_PLACE': '正在前往放置位置',
    'DESCENDING': '正在下降', 'RELEASING': '正在放置', 'RETREATING': '正在退回',
    'RETURN_HOME': '正在返回观察位', 'SUCCESS': '任务完成', 'FAILED': '任务失败',
    'DRY_RUN': '演练完成，未发送运动指令',
    'GO_HOME': '正在返回原始收起姿态', 'HOME': '已返回原始收起姿态；抓取前请说“准备”',
    'RECOVERING_STATE': '正在恢复夹持物体的碰撞场景',
    'WAITING_PLACE': '已抓起方块并返回观察位，等待放置指令',
}
static = (Path(get_package_share_directory('x5a_web_agent')) / 'static' / 'index.html').resolve().parent
app = FastAPI(title='X5A 腕部颜色抓放')
app.mount('/static', StaticFiles(directory=str(static)), name='static')
bridge = None
asr = create_asr(os.environ.get('X5A_ASR_BACKEND', 'sensevoice'))
state = {'stage': 'IDLE', 'stage_zh': STAGE_ZH['IDLE'], 'busy': False,
         'prepared': False, 'holding': None, 'transcript': '', 'parsed': None,
         'last_result': None, 'server_ready': False}
clients = set()
observers = []
_job = None
_submission_lock = asyncio.Lock()


class TextIn(BaseModel):
    text: str


class TaskIn(BaseModel):
    action: str = 'pick_place'
    target_color: str = ''
    destination_color: str = 'box'
    observation_side: str = ''
    tasks: list = []


async def broadcast(payload):
    for ws in list(clients):
        try:
            await ws.send_json(payload)
        except Exception:
            clients.discard(ws)
    for callback in observers:
        callback(payload)


async def update(**values):
    state.update(values)
    state['stage_zh'] = STAGE_ZH.get(state['stage'], state['stage'])
    if not state.get('busy'):
        if state['stage'] == 'IDLE' and state.get('last_result') is None:
            # The installed Android app retains its old error when message is null.
            state['last_result'] = {'success': True, 'kind': 'status',
                                    'message': '当前没有运行中的任务。请先说“准备”，到观察位后再抓取'}
        result = state.get('last_result')
        if state['stage'] == 'FAILED' and result and not result.get('success'):
            result = dict(result)
            original = result.setdefault('task_message', result.get('message', ''))
            result['message'] = '上次任务失败：' + original
            state['last_result'] = result
        vision = state.get('vision', {})
        parsed = state.get('parsed') or {}
        colors = ([item.get('target_color') for item in parsed.get('tasks', [])]
                  if parsed.get('action') == 'sequence' else [parsed.get('target_color', 'red')])
        color = colors[0] if colors else 'red'
        if color not in ('red', 'white', 'orange'):
            color = 'red'
        destination = (state.get('parsed') or {}).get('destination_color', 'box')
        if color in vision and destination in vision and state['stage'] in ('IDLE', 'READY', 'FAILED', 'SUCCESS'):
            label = {'red': '红块', 'white': '白块', 'orange': '橙块'}[color]
            destination_label = {'box': '盒子', 'red': '目标红块', 'white': '目标白块', 'orange': '目标橙块'}[destination]
            labels = {'red': '红块', 'white': '白块', 'orange': '橙块'}
            objects = '；'.join(labels[c] + vision[c]['status_zh'] for c in colors if c in labels and c in vision)
            current = f"当前定位：{objects or label + vision[color]['status_zh']}；{destination_label}{vision[destination]['status_zh']}"
            if parsed.get('action') == 'pick':
                current = '当前定位：' + label + vision[color]['status_zh']
            elif parsed.get('action') == 'place':
                current = '放置目标：' + destination_label + vision[destination]['status_zh']
            state['stage_zh'] += '。' + current
            if not state.get('initialized', state.get('prepared')) and state['stage'] != 'IDLE':
                state['stage_zh'] += '。请先说“准备”'
            elif not state.get('arm_fault') and state.get('initialized') and not state.get('at_observation', True):
                state['stage_zh'] += '。下次抓取会先自动返回观察位'
        if state.get('arm_fault'):
            state['stage_zh'] = '机械臂状态异常：' + state['arm_fault'] + '。' + state['stage_zh']
        elif state.get('placement_ready') and state.get('prepared') and state['stage'] == 'WAITING_PLACE':
            label = {'red':'红块','white':'白块','orange':'橙块'}.get(state.get('holding_color'),'方块')
            example = '白色' if state.get('holding_color') == 'red' else '红色'
            state['stage_zh'] = f'已抓起{label}并返回观察位，等待“放到{example}方块上”或“放进盒子里”等放置指令'
    await broadcast({'type': 'state', **state})


async def _status(stage):
    # Keep busy until the operation future finishes, even if ROS publishes a
    # terminal stage a moment before returning the result.
    await update(stage=stage)


class ConsoleBackend(TerminalBackend):
    name = '腕部颜色抓放'

    def __init__(self, loop):
        self.loop = loop

    def submit_text(self, text):
        asyncio.run_coroutine_threadsafe(handle_text(text), self.loop)

    def submit_task(self, task):
        asyncio.run_coroutine_threadsafe(dispatch(task), self.loop)

    def cancel(self):
        asyncio.run_coroutine_threadsafe(dispatch({'action': 'cancel'}), self.loop)

    def state(self):
        return dict(state)


@app.on_event('startup')
async def startup():
    global bridge
    loop = asyncio.get_running_loop()
    bridge = WristBridge(dry_run=os.environ.get('X5A_WEB_DRY_RUN', '0') == '1')
    bridge.start()
    bridge.set_stage_callback(lambda stage: asyncio.run_coroutine_threadsafe(_status(stage), loop))
    if os.environ.get('X5A_WEB_TERMINAL') == '1':
        console = TerminalConsole(ConsoleBackend(loop), prompt='wrist> ')
        observers.append(console.on_payload)
        console.start()
    if isinstance(asr, (WhisperSpeechToText, SenseVoiceSpeechToText)):
        def preload():
            try:
                asr.preload()
            except Exception as exc:
                asr.error = str(exc)
                print(f'语音模型加载失败：{exc}', flush=True)
        threading.Thread(target=preload, daemon=True).start()


@app.on_event('shutdown')
async def shutdown():
    if bridge:
        bridge.shutdown()


async def refresh():
    snapshot = await asyncio.to_thread(bridge.state)
    snapshot['busy'] = bool(snapshot.get('busy') or (_job is not None and not _job.done()))
    # Local submission must not be hidden by a stale IDLE service snapshot.
    if snapshot['busy'] and snapshot.get('stage') == 'IDLE':
        snapshot.pop('stage', None)
    await update(**snapshot)
    return state


async def _run(task):
    action = task['action']
    try:
        if action == 'pick_place':
            kwargs = {'destination_color': task['destination_color']} if task.get('destination_color', 'box') != 'box' else {}
            result = await asyncio.to_thread(bridge.send_pick_place, task['target_color'], **kwargs)
        elif action == 'pick':
            result = await asyncio.to_thread(bridge.send_pick, task['target_color'])
        elif action == 'place':
            result = await asyncio.to_thread(bridge.place, task['destination_color'])
        elif action == 'sequence':
            result = await asyncio.to_thread(bridge.send_sequence, [item['target_color'] for item in task['tasks']])
        elif action == 'prepare' and task.get('observation_side'):
            result = await asyncio.to_thread(bridge.prepare, task['observation_side'])
        else:
            method = getattr(bridge, action)
            result = await asyncio.to_thread(method)
    except Exception as exc:
        result = {'success': False, 'message': str(exc)}
    snapshot = await asyncio.to_thread(bridge.state)
    ok = bool(result.get('success'))
    stage = 'DRY_RUN' if ok and bridge.dry_run else ('HOME' if ok and action == 'go_home' else
                                                  'WAITING_PLACE' if ok and snapshot.get('placement_ready') and snapshot.get('prepared') else
                                                  'READY' if ok and snapshot.get('prepared') else
                                                   ('SUCCESS' if ok else 'FAILED'))
    await update(**{**snapshot, 'busy': bool(snapshot.get('busy')), 'stage': stage, 'last_result': result})
    await broadcast({'type': 'result', 'ok': ok, **result, 'stage': stage,
                     'stage_zh': state['stage_zh'], 'parsed': task})


async def dispatch(task):
    global _job
    action = task.get('action')
    if action in ('help', 'status'):
        if action == 'help':
            message = ('先说“准备”；支持“夹起红色小方块”“把橘色积木放到白色方块上”'
                       '“先抓白块，然后放进盒子”“把三个方块都收进盒子里”'
                       '“按橙白红顺序放进盒子”“你拿着什么颜色”“看到了哪些方块”和“停止”。')
            payload = {'ok': True, 'type': 'info', 'message': message}
        else:
            snapshot = dict(await refresh())
            labels = {'red': '红块', 'white': '白块', 'orange': '橙块'}
            query = task.get('query', 'all')
            if query == 'holding':
                held = snapshot.get('holding')
                message = ('当前夹着' + labels.get(snapshot.get('holding_color'), '方块') if held else
                           '当前没有夹持方块' if held is False else '当前持物状态不可用，请检查服务连接')
            elif query == 'vision':
                vision = snapshot.get('vision', {})
                message = '；'.join(label + '：' + vision.get(color, {}).get('status_zh', '尚无定位')
                                   for color, label in labels.items())
            else:
                message = snapshot.get('stage_zh') or STAGE_ZH.get(snapshot.get('stage'), '状态未就绪')
            payload = {'ok': True, 'type': 'info', 'message': message, 'state': snapshot}
        await broadcast(payload)
        return payload
    if action == 'cancel':
        result = await asyncio.to_thread(bridge.cancel)
        payload = {'ok': bool(result['success']), 'type': 'info', 'message': result['message']}
        await broadcast(payload)
        return payload
    tasks = task.get('tasks')
    valid_sequence = (action == 'sequence' and isinstance(tasks, list) and 2 <= len(tasks) <= 3
                      and all(isinstance(item, dict) and item.get('action') == 'pick_place'
                              and item.get('target_color') in ('red', 'white', 'orange')
                              and item.get('destination_color', 'box') == 'box' for item in tasks)
                      and len({item['target_color'] for item in tasks}) == len(tasks))
    grasp_action = action in ('pick', 'place', 'pick_place', 'sequence')
    valid = valid_sequence or action in ('release', 'go_home') or (action == 'prepare' and
             task.get('observation_side', '') in ('', 'current', 'mirror')) or (action == 'pick_place' and
             task.get('target_color') in ('red', 'white', 'orange') and
             task.get('destination_color', 'box') in ('box', 'red', 'white', 'orange') and
             task.get('target_color') != task.get('destination_color', 'box')) or (
             action == 'pick' and task.get('target_color') in ('red','white','orange')) or (
             action == 'place' and task.get('destination_color') in ('box','red','white','orange'))
    if not valid:
        payload = {'ok': False, 'type': 'error', 'message': task.get('message') or '支持准备、归位和红白橙方块抓放；叠放目标必须是另一种颜色'}
        await broadcast(payload)
        return payload
    async with _submission_lock:
        snapshot = await refresh()
        if snapshot['busy']:
            payload = {'ok': False, 'type': 'error', 'message': '机械臂正在执行任务，请稍后'}
        elif not snapshot.get('server_ready'):
            payload = {'ok': False, 'type': 'error', 'message': '腕部抓放服务未就绪，请先启动抓放服务'}
        elif grasp_action and not snapshot.get('initialized', snapshot.get('prepared')):
            payload = {'ok': False, 'type': 'error', 'message': '请先说“准备”，确认到观察位后再抓取'}
        elif grasp_action and snapshot.get('can_accept') is False:
            payload = {'ok': False, 'type': 'error', 'message': '机械臂反馈暂不可用，请恢复反馈后直接重试'}
        elif action == 'place' and not snapshot.get('placement_ready'):
            payload = {'ok':False,'type':'error','message':'当前没有已定位夹持的方块，请先说“抓白色方块”等抓取指令'}
        elif action == 'place' and snapshot.get('holding_color') == task['destination_color']:
            payload = {'ok':False,'type':'error','message':'放置目标必须是盒子或另一种颜色的方块'}
        else:
            await update(busy=True, stage='ACCEPTED', parsed=task, last_result=None)
            _job = asyncio.create_task(_run(task))
            labels = {'prepare': '开始移动到观察位', 'release': '正在松开夹爪',
                      'go_home': '开始返回原始收起姿态', 'pick_place': '开始颜色抓取放置',
                      'pick': '开始抓取方块并返回观察位等待', 'place':'开始放置当前夹持的方块',
                      'sequence': '开始按颜色顺序逐块放进盒子'}
            payload = {'ok': True, 'type': 'info', 'message': labels[action], 'parsed': task}
        await broadcast(payload)
        return payload


async def handle_text(text):
    task = parse_wrist_command(text)
    await update(transcript=text, parsed=task)
    await broadcast({'type': 'transcript', 'text': text})
    await broadcast({'type': 'parsed', 'result': task})
    return await dispatch(task)


@app.get('/')
async def index():
    return FileResponse(static / 'index.html')


@app.get('/api/state')
async def api_state():
    return dict(await refresh())


@app.get('/api/health')
async def health():
    await refresh()
    return {'ok': True, 'asr': asr.name, 'asr_ready': asr.ready,
            'asr_error': getattr(asr, 'error', ''),
            'dry_run': bridge.dry_run, 'task_server': state['server_ready'], 'state': dict(state)}


@app.post('/api/parse')
async def api_parse(body: TextIn):
    return {'text': body.text, 'parsed': parse_wrist_command(body.text)}


@app.post('/api/command')
async def command(body: TextIn):
    result = await handle_text(body.text)
    return JSONResponse(result, status_code=200 if result['ok'] else 409)


@app.post('/api/task')
async def task(body: TaskIn):
    result = await dispatch(body.dict())
    return JSONResponse(result, status_code=200 if result['ok'] else 409)


@app.post('/api/cancel')
async def cancel():
    return await dispatch({'action': 'cancel'})


@app.post('/api/asr')
async def audio(audio: UploadFile = File(...)):
    data = await audio.read(8 * 1024 * 1024 + 1)
    if not data or len(data) > 8 * 1024 * 1024:
        return JSONResponse({'ok': False, 'message': '录音为空或超过大小限制'}, status_code=400)
    try:
        text = await asyncio.to_thread(asr.transcribe, data, audio.content_type or 'audio/webm', 'zh')
        return {'ok': True, 'text': text, 'asr': asr.name}
    except Exception as exc:
        return JSONResponse({'ok': False, 'message': f'识别失败：{exc}'}, status_code=500)


@app.websocket('/ws')
async def websocket(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    await ws.send_json({'type': 'state', **state})
    try:
        while True:
            message = await ws.receive_json()
            kind = message.get('type')
            if kind in ('text', 'text_command'):
                await handle_text(str(message.get('text') or ''))
            elif kind in ('task', 'manual_task'):
                await dispatch({'action': message.get('action', 'pick_place'),
                                'target_color': message.get('target_color'),
                                'destination_color': message.get('destination_color', 'box'),
                                'tasks': message.get('tasks', [])})
            elif kind == 'cancel':
                await dispatch({'action': 'cancel'})
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(ws)


def main(args=None):
    from rclpy.utilities import remove_ros_args
    parser = argparse.ArgumentParser(description='腕部颜色抓放：准备 → 抓指定颜色 → 放盒')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8001)
    parser.add_argument('--https-port', type=int, default=8444)
    parser.add_argument('--no-https', action='store_true')
    parser.add_argument('--legacy-ports', action='store_true',
                        help='Also serve HTTP 8000 and HTTPS 8443 for the existing Android app')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--terminal', action='store_true')
    # launch_ros appends --ros-args / node remaps to executable arguments.
    # Keep those for rclpy, but remove them before argparse handles UI options.
    argv = None if args is None else ['wrist_web_agent', *args]
    options = parser.parse_args(remove_ros_args(args=argv)[1:])
    if options.dry_run:
        os.environ['X5A_WEB_DRY_RUN'] = '1'
    if options.terminal:
        os.environ['X5A_WEB_TERMINAL'] = '1'
    import uvicorn
    print(f'腕部网页：http://192.168.0.50:{options.port}  先说“准备”，再说“抓红色”', flush=True)
    if not options.no_https:
        certs = Path(os.environ.get('X5A_WRIST_CERT_DIR', str(Path.home()/'.config/x5a_wrist/certs')))
        certs.mkdir(parents=True, exist_ok=True)
        cert, key = certs/'cert.pem', certs/'key.pem'
        if not cert.exists() or not key.exists():
            subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
                            '-keyout', str(key), '-out', str(cert), '-days', '365',
                            '-subj', '/CN=x5a-wrist', '-addext',
                            'subjectAltName=IP:192.168.0.50,IP:127.0.0.1,DNS:localhost'],
                           check=True, capture_output=True)
        # Both listeners use ONE app/event loop/bridge and one operation lock.
        # Only the HTTP listener manages app lifespan.
        async def serve():
            http = uvicorn.Server(uvicorn.Config(app, host=options.host, port=options.port, ws='wsproto'))
            https = uvicorn.Server(uvicorn.Config(app, host=options.host, port=options.https_port,
                                                ssl_certfile=str(cert), ssl_keyfile=str(key), lifespan='off', ws='wsproto'))
            servers = [http, https]
            if options.legacy_ports:
                if options.port != 8000:
                    servers.append(uvicorn.Server(uvicorn.Config(
                        app, host=options.host, port=8000, lifespan='off', ws='wsproto')))
                if options.https_port != 8443:
                    servers.append(uvicorn.Server(uvicorn.Config(
                        app, host=options.host, port=8443, lifespan='off',
                        ssl_certfile=str(cert), ssl_keyfile=str(key), ws='wsproto')))
            for server in servers:
                server.install_signal_handlers = lambda: None
            def stop():
                for server in servers:
                    server.should_exit = True
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, stop)
            await asyncio.gather(*(server.serve() for server in servers))
        print(f'手机按住说话：https://192.168.0.50:{options.https_port}', flush=True)
        if options.legacy_ports:
            print('原手机 App 兼容入口：http://192.168.0.50:8000 / https://192.168.0.50:8443', flush=True)
        asyncio.run(serve())
    else:
        uvicorn.run(app, host=options.host, port=options.port, ws='wsproto')
