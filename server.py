"""ブラウザからプレイできるようにする実験用サーバー(ローカル動作確認版)。

既存の `play_main.py` のゲーム構築処理をそのまま再利用し、pygame の
入出力だけを WebSocket に差し替える。

    起動 = python agent/agent/play_main.py --agent0 CSP --agent1 human \
           --map ring --deadline 0 --sc_2agent --order experiment1 \
           --instruction_request_timing free

と同じ構成が既定で立ち上がる。play_main.py と同じオプション名を受け付けるので、
実験条件を変えたいときは同じ書き方で指定できる。

仕組み:
  - SDL_VIDEODRIVER=dummy にして pygame をウィンドウなしで動かす。
    描画・指示パネル・イベント処理は一切変更せず、そのまま動く。
  - 描画結果(display Surface)を PNG にして WebSocket でブラウザへ送る。
    ring マップは 320x400px しかないため、10Hz でも約 1Mbps で収まる。
  - ブラウザのキー入力/クリックを pygame のイベントに変換して post する。
    GamePlay._run_human の pygame.event.get() がそのまま拾ってくれる。

この方式は「ローカル版と見た目・挙動が完全に同じ」ことが保証できるのが利点。
状態JSONを送ってブラウザ側で描画し直す方式(約27kbps)へは後から移行できる。
"""
import os

# pygame を import する前に設定しないと効かない。
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
# ウィンドウが無いので音も要らない。環境によっては初期化で数秒待たされる。
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

import argparse
import asyncio
import io
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import pygame
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

ROOT = Path(__file__).resolve().parent
# pip install -e されていない環境でも動くように、パッケージの場所を通しておく。
for extra in (ROOT / 'agent', ROOT / 'testbed-cooking'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from agent import play_main  # noqa: E402
from agent.gameplay import (  # noqa: E402
    INSTRUCTION_TIMINGS,
    INSTRUCTION_TIMING_FREE,
)
from gym_cooking.utils.order_preset import preset_names  # noqa: E402

WEB_DIR = ROOT / 'web'

# スレッド間で GIL を渡す間隔(既定 5ms)。ローカル版と違い Web 版は
# 環境・AI・配信・エンコードが同じプロセスで同時に動くため、既定のままだと
# 1度 GIL を握ったスレッドが最大 5ms 手放さず、10Hz で回りたい環境スレッドの
# 周期が伸びる。短くすると切り替えが増える代わりに、待たされる最大時間が縮む。
sys.setswitchinterval(0.001)

# ブラウザのキー名 -> pygame のキー定数。
# 矢印キーは移動/インタラクト、Space は指示パネル、Escape はパネルのキャンセル。
KEY_MAP = {
    'ArrowUp': pygame.K_UP,
    'ArrowDown': pygame.K_DOWN,
    'ArrowLeft': pygame.K_LEFT,
    'ArrowRight': pygame.K_RIGHT,
    'Space': pygame.K_SPACE,
    'Escape': pygame.K_ESCAPE,
}



class RemoteMouse:
    """ブラウザから送られてきたカーソル位置を pygame へ橋渡しする。

    SDL の dummy ドライバは実際のカーソルを持たないため、
    pygame.mouse.get_pos() が常に (0, 0) を返す。指示パネル
    (InstructionPanel.run)はホバー中のカードを get_pos() で判定しているので、
    そのままだとどのカードを選ぼうとしているかが分からない。
    get_pos() だけを差し替えて、ブラウザ側の座標を返すようにする。
    """

    def __init__(self):
        self._pos = (0, 0)
        self._original = pygame.mouse.get_pos
        pygame.mouse.get_pos = self.get_pos

    def get_pos(self):
        return self._pos

    def set_pos(self, x, y):
        self._pos = (int(x), int(y))


class WebGamePlay:
    """GamePlay をヘッドレスで動かし、画面と入力を WebSocket につなぐ。"""

    def __init__(self, args):
        self.args = args
        self.game = None
        self.env = None
        self.replay = None
        self.mouse = None

        # 描画完了した1枚をそのまま抱えておく退避先(描画スレッドが書く)。
        self._pending_surface = None
        self._pending_lock = threading.Lock()
        # 新しい1枚が置かれたことを配信側へ知らせる。
        self._frame_ready = threading.Event()

        # 最新フレーム。capture 側(executor スレッド)が書き、送信側が読む。
        self._frame = None            # PNG バイト列
        self._frame_version = 0
        self._frame_size = (0, 0)
        self._frame_lock = threading.Lock()

        # 最初のブラウザ接続を待ってからゲームを開始する。
        # 誰も見ていない間に注文の時間が進んでしまうのを防ぐ。
        self.client_connected = threading.Event()

        # 配信経路のどこでコマが落ちているかを見るための計数。/api/perf で読む。
        self.perf = {'rendered': 0, 'encoded': 0, 'sent': 0, 'started': time.time()}
        self.state = 'waiting'        # waiting -> running -> finished
        self.result = None

    # ------------------------------------------------------------------
    # 構築と実行
    # ------------------------------------------------------------------
    def build(self):
        """play_main と同じ手順で env / AI / GamePlay を組み立てる。"""
        a = self.args

        # init_env_replay はモジュールグローバルの arglist を参照している
        # (sc_2agent / deadline)。CLI と同じ値を渡すために差し込む。
        play_main.arglist = argparse.Namespace(
            sc_2agent=a.sc_2agent,
            deadline=a.deadline,
        )

        self.game, self.env, self.replay = play_main.init_env_replay(
            a.map, a.agent0, a.agent1, a.task,
            a.no_reschedule, a.debug,
            a.orders, a.order_seed,
            a.instruction_request_timing,
        )
        return self.game

    def run_forever(self):
        """メインスレッドで呼ぶ。ブラウザ接続を待ってからゲーム本体を回す。"""
        print('[server] ブラウザからの接続を待っています...')
        self.client_connected.wait()

        self.state = 'running'
        self.perf.update(rendered=0, encoded=0, sent=0, started=time.time())
        print('[server] ゲームを開始します')
        try:
            success = self.game.on_execute()
        finally:
            self.state = 'finished'
            self._save_replay()

        order = self.env.order_scheduler
        self.result = {
            'success': bool(success),
            'served': getattr(order, 'successful_orders', 0),
            'failed': getattr(order, 'failed_orders', 0),
            'reward': getattr(order, 'reward', 0),
        }
        print(f'[server] ゲーム終了: {self.result}')

    def on_init_done(self):
        """pygame の初期化後に呼ぶ(display が出来てから差し替える)。"""
        self.mouse = RemoteMouse()
        self._install_frame_hook()

    def _install_frame_hook(self):
        """描画が完了した瞬間だけフレームを取り込むようにする。

        display Surface を別スレッドから好きなタイミングで読むと、
        on_render の「screen.fill -> 各オブジェクトを順に描く」の途中を
        掴んでしまい、カウンターやプレイヤーが抜けたコマが混ざる
        (毎フレーム何かが消えて見える原因)。

        on_render も指示パネルも、1枚を描き終えた最後に必ず
        pygame.display.flip() を呼ぶ。そこへ割り込んで、描画したスレッド
        自身に完成品を複製させれば、常に整合の取れた1枚だけが手に入る。
        """
        original_flip = pygame.display.flip

        def flip(*args, **kwargs):
            original_flip(*args, **kwargs)
            surface = pygame.display.get_surface()
            if surface is None:
                return
            try:
                # 複製しておけば、この後 PNG 化している間に次の描画が
                # 始まっても壊れない。320x400 の複製は数十マイクロ秒。
                snapshot = surface.copy()
            except pygame.error:
                return
            with self._pending_lock:
                self._pending_surface = snapshot
            self.perf['rendered'] += 1
            self._frame_ready.set()

        pygame.display.flip = flip

    def _save_replay(self):
        a = self.args
        repdir = ROOT / 'agent' / 'agent' / 'replay'
        repdir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = repdir / f'web-{a.map}-{a.agent0}-{a.agent1}-{stamp}.rep'
        try:
            self.replay.save(path)
            print(f'[server] リプレイを保存しました: {path}')
        except Exception as e:
            print(f'[server] リプレイ保存に失敗: {e}')

    # ------------------------------------------------------------------
    # 画面キャプチャ
    # ------------------------------------------------------------------
    def capture(self):
        """描画完了済みのフレームを PNG にする。新しい絵が無ければ None。

        executor スレッドから呼ばれる(PNG エンコードに約3msかかるため、
        イベントループを塞がないようにする)。
        """
        with self._pending_lock:
            surface = self._pending_surface
            self._pending_surface = None
        if surface is None:
            return None

        buf = io.BytesIO()
        try:
            pygame.image.save(surface, buf, 'png')
        except Exception:
            return None
        data = buf.getvalue()

        with self._frame_lock:
            if data == self._frame:
                return None
            self._frame = data
            self._frame_version += 1
            self._frame_size = surface.get_size()
            self.perf['encoded'] += 1
            return self._frame_version

    def wait_and_capture(self, timeout=0.5):
        """次の1枚が描かれるまで待ってから PNG にする。

        以前は一定周期(15Hz)で見に行っていたが、描画は 10Hz なので
        「描かれてから見に行くまで」に平均33ms・最悪67msの待ちが乗っていた。
        描画完了を待って即座に送れば、この待ちがまるごと無くなる。

        executor スレッドから呼ぶ。待っている間は GIL を手放すので、
        ゲーム側のスレッドを邪魔しない。
        """
        if not self._frame_ready.wait(timeout):
            return None
        self._frame_ready.clear()
        return self.capture()

    def latest_frame(self):
        with self._frame_lock:
            return self._frame_version, self._frame, self._frame_size

    # ------------------------------------------------------------------
    # 入力
    # ------------------------------------------------------------------
    def post_key(self, code):
        key = KEY_MAP.get(code)
        if key is None:
            return False
        pygame.event.post(pygame.event.Event(
            pygame.KEYDOWN, key=key, mod=0, unicode='', scancode=0))
        return True

    def post_mouse_move(self, x, y):
        if self.mouse is not None:
            self.mouse.set_pos(x, y)
        pygame.event.post(pygame.event.Event(
            pygame.MOUSEMOTION, pos=(int(x), int(y)), rel=(0, 0), buttons=(0, 0, 0)))

    def post_mouse_down(self, x, y):
        if self.mouse is not None:
            self.mouse.set_pos(x, y)
        pygame.event.post(pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, button=1, pos=(int(x), int(y))))
        pygame.event.post(pygame.event.Event(
            pygame.MOUSEBUTTONUP, button=1, pos=(int(x), int(y))))


# ----------------------------------------------------------------------
# FastAPI
# ----------------------------------------------------------------------
session: WebGamePlay | None = None
app = FastAPI(title='Overcooked CSP Web')


@app.get('/')
async def index():
    return FileResponse(WEB_DIR / 'index.html')


@app.get('/api/config')
async def config():
    a = session.args
    return JSONResponse({
        'map': a.map,
        'agent0': a.agent0,
        'agent1': a.agent1,
        'sc_2agent': a.sc_2agent,
        'orders': a.orders,
        'deadline': a.deadline,
        'instruction_request_timing': a.instruction_request_timing,
        'state': session.state,
        'result': session.result,
    })


@app.get('/api/state')
async def state():
    """動作確認用。入力がちゃんと環境へ届いているかをここで見る。

    将来ブラウザ側で描画する方式へ移すときは、この内容をそのまま
    WebSocket で流せばよい(実測 gzip 335B/フレーム)。
    """
    env = session.env
    if env is None:
        return JSONResponse({'state': session.state})

    order = env.order_scheduler
    return JSONResponse({
        'state': session.state,
        'time': round(getattr(env, 'current_time', 0.0), 2),
        'human_idx': session.game.idx_human,
        'agents': [
            {
                'idx': i,
                'name': a.name,
                'pos': list(a.location),
                'holding': getattr(getattr(a, 'holding', None), 'full_name', None),
            }
            for i, a in enumerate(env.sim_agents)
        ],
        # current_orders は (goal_obj, restTime, timeLimit, bonus) のタプル。
        'orders': [
            {'name': goal.full_name, 'rest': round(rest, 1)}
            for goal, rest, _limit, _bonus in getattr(order, 'current_orders', [])
        ],
        'served': getattr(order, 'successful_orders', 0),
        'failed': getattr(order, 'failed_orders', 0),
        'reward': getattr(order, 'reward', 0),
    })


@app.get('/api/perf')
async def perf():
    """どこが遅いかを切り分けるための実測値。

    env_period_ms が 1/fps(=100ms)より大きいのに env_work_ms が小さければ、
    処理が重いのではなく、他スレッドに邪魔されて環境が回れていない。
    """
    p = dict(session.perf)
    elapsed = max(1e-6, time.time() - p.pop('started'))
    stats = getattr(session.game, 'loop_stats', {}) or {}
    return JSONResponse({
        'elapsed_s': round(elapsed, 1),
        'rendered_fps': round(p['rendered'] / elapsed, 2),
        'encoded_fps': round(p['encoded'] / elapsed, 2),
        'sent_fps': round(p['sent'] / elapsed, 2),
        'env_work_ms': round(stats.get('work_s', 0.0) * 1000, 1),
        'env_period_ms': round(stats.get('period_s', 0.0) * 1000, 1),
        'env_target_ms': round(1000 / max(session.game.fps, 1), 1),
        'cpu_count': os.cpu_count(),
        **p,
    })


@app.websocket('/ws')
async def ws(sock: WebSocket):
    await sock.accept()
    session.client_connected.set()

    loop = asyncio.get_running_loop()
    sent_version = 0
    sent_size = None
    last_state = None

    async def pump_input():
        """ブラウザからの入力を pygame イベントへ流し続ける。"""
        while True:
            raw = await sock.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            kind = msg.get('type')
            if kind == 'key':
                session.post_key(msg.get('code'))
            elif kind == 'mousemove':
                session.post_mouse_move(msg.get('x', 0), msg.get('y', 0))
            elif kind == 'mousedown':
                session.post_mouse_down(msg.get('x', 0), msg.get('y', 0))
            elif kind == 'ping':
                # RTT 計測用。クライアントの送信時刻をそのまま返す。
                await sock.send_text(json.dumps({'type': 'pong', 't': msg.get('t')}))

    task = asyncio.create_task(pump_input())
    try:
        while True:
            # 次の描画を待って PNG 化する。待ちもエンコードも executor 側なので
            # イベントループは塞がらず、入力(ping/キー)は待たされない。
            await loop.run_in_executor(None, session.wait_and_capture)
            version, data, size = session.latest_frame()

            if size != sent_size and size != (0, 0):
                sent_size = size
                # base_w/base_h はゲーム画面そのものの大きさ。指示パネルを
                # 出すと display はこれより横に広がるが、左側 base_w ぶんは
                # 常にゲーム画面なので、クライアントはそこだけを切り出して
                # 位置を動かさずに描き続けられる。
                await sock.send_text(json.dumps(
                    {'type': 'meta', 'w': size[0], 'h': size[1],
                     'base_w': session.game.width, 'base_h': session.game.height}))

            if data is not None and version != sent_version:
                sent_version = version
                await sock.send_bytes(data)
                session.perf['sent'] += 1

            if session.state != last_state:
                last_state = session.state
                await sock.send_text(json.dumps(
                    {'type': 'status', 'state': session.state,
                     'result': session.result}))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        task.cancel()


def start_server_thread(host, port):
    """uvicorn を別スレッドで動かす。メインスレッドは pygame に使う。"""
    import uvicorn

    config = uvicorn.Config(app, host=host, port=port, log_level='warning',
                            ws_ping_interval=None)
    server = uvicorn.Server(config)
    # メインスレッド以外ではシグナルハンドラを登録できない。
    server.install_signal_handlers = False

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # 起動を待ってから URL を出す(押しても繋がらない案内を出さないため)。
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    return server


def parse_arguments():
    """play_main.py と同じオプション名を受け付ける。

    既定値は実験で使う下記のコマンドと同じ構成:
      --agent0 CSP --agent1 human --map ring --deadline 0 --sc_2agent
      --order experiment1 --instruction_request_timing free
    """
    p = argparse.ArgumentParser('Overcooked CSP web server')

    p.add_argument('--host', type=str, default='127.0.0.1',
                   help='待ち受けアドレス。LAN の別端末から繋ぐなら 0.0.0.0')
    p.add_argument('--port', type=int, default=8000)

    p.add_argument('--map', type=str, default='ring',
                   choices=['ring', 'bottleneck', 'partition', 'quick'])
    agents = ['human', 'HLA', 'SMOA', 'FMOA', 'NEA', 'Random',
              'TSPSolver', 'Greedy', 'CSP', 'Task', 'choponly']
    p.add_argument('--agent0', type=str, default='CSP', choices=agents)
    p.add_argument('--agent1', type=str, default='human', choices=agents)
    p.add_argument('--task', type=str, default=None)
    p.add_argument('--no_reschedule', action='store_true')
    p.add_argument('--sc_2agent', action=argparse.BooleanOptionalAction, default=True,
                   help='2エージェント向けスケジューリング(既定で有効。切るなら --no-sc_2agent)')
    p.add_argument('--debug', action='store_true')
    p.add_argument('--orders', '--order', dest='orders', type=str, default='experiment1',
                   help=f'注文プリセット名({", ".join(preset_names())})か注文ファイル名')
    p.add_argument('--order-seed', type=int, default=None)
    p.add_argument('--instruction_request_timing', type=str,
                   default=INSTRUCTION_TIMING_FREE, choices=list(INSTRUCTION_TIMINGS))
    p.add_argument('--deadline', type=float, default=0.0)

    return p.parse_args()


def main():
    global session

    args = parse_arguments()
    session = WebGamePlay(args)
    session.build()

    # GamePlay.on_init のあとに pygame.mouse を差し替えたいので、フックしておく。
    original_on_init = session.game.on_init

    def on_init():
        ret = original_on_init()
        session.on_init_done()
        return ret

    session.game.on_init = on_init

    start_server_thread(args.host, args.port)

    shown_host = 'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host
    print('=' * 60)
    print(f'  Overcooked CSP web server')
    print(f'  URL: http://{shown_host}:{args.port}/')
    print(f'  構成: map={args.map} agent0={args.agent0} agent1={args.agent1} '
          f'sc_2agent={args.sc_2agent} orders={args.orders} '
          f'deadline={args.deadline} timing={args.instruction_request_timing}')
    print('=' * 60)

    try:
        session.run_forever()
    except KeyboardInterrupt:
        print('\n[server] 中断しました')
        return

    # 終了後もアンケートへの導線を出せるよう、サーバーは動かしたままにする。
    print('[server] 終了しました。Ctrl+C でサーバーを止められます。')
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
