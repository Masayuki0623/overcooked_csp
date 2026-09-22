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
  - 画面は、pygame が描くときに使った命令(どの絵をどこに描くか)だけを
    WebSocket で送り、ブラウザで同じ順に描き直す。同じ命令には番号を振り、
    前のコマと同じ部分は送らないので、1コマ約 200 バイト(約 15kbps)。
    画像で送ると 1コマ約 9KB(約 700kbps)で、Tailscale Funnel 経由では
    送る量がそのまま操作の遅れになっていた(往復 137ms、家の中で直接
    つなぐと 5ms)。
    指示パネルは命令を残さないので、開いている間だけ画像で送る。
    ?mode=png で開くと、従来どおり常に画像で送る。
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
import csv
from copy import deepcopy
import io
import json
import random
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import pygame
try:
    from PIL import Image
except ImportError:      # 無くても動く(画面が少し重くなるだけ)
    Image = None
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
# pip install -e されていない環境でも動くように、パッケージの場所を通しておく。
for extra in (ROOT / 'agent', ROOT / 'testbed-cooking'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from agent import play_main  # noqa: E402
from agent.gameplay import (  # noqa: E402
    INSTRUCTION_TIMINGS,
    INSTRUCTION_TIMING_FREE,
    INSTRUCTION_TIMING_NO_INSTRUCTION,
    INSTRUCTION_TIMING_ONCE_AT_START,
)
from agent.instruction_panel import (  # noqa: E402
    card_action, card_icon_name, card_label)
from gym_cooking.utils import config as game_config  # noqa: E402
from gym_cooking.utils.order_preset import (  # noqa: E402
    enumerate_order_recipes, experiment_case_indices, preset_names)

WEB_DIR = ROOT / 'web'
# ゲームの絵(pygame が使うのと同じ PNG)。ブラウザ側で描くときに読み込む。
GRAPHICS_DIR = ROOT / 'testbed-cooking' / 'gym_cooking' / 'misc' / 'game' / 'graphics'
# ブラウザに配る縮小版の置き場。元の絵は最大 2475px 四方で合計約 13MB あり、
# スマホで毎回落とすには重すぎる。画面では 1マス 40px で描くので、
# その2倍(80px)あれば見た目は変わらない。
WEB_GRAPHICS_DIR = ROOT / '.cache' / 'web_graphics'
WEB_SPRITE_PX = 80


def prepare_web_graphics():
    """ブラウザに配る縮小版の絵を作る(元の絵より新しければ作り直さない)。"""
    WEB_GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)
    for src in GRAPHICS_DIR.glob('*.png'):
        dst = WEB_GRAPHICS_DIR / src.name
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            continue
        if Image is None:
            dst.write_bytes(src.read_bytes())
            continue
        with Image.open(src) as im:
            im = im.convert('RGBA')
            w, h = im.size
            k = min(1.0, WEB_SPRITE_PX / max(w, h))
            if k < 1.0:
                im = im.resize((max(1, round(w * k)), max(1, round(h * k))), Image.LANCZOS)
            im.save(dst, 'PNG', optimize=True)


prepare_web_graphics()

# 遊ぶ前に選べる地図とレシピ。地図の中身は play_test.MAP_SETTINGS を参照。
MAP_CHOICES = [
    ('exp_partition', '仕切り', '左右が仕切られていて行き来できない。材料は仕切りの台で受け渡す'),
    ('exp_bottleneck', 'ボトルネック', '仕切りの真ん中に1マスだけ通れる穴がある'),
    ('exp_ring', 'リング', '真ん中の島のまわりをぐるっと回れる'),
]
RECIPE_CHOICES = [
    ('experiment1', '野菜のみ', 'サラダ2品 + スープ1品'),
    ('experiment2', '野菜 + フルーツ', 'サラダ + スープ + ジュース'),
]
_JP_FOOD = {'Lettuce': 'レタス', 'Onion': '玉ねぎ', 'Tomato': 'トマト',
            'Apple': 'リンゴ', 'Orange': 'オレンジ', 'Banana': 'バナナ'}
_JP_DISH = {'Salad': 'サラダ', 'Soup': 'スープ', 'Juice': 'ジュース'}


def uses_fruit(recipes):
    """注文の中にフルーツを使うもの(ジュース)があるか。"""
    return any(f in r for r in recipes for f in ('Apple', 'Orange', 'Banana', 'Juice'))


def recipe_label(name):
    """'OnionLettuceSoup' -> '玉ねぎ・レタスのスープ'。"""
    words = re.findall(r'[A-Z][a-z]*', name)
    dish = _JP_DISH.get(words[-1], words[-1]) if words else name
    if words and words[0] == 'Full':
        foods = 'レタス・玉ねぎ・トマト'
    else:
        foods = '・'.join(_JP_FOOD.get(w, w) for w in words[:-1])
    return f'{foods}の{dish}'


def order_sets_for(preset):
    return enumerate_order_recipes(preset)


# ----------------------------------------------------------------------
# 実験(参加者ID を入れて遊ぶとき)
# ----------------------------------------------------------------------
# 条件は「地図3種 × AI が指示を後回しにできる量(skip_budget)3種」の9通り。
# 参加者は9セッション全部を遊び、順番だけ人ごとにランダムにする。
SKIP_BUDGETS = (0, 2, 4)
EXPERIMENT_PRESET = 'experiment2'
ASSIGN_PATH = ROOT / 'results' / 'assignments.json'
SURVEY_PATH = ROOT / 'results' / 'survey.csv'
SESSION_LOG_PATH = ROOT / 'results' / 'web_sessions.csv'
_assign_lock = threading.Lock()


def all_conditions():
    return [{'map': m, 'skip_budget': b} for m, _, _ in MAP_CHOICES for b in SKIP_BUDGETS]


class CrossProcessLock:
    """プロセスをまたいで1つずつにする鍵。

    4人同時に遊べるようにすると、記録のファイルを別々のプロセスが同時に
    書く。スレッドの鍵(threading.Lock)は同じプロセスの中でしか効かないので、
    OS の鍵を使う。プロセスが落ちても OS が外してくれるため、鍵が残ったまま
    次の人が始められなくなることがない。
    """

    def __init__(self, path):
        self.path = Path(str(path) + '.lock')
        self._fh = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, 'a+b')
        try:
            if os.name == 'nt':
                import msvcrt
                # msvcrt は 10 秒あきらめると例外を出すので、取れるまで繰り返す。
                while True:
                    self._fh.seek(0)
                    try:
                        msvcrt.locking(self._fh.fileno(), msvcrt.LK_LOCK, 1)
                        break
                    except OSError:
                        time.sleep(0.05)
            else:
                import fcntl
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
        except Exception:
            # 鍵が使えない環境でも、記録を落とすより書いたほうがよい。
            pass
        return self

    def __exit__(self, *exc):
        try:
            if os.name == 'nt':
                import msvcrt
                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        finally:
            self._fh.close()
            self._fh = None
        return False


def _load_assignments():
    try:
        return json.loads(ASSIGN_PATH.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _save_assignments(data):
    ASSIGN_PATH.parent.mkdir(parents=True, exist_ok=True)
    ASSIGN_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')


def assignment_for(participant):
    """その参加者の条件の並び(9通り)と、次が何セッション目かを返す。

    並びは最初に作ったときの1回だけ決め、ファイルに残す。途中でサーバーを
    止めても同じ順番の続きから遊べる。
    """
    with _assign_lock, CrossProcessLock(ASSIGN_PATH):
        data = _load_assignments()
        rec = data.get(participant)
        if not rec or len(rec.get('order') or []) != len(all_conditions()):
            order = all_conditions()
            random.shuffle(order)
            rec = {'order': order, 'done': 0, 'created': datetime.now().isoformat(timespec='seconds')}
            data[participant] = rec
            _save_assignments(data)
        return rec


def note_session_done(participant):
    """1セッション終わったので、次の条件へ進める。"""
    with _assign_lock, CrossProcessLock(ASSIGN_PATH):
        data = _load_assignments()
        rec = data.get(participant)
        if not rec:
            return
        rec['done'] = min(len(rec['order']), int(rec.get('done', 0)) + 1)
        _save_assignments(data)


def append_csv(path, fields, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with CrossProcessLock(path):
        _append_csv_locked(path, fields, row)


def _append_csv_locked(path, fields, row):
    new = not path.exists()
    if not new:
        # 項目が増えたのに古い見出しのまま足すと、列がずれて読めなくなる。
        # 見出しが変わっていたら、古いファイルは名前を変えて残す。
        try:
            with path.open('r', encoding='utf-8', newline='') as f:
                head = next(csv.reader(f), [])
        except OSError:
            head = []
        if head and head != list(fields):
            backup = path.with_name(
                f'{path.stem}-{datetime.now():%Y%m%d_%H%M%S}{path.suffix}')
            path.rename(backup)
            print(f'[server] 記録の項目が変わったので、古い分は {backup.name} に移しました')
            new = True
    with path.open('a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        if new:
            w.writeheader()
        w.writerow(row)


# 枠を取ってからスタートを押すまでの猶予(秒)。
START_TIMEOUT_S = 120
# 操作している人の端末から、これだけ何も届かなければ居なくなったとみなす(秒)。
# ブラウザは1秒ごとに ping を送る。スマホは電波が切れても「閉じた」とは
# 知らせてこないので、待っているだけでは枠が永久に空かない。
PLAYER_SILENCE_TIMEOUT_S = 10
# 1回の送信を待つ上限(秒)。これを超えたら相手は居ないとみなす。
SEND_TIMEOUT_S = 5
# 返事待ちで送れる盤面の数の下限と上限。往復時間に合わせて、この間で決める。
# 少なすぎると、往復の遅い回線では「1往復に数コマ」しか送れず盤面がカクつく
# (実測: 往復 1秒のスマホで毎秒3コマ)。多すぎると盤面が溜まって遅れる。
MIN_UNACKED_FRAMES = 3
MAX_UNACKED_FRAMES_CAP = 8
# 端末が「描いた」と返事をしていない盤面が、これだけ溜まっていたら次は送らない。
# 送り続けると、端末や途中の経路に盤面が溜まり、その分だけ操作の応答が遅れる
# (実測: スマホで往復 900ms、送れたのは描いた分の4割)。追いつけないときは
# 途中を飛ばして、いつも最新の盤面だけを送る。
MAX_UNACKED_FRAMES = 3


def unacked_limit(rtt_ms):
    """往復時間から、返事待ちで送ってよい盤面の数を決める。

    盤面は 0.1 秒ごとなので、往復時間のぶんだけ「途中にある」状態が
    ふつう。窓をそれに合わせると、遅い回線でも毎秒のコマ数が落ちない。
    """
    if not rtt_ms:
        return MAX_UNACKED_FRAMES
    return max(MIN_UNACKED_FRAMES,
               min(MAX_UNACKED_FRAMES_CAP, int(rtt_ms / 100) + 1))

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



def encode_frame(surface):
    """1枚の画面を PNG にする。色を 256 色に減らしてから圧縮する。

    ドット絵なので色数が少なく(実測 約1000色)、256色に落としても見た目は
    ほぼ変わらない。そのまま PNG にすると 1枚 約19KB(10fps で 1.6Mbps)、
    減色すると 約9KB(0.7Mbps)で、かかる時間はどちらも数ms。スマホの
    回線では送る量がそのまま遅れになるので、軽い方を送る。
    Pillow が無い環境では、従来どおり pygame でそのまま保存する。
    """
    buf = io.BytesIO()
    try:
        if Image is not None:
            w, h = surface.get_size()
            img = Image.frombytes('RGB', (w, h), pygame.image.tobytes(surface, 'RGB'))
            img = img.quantize(colors=256, method=Image.Quantize.FASTOCTREE)
            img.save(buf, 'PNG')
        else:
            pygame.image.save(surface, buf, 'png')
    except Exception:
        return None
    return buf.getvalue()


def _num(v):
    return int(round(float(v)))


def _rgb(c):
    return [_num(c[0]), _num(c[1]), _num(c[2])]


def normalize_elements(elements):
    """pygame が1枚を描くのに使った命令を、JSON で送れる短い形にする。

    Game.on_render は描いたものを順に記録している(get_visualization)。
    画像そのものではなくこの命令列を送り、ブラウザで同じ絵を同じ順に
    描き直せば、見た目はローカル版とまったく同じになる。
      ['F', r, g, b]                   塗りつぶし
      ['R', r, g, b, x, y, w, h, 線幅]  四角(線幅 0 は塗り)
      ['I', 絵の名前, x, y, w, h]        画像
      ['T', 文字, r, g, b, x, y, 大きさ] 文字
    """
    out = []
    for kind, a in elements:
        if kind == 'Fill':
            out.append(['F'] + _rgb(a['color']))
        elif kind == 'Rect':
            x, y, w, h = a['box']
            out.append(['R'] + _rgb(a['color'])
                       + [_num(x), _num(y), _num(w), _num(h), _num(a.get('width', 0))])
        elif kind == 'Image':
            x, y = a['location']
            w, h = a['size']
            out.append(['I', a['path'], _num(x), _num(y), _num(w), _num(h)])
        elif kind == 'Text':
            x, y = a['location']
            out.append(['T', str(a['text'])] + _rgb(a['color'])
                       + [_num(x), _num(y), _num(a.get('px', 12))])
    return out


class DrawEncoder:
    """描画命令の列を、前のコマとの差分にして送る(接続ごとに1つ持つ)。

    同じ命令(同じ場所の同じ絵)は何度も出てくるので、初めて出たときだけ
    中身を送って番号を振り、以後は番号だけを送る。さらに、前のコマと先頭
    から一致している部分(床やカウンターなど、ほとんど動かない背景)は
    「何個目まで同じ」とだけ伝える。1コマあたり数百バイトに収まる。
    """

    RESET_AT = 50000   # 番号表がこれより大きくなったら作り直す

    def __init__(self):
        self.ids = {}
        self.prev = []

    def encode(self, elements):
        reset = False
        if len(self.ids) > self.RESET_AT:
            self.ids, self.prev, reset = {}, [], True
        defs = {}
        seq = []
        for e in elements:
            key = json.dumps(e, ensure_ascii=False, separators=(',', ':'))
            i = self.ids.get(key)
            if i is None:
                i = len(self.ids)
                self.ids[key] = i
                defs[i] = e
            seq.append(i)
        keep = 0
        for a, b in zip(self.prev, seq):
            if a != b:
                break
            keep += 1
        self.prev = seq
        msg = {'type': 'draw', 'defs': defs, 'keep': keep, 'tail': seq[keep:]}
        if reset:
            msg['reset'] = True
        return msg


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

        # 最新の描画命令(ブラウザ側で描くとき用)。指示パネルを出している
        # 間は pygame が別の方法で描くので None にして、画像で送る。
        self._draw = None
        self._draw_me = None
        self._draw_version = 0
        # pygame のフォント -> 文字の大きさ(px)。描画命令には大きさが
        # 残らないので、put_text を包んで書き足す。
        self._font_px = {}

        # 最初のブラウザ接続を待ってからゲームを開始する。
        # 誰も見ていない間に注文の時間が進んでしまうのを防ぐ。
        self.client_connected = threading.Event()

        # 操作できるのは1人だけ。後から来た人には「別の人がプレイ中」と
        # 残り時間を見せて待ってもらう。player はいま操作している接続の目印。
        self.player = None
        self._player_lock = threading.Lock()
        # 何回目のゲームか。1回終わるごとに次のゲームを用意し直す。
        # 終わった回の結果は、その回の参加者へ届けるために番号で残す。
        self.game_id = 0
        self.results = {}
        # 遊ぶ人がスタート画面で選んだ地図・レシピ・注文の組み合わせ。
        # 選ばれた内容でゲームを組み立ててから始める。
        self.selection = None
        self._hooks_installed = False

        # 指示の選択(ブラウザ側に出す)
        self.instruction_natural_rank = None
        self.instruction_kinds = ''
        self.instruction_request = None
        self._instruction_answer = None
        self._instruction_seq = 0
        self._instruction_lock = threading.Lock()
        self._instruction_done = threading.Event()
        self._me_range = None          # 人のキャラを描いた命令の範囲
        self._other_range = None       # 相手(AI)のキャラの範囲
        self._me_mark = None           # 人の「向いている先」の枠の位置
        self._mark_index = None
        self.timeline = []             # 1秒ごとの通信の様子(ゲーム後にファイルへ)
        self.connection_info = None
        self.disconnect_reason = None
        # 配信経路のどこでコマが落ちているかを見るための計数。/api/perf で読む。
        self.perf = {'rendered': 0, 'encoded': 0, 'sent': 0, 'started': time.time(),
                     'client_rtt_ms': []}
        # preparing -> waiting -> running -> finished -> preparing ...
        self.state = 'preparing'
        self.result = None

        # ブラウザへ出す短いお知らせ(「いま指示できる作業はありません」等)。
        # ゲーム側のスレッドが積み、WebSocket 側が取り出して送る。
        self._notices = []
        self._notices_lock = threading.Lock()

    def try_acquire(self, token):
        """空いていれば、この接続を操作する人にする。

        枠を取っただけではゲームは始めない。URL を開いた瞬間に始まると、
        画面を見る前に時間が進んでしまう。スタートボタンで start() を呼ぶ。
        """
        with self._player_lock:
            if self.player is None and self.state == 'waiting':
                self.player = token
                return True
            return False

    def start(self, token, choice=None):
        """操作する人がスタートを押した。選んだステージで組み立てて始める。"""
        with self._player_lock:
            if self.player is not token or self.state != 'waiting':
                return
            self.selection = self._resolve_choice(choice or {})
            self.client_connected.set()

    def go(self, token):
        """カウントダウンが終わった。時間を進め始める。"""
        with self._player_lock:
            if self.player is not token or self.state != 'ready' or self.game is None:
                return
            if getattr(self, '_released', False):
                return
            self._released = True
            self.game._q_env.put(('Continue', {}))
            self.state = 'running'
            print(f'[server] #{self.game_id} ゲームを開始します')

    def _resolve_choice(self, choice):
        """画面で選ばれた内容を、組み立てに使える形にする。おかしな値は既定に戻す。

        参加者ID が入っているときは実験のセッション。地図とレシピは選ばせず、
        その参加者に割り当てた順番どおりの条件で遊ぶ。
        """
        participant = str(choice.get('participant') or '').strip()
        if participant:
            rec = assignment_for(participant)
            done = int(rec.get('done', 0))
            order = rec['order']
            cond = order[min(done, len(order) - 1)]
            sets = order_sets_for(EXPERIMENT_PRESET)
            cases = experiment_case_indices(EXPERIMENT_PRESET) or list(range(len(sets)))
            case = random.choice(cases)
            return {'map': cond['map'], 'preset': EXPERIMENT_PRESET, 'case': case,
                    'recipes': list(sets[case]), 'picked_by': 'experiment',
                    'participant': participant, 'session': done + 1,
                    'sessions_total': len(order), 'skip_budget': cond['skip_budget']}
        maps = [m for m, _, _ in MAP_CHOICES]
        presets = [r for r, _, _ in RECIPE_CHOICES]
        map_name = choice.get('map') if choice.get('map') in maps else maps[0]
        preset = choice.get('preset') if choice.get('preset') in presets else presets[-1]
        sets = order_sets_for(preset)
        case = choice.get('case')
        if not isinstance(case, int) or not (0 <= case < len(sets)):
            # おまかせ: 組み合わせの中から毎回ランダムに選ぶ
            case = random.randrange(len(sets))
            picked_by = 'random'
        else:
            picked_by = 'chosen'
        instruction = choice.get('instruction')
        if instruction not in (INSTRUCTION_TIMING_ONCE_AT_START,
                               INSTRUCTION_TIMING_NO_INSTRUCTION):
            instruction = INSTRUCTION_TIMING_ONCE_AT_START
        skip_budget = choice.get('skip_budget')
        if skip_budget not in SKIP_BUDGETS:
            skip_budget = SKIP_BUDGETS[0]
        return {'map': map_name, 'preset': preset, 'case': case,
                'recipes': list(sets[case]), 'picked_by': picked_by,
                'instruction': instruction, 'skip_budget': skip_budget}

    def note_client_rtt(self, rtt_ms):
        """参加者の端末で測った往復時間を残す。/api/perf で見る。

        サーバーの手元で測っても参加者の遅さは分からない(Funnel を
        経由しても、この PC から自分へつなぐと 1ms で返ってくる)。
        """
        try:
            v = float(rtt_ms)
        except (TypeError, ValueError):
            return
        buf = self.perf.setdefault('client_rtt_ms', [])
        buf.append(v)
        del buf[:-120]

    def note_client_stat(self, key, value):
        """端末が報告した数値(描画にかかった時間など)を残す。/api/perf で見る。"""
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        buf = self.perf.setdefault(key, [])
        buf.append(v)
        del buf[:-120]

    def note_timeline(self, row):
        """遊んでいる間の通信の様子を1秒ごとに残す(ゲーム後にファイルへ書く)。

        平均だけでは「序盤は速いのに途中から遅くなる」のような変化が
        分からないので、時刻つきで全部残す。
        """
        if self.state not in ('ready', 'running'):
            return
        env = self.env
        row = dict(row, wall=round(time.time(), 2), state=self.state,
                   game_t=round(float(getattr(env, 'current_time', 0.0) or 0.0), 1))
        self.timeline.append(row)

    SESSION_FIELDS = ['timestamp', 'participant_id', 'session', 'map', 'skip_budget',
                      'case', 'orders', 'instruction', 'instruction_verb',
                      'instruction_obj', 'quality', 'wait_seconds', 'wait_censored',
                      'exec_rank', 'natural_rank', 'rank_gain', 'tasks_before',
                      'served', 'failed', 'completed',
                      'makespan_s', 'aborted', 'game_id']

    def _log_session(self):
        """実験のセッションを results/web_sessions.csv に1行ずつ残す。

        アンケート(results/survey.csv)とは participant_id と session で
        突き合わせる。
        """
        sel = self.selection or {}
        if not sel.get('participant'):
            return
        res = self.result or {}
        env = self.env
        append_csv(SESSION_LOG_PATH, self.SESSION_FIELDS, {
            **self.instruction_record(),
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'participant_id': sel['participant'], 'session': sel.get('session'),
            'map': sel.get('map'), 'skip_budget': sel.get('skip_budget'),
            'case': sel.get('case'), 'orders': '|'.join(sel.get('recipes', [])),
            'served': res.get('served'), 'failed': res.get('failed'),
            'completed': int(bool(res.get('success'))),
            'makespan_s': round(float(getattr(env, 'current_time', 0.0) or 0.0), 1),
            'aborted': int(bool(res.get('aborted'))), 'game_id': self.game_id,
        })
        if not res.get('aborted'):
            # 最後までやったセッションだけ数える(途中で切れた回はやり直し)
            note_session_done(sel['participant'])

    def instruction_record(self):
        """この回の指示と、その効き方。skip_budget の効果を見るための値。

        wait_seconds : 指示してから、AI がその作業に取りかかるまでの秒数
        exec_rank    : AI が何番目にその作業をやったか(間に挟んだ数+1)
        natural_rank : 指示しなかったら何番目になるはずだったか
        rank_gain    : 何番手ぶん繰り上がったか
        """
        env = self.env
        pend = list(getattr(env, '_pending_instructions', []) or []) if env else []
        if not pend:
            return {}
        p = pend[0]
        payload = p.get('task')
        if isinstance(payload, (list, tuple)) and len(payload) >= 2:
            payload = payload[1]
        verb = payload.get('verb') if isinstance(payload, dict) else None
        obj = payload.get('obj') if isinstance(payload, dict) else None
        started = p.get('started_env_time')
        tasks_before = p.get('tasks_before')
        exec_rank = (tasks_before + 1) if tasks_before is not None else None
        natural = self.instruction_natural_rank
        return {
            'instruction': f'{verb}_{obj}' if verb else '',
            'instruction_verb': verb or '', 'instruction_obj': obj or '',
            'quality': self.instruction_kinds or '',
            'wait_seconds': started if started is not None else
                            round(float(getattr(env, 'current_time', 0.0) or 0.0), 1),
            'wait_censored': int(started is None),
            'exec_rank': exec_rank,
            'natural_rank': natural,
            'rank_gain': (natural - exec_rank) if (natural and exec_rank) else None,
            'tasks_before': tasks_before,
        }

    def _dish_kinds_of(self, payload):
        """その指示が、どの系統の料理(サラダ/スープ/ジュース)を進めるものか。"""
        try:
            state = getattr(self.game, '_latest_env_state', None)
            ai = getattr(self.game, 'ai', None)
            if state is None or ai is None:
                return ''
            kinds = {}
            for o in ai._build_order_tasks(deepcopy(state)):
                name = str(o.get('name') or '')
                for k in ('salad', 'soup', 'juice'):
                    if name.endswith(k):
                        kinds[o.get('order')] = k
            got = {kinds.get(u) for u in (payload.get('order_uids') or [])}
            return '/'.join(sorted(k for k in got if k))
        except Exception:
            return ''

    def measure_natural_rank(self, verb, obj):
        """指示しなかったら、その作業は AI の何番目になるはずだったかを測る。

        指示を受けた場面をそのまま別の AI に解かせて、順番だけ見る。
        ゲームの進行とは別のスレッドで動かす(CP-SAT に数秒かかるため)。
        """
        state = getattr(self.game, '_latest_env_state', None)
        if state is None:
            return

        def work():
            try:
                from agent.myagent.CSPAgent import CSPAgent
                from gym_cooking.utils.replay import Replay as _Replay
                ai = CSPAgent(10, _Replay(), sc_2agent=True,
                              skip_budget=getattr(self.game.ai, 'skip_budget', None))
                ai.human_counterpart_mode = True
                ai.own_agent_idx = getattr(self.game.ai, 'own_agent_idx', 0)
                ai.priority_weights = {}
                ai.gui_text_input = ''
                ai.gui_constraint_input = ''
                ai.active_constraints = []
                ai(deepcopy(state))
                sched = (ai.schedule_per_agent or {}).get(ai.own_agent_idx) or []
                for i, t in enumerate(sched, 1):
                    tid = t.get('id')
                    if tid and str(tid[0]) == str(verb) and str(tid[1]) == str(obj):
                        self.instruction_natural_rank = i
                        return
            except Exception as e:
                print(f'[server] 指示なしの順番を測れませんでした: {e}')

        threading.Thread(target=work, daemon=True).start()

    def _save_timeline(self, reason):
        if not self.timeline:
            return
        outdir = ROOT / 'results' / 'web_perf'
        outdir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = outdir / f'{stamp}-i{INSTANCE_ID}-game{self.game_id}.json'
        try:
            path.write_text(json.dumps({
                'selection': self.selection, 'end_reason': reason,
                'disconnect': self.disconnect_reason,
                'connection': self.connection_info, 'timeline': self.timeline,
            }, ensure_ascii=False, indent=1), encoding='utf-8')
            print(f'[server] 通信の記録を保存しました: {path}')
        except Exception as e:
            print(f'[server] 通信の記録の保存に失敗: {e}')

    def release(self, token):
        """接続が切れた。遊んでいる途中なら、そのゲームは打ち切る。

        つなぎ直しても続きからは遊べない(最初からやり直し)。途中で切れた
        ゲームを続けると、その間 AI だけが動いた記録が残ってしまう。
        """
        with self._player_lock:
            if self.player is token:
                self.player = None
                if self.state in ('preparing', 'ready', 'running'):
                    self._abort()

    def _abort(self):
        self._aborted = True
        game = self.game
        if game is not None and self.state in ('ready', 'running'):
            print(f'[server] #{self.game_id} 接続が切れたので、このゲームを打ち切ります')
            game._q_control.put(('Quit', {}))
            # 指示パネルを開いている間は、ゲーム側は _q_control を見ていない
            # (パネル自身の入力待ちの中にいる)。pygame の終了イベントを
            # 流せばパネルが閉じ、そのまま打ち切りに進む。これが無いと、
            # パネルを出したまま切れた回が終わらず、枠が永久に空かなかった。
            try:
                pygame.event.post(pygame.event.Event(pygame.QUIT))
            except Exception:
                pass

    def slot_free(self):
        return self.player is None and self.state == 'waiting'

    def remaining_seconds(self):
        """いまのゲームの残り時間(ゲーム内の秒)。遊んでいなければ None。"""
        env = self.env
        if self.state != 'running' or env is None:
            return None
        limit = getattr(getattr(env, 'arglist', None), 'max_num_timesteps', 0) or 0
        if not limit:
            return None
        return max(0.0, float(limit) - float(getattr(env, 'current_time', 0.0)))

    def ask_instruction(self, candidates, env_summary):
        """指示の候補をブラウザへ送り、選ばれるまで待つ。

        ゲームのスレッドから呼ばれる(選んでいる間ゲームは止まっている)。
        時間制限は付けない。接続が切れたときだけ、待つのをやめる。
        """
        items = []
        for display, payload in candidates:
            verb = payload.get('verb') if isinstance(payload, dict) else None
            obj = payload.get('obj') if isinstance(payload, dict) else None
            items.append({
                'label': card_label(verb, obj) if verb else str(display),
                'action': card_action(verb) if verb else '',
                'icon': card_icon_name(verb, obj) if verb else None,
                'verb': verb, 'obj': obj,
            })
        with self._instruction_lock:
            self._instruction_answer = None
            self._instruction_done.clear()
            self._instruction_seq += 1
            self.instruction_request = {
                'seq': self._instruction_seq,
                'items': items,
                'players': (env_summary or {}).get('players', []),
            }
        try:
            while not self._instruction_done.wait(0.5):
                if self.state not in ('ready', 'running') or getattr(self, '_aborted', False):
                    return None      # 打ち切り(接続が切れた等)
            idx = self._instruction_answer
            if idx is None or not (0 <= idx < len(candidates)):
                return None
            chosen = candidates[idx]
            payload = chosen[1] if isinstance(chosen, (list, tuple)) and len(chosen) > 1 else {}
            if isinstance(payload, dict) and payload.get('verb'):
                self.instruction_kinds = self._dish_kinds_of(payload)
                self.measure_natural_rank(payload['verb'], payload.get('obj'))
            return chosen
        finally:
            with self._instruction_lock:
                self.instruction_request = None

    def answer_instruction(self, seq, index):
        """ブラウザで選ばれた指示を受け取る。"""
        with self._instruction_lock:
            req = self.instruction_request
            if not req or req['seq'] != seq:
                return False
            self._instruction_answer = int(index)
            self._instruction_done.set()
            return True

    def notify(self, text):
        with self._notices_lock:
            self._notices.append(text)

    def take_notices(self):
        with self._notices_lock:
            out, self._notices = self._notices, []
        return out

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

        sel = self.selection
        map_name = sel['map'] if sel else a.map
        orders = sel['recipes'] if sel else a.orders
        if sel and not uses_fruit(orders):
            # 野菜だけの注文では、フルーツ・ミキサー・コップのない版の地図を使う
            map_name = f'{map_name}_veg'
        # 実験のセッションでは、指示は開始直後に1回だけ(見送り不可)。
        # 自由に遊ぶときは、ステージ選択で「指示あり/なし」を選べる。
        if sel and sel.get('participant'):
            timing = INSTRUCTION_TIMING_ONCE_AT_START
        elif sel and sel.get('instruction') in INSTRUCTION_TIMINGS:
            timing = sel['instruction']
        else:
            timing = a.instruction_request_timing
        self.game, self.env, self.replay = play_main.init_env_replay(
            map_name, a.agent0, a.agent1, a.task,
            a.no_reschedule, a.debug,
            orders, a.order_seed,
            timing,
        )
        if sel:
            # 何を選んで遊んだかをリプレイにも残す
            self.replay['web_selection'] = dict(sel)
        if sel and sel.get('skip_budget') is not None:
            # AI が指示を後回しにできる量。実験では条件として割り当て、
            # 自由に遊ぶときはステージ選択で選ぶ。時間の締め切り
            # (deadline_seconds)は使わない(シミュレーション側と同じ条件)。
            ai = getattr(self.game, 'ai', None)
            if ai is not None:
                ai.skip_budget = int(sel['skip_budget'])
                ai.deadline_seconds = None
        return self.game

    def prepare(self):
        """次の1ゲームを組み立て、Web 版に要る差し替えを入れる。"""
        self.state = 'preparing'
        self.take_notices()
        game = self.build()

        self._me_range = None
        self._other_range = None
        # 指示はブラウザ側に出す(ゲーム画面を隠さないため)
        game.instruction_chooser = self.ask_instruction
        self._install_agent_hook(game)

        # pygame の初期化後に pygame.mouse / display を差し替えたいので、フックしておく。
        original_on_init = game.on_init

        def on_init():
            ret = original_on_init()
            self.on_init_done()
            return ret

        game.on_init = on_init

        # 指示の候補が1つも無いとき、ローカル版は文字入力の窓(Tk)を開く。
        # その窓はサーバーの PC の画面に出るため、ブラウザからは閉じられず、
        # ゲームが一時停止したまま止まる。Web 版では開かない。
        original_request = game._request_instruction

        def request_instruction(trigger='space', allow_text_fallback=True):
            # 指示できるのは「AI がいま着手できる作業」だけ。1つも無いときに
            # 何も起きないと、ボタンが壊れているように見えるので知らせる。
            if not game._get_unexecuted_task_candidates():
                self.notify('いま AI に指示できる作業はありません')
                return None
            return original_request(trigger=trigger, allow_text_fallback=False)

        game._request_instruction = request_instruction

        self._font_px = {id(game.small_font): 12, id(game.font): 16,
                         id(game.large_font): 40}
        original_put_text = game.put_text

        def put_text(font, text, color, loc):
            original_put_text(font, text, color, loc)
            elements = game.get_visualization()
            if elements and elements[-1][0] == 'Text':
                elements[-1][1]['px'] = self._font_px.get(id(font), 12)

        game.put_text = put_text

    def run_forever(self):
        """メインスレッドで呼ぶ。1人遊び終わるたびに、次のゲームを用意する。

        遊ぶ人が地図とレシピを選んでスタートを押してから組み立てる。
        """
        while True:
            self.game = None
            self.env = None
            self.selection = None
            self._aborted = False
            self.timeline = []
            self.disconnect_reason = None
            self.instruction_natural_rank = None
            self.instruction_kinds = ''
            with self._pending_lock:
                self._draw = None      # 前のゲームの盤面を残さない
            self.state = 'waiting'
            print(f'[server] #{self.game_id} ブラウザからの接続を待っています...')
            self.client_connected.wait()

            self.prepare()
            sel = self.selection or {}
            print(f"[server] #{self.game_id} 選択: {sel.get('map')} / {sel.get('preset')} "
                  f"/ {sel.get('recipes')}")

            # 時間を止めたまま始める。盤面は描いて送るので、端末は絵を読み
            # 込んで最初の盤面を描き終えてから 3・2・1 を出し、合図(go)を
            # 送ってくる。そこで時間を進め始める。組み立てた直後に進めると、
            # スマホではまだ絵が描けていないうちにゲームが始まっていた。
            self.game._q_env.put(('Pause', {}))
            self._released = False
            self.state = 'ready'
            self.perf.update(rendered=0, encoded=0, sent=0, started=time.time(),
                             client_rtt_ms=[], client_paint_ms=[], client_recv_fps=[],
                             send_wait_ms=[])
            print(f'[server] #{self.game_id} 盤面を用意しました(開始の合図待ち)')
            if self._aborted:
                # 組み立てている間に切れていた
                self._abort()
            success = False
            try:
                success = self.game.on_execute()
            finally:
                self.state = 'finished'
                self._save_replay()
                self._save_timeline('aborted' if self._aborted else 'finished')

            order = self.env.order_scheduler
            self.result = {
                'success': bool(success),
                'served': getattr(order, 'successful_orders', 0),
                'failed': getattr(order, 'failed_orders', 0),
                'reward': getattr(order, 'reward', 0),
                'aborted': bool(self._aborted),
                'makespan_s': round(float(getattr(self.env, 'current_time', 0.0) or 0.0), 1),
            }
            print(f'[server] #{self.game_id} ゲーム終了: {self.result}')
            self._log_session()

            # 結果を残してから番号を進める(遊んだ人の接続は、番号が
            # 変わったのを見て結果を受け取り、画面を結果表示に切り替える)。
            self.results[self.game_id] = self.result
            with self._player_lock:
                self.player = None
            self.client_connected.clear()
            self.game_id += 1

    def on_init_done(self):
        """pygame の初期化後に呼ぶ(display が出来てから差し替える)。

        ゲームを作り直すたびに呼ばれるが、差し替えはモジュールの関数に
        対して行うので、1回だけでよい(2回やると二重に包まれる)。
        """
        if self._hooks_installed:
            return
        self._hooks_installed = True
        self.mouse = RemoteMouse()
        self._install_frame_hook()

    def _install_agent_hook(self, game):
        """人が操作するキャラを描いた範囲を、コマごとに覚えておく。

        端末は押した瞬間に自分のキャラだけ先に動かして見せる(先読み)。
        そのとき、サーバーが送ってきた位置のキャラは消して、自分で
        先読みした位置に描き直す必要がある。どの命令がそのキャラの分かは
        描いた順でしか分からないので、描いている間に範囲を控えておく。
        """
        idx = getattr(game, 'idx_human', None)
        if idx is None or idx >= len(game.sim_agents):
            return
        me = game.sim_agents[idx]
        original = game.draw_agent
        original_facing = game.draw_agent_facing

        def draw_agent_facing(agent):
            # 向きの枠だけは、端末が自分で描き直せるように位置を控える
            self._mark_index = len(game.get_visualization())
            original_facing(agent)

        def draw_agent(agent):
            plot = game.get_visualization()
            start = len(plot)
            self._mark_index = None
            original(agent)
            rng = (start, len(plot))
            mine = agent is me or getattr(agent, 'name', None) == me.name
            if mine:
                self._me_range = rng
                self._me_mark = self._mark_index
            else:
                self._other_range = rng

        game.draw_agent_facing = draw_agent_facing
        game.draw_agent = draw_agent

    def walkable_grid(self):
        """通れるマスの地図。端末の先読みで「そこへ動けるか」を見るのに使う。"""
        env = self.env
        world = getattr(env, 'world', None)
        if world is None:
            return None
        blocked = set()
        for name, objs in world.objects.items():
            for o in objs:
                if getattr(o, 'collidable', False):
                    blocked.add(tuple(o.location))
        return [''.join('#' if (x, y) in blocked else '.' for x in range(world.width))
                for y in range(world.height)]

    def me_state(self):
        """人のキャラの位置・相手の位置・処理済みの入力数。"""
        game, env = self.game, self.env
        idx = getattr(game, 'idx_human', None)
        if game is None or env is None or idx is None:
            return None
        agents = env.sim_agents
        if idx >= len(agents):
            return None
        other = agents[1 - idx] if len(agents) > 1 else None
        return {
            'pos': list(agents[idx].location),
            'facing': list(getattr(agents[idx], 'facing', (0, 1))),
            'other': list(other.location) if other is not None else None,
            'done': int(getattr(game, 'human_inputs_done', 0)),
            'range': list(self._me_range) if self._me_range else None,
            'mark': self._me_mark,
            'other_range': list(self._other_range) if self._other_range else None,
        }

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
            # ゲーム画面だけを描いたコマなら、描画命令も取っておく。
            # 指示パネルを出しているとき(画面が横に広がる)は、パネルが
            # 命令を残さないので画像で送る。
            draw = None
            me = None
            game = self.game
            if game is not None and snapshot.get_size() == (game.width, game.height):
                try:
                    draw = normalize_elements(game.get_visualization())
                    me = self.me_state()
                except Exception:
                    draw = None
            with self._pending_lock:
                self._pending_surface = snapshot
                self._draw = draw
                self._draw_me = me
                self._draw_version += 1
            self.perf['rendered'] += 1
            self._frame_ready.set()

        pygame.display.flip = flip

    def _save_replay(self):
        a = self.args
        repdir = ROOT / 'agent' / 'agent' / 'replay'
        repdir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        map_name = (self.selection or {}).get('map') or a.map
        path = repdir / f'web{INSTANCE_ID}-{map_name}-{a.agent0}-{a.agent1}-{stamp}.rep'
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

        data = encode_frame(surface)
        if data is None:
            return None

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

    def wait_frame(self, timeout=0.5):
        """次の1枚が描かれるまで待つ(画像にはしない)。executor から呼ぶ。"""
        if not self._frame_ready.wait(timeout):
            return False
        self._frame_ready.clear()
        return True

    def latest_draw(self):
        with self._pending_lock:
            return self._draw_version, self._draw, self._draw_me

    def latest_frame(self):
        with self._frame_lock:
            return self._frame_version, self._frame, self._frame_size

    # ------------------------------------------------------------------
    # 入力
    # ------------------------------------------------------------------
    def post_key(self, code, up=False):
        key = KEY_MAP.get(code)
        if key is None:
            return False
        # 離したことも伝える。長押し中に「置く/取る」を何度も繰り返さない
        # ようにするため、ゲーム側が押しっぱなしかどうかを見ている。
        pygame.event.post(pygame.event.Event(
            pygame.KEYUP if up else pygame.KEYDOWN,
            key=key, mod=0, unicode='', scancode=0))
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
app.mount('/graphics', StaticFiles(directory=str(WEB_GRAPHICS_DIR)), name='graphics')


# いま使っている低遅延の入口(Cloudflare のトンネル)の URL。
# tools/serve_public.py が起動のたびにここへ書く。
PUBLIC_URL_PATH = ROOT / '.cache' / 'public_url.txt'

# 何番目の入口か。4人同時に遊べるようにすると同じ PC で複数のサーバーが
# 動くので、残すファイルの名前が秒単位で衝突する(リプレイ・通信の記録)。
# 番号を名前に入れて、別々の記録として残す。
INSTANCE_ID = 0


def public_url():
    try:
        url = PUBLIC_URL_PATH.read_text(encoding='utf-8').strip()
    except OSError:
        return None
    return url if url.startswith('http') else None


def came_via_funnel(req):
    host = (req.headers.get('host') or '').lower()
    return 'tailscale-funnel-request' in req.headers or host.endswith('.ts.net')


@app.get('/')
async def index(req: Request):
    # 固定 URL(Tailscale Funnel)は受付だけにして、実際に遊ぶのは
    # そのとき立てている低遅延の入口へ送る。Funnel は中継が遠く、
    # 同じ Wi-Fi からでも往復1秒近くかかることがあった。
    url = public_url()
    if url and came_via_funnel(req):
        body = ('<!doctype html><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                f'<meta http-equiv="refresh" content="0;url={url}">'
                '<title>Overcooked CSP</title>'
                '<body style="background:#14171c;color:#e8ecf1;font-family:sans-serif;'
                'display:flex;flex-direction:column;align-items:center;justify-content:center;'
                'height:100vh;margin:0;text-align:center;gap:14px">'
                '<div>遊ぶ画面へ移動しています...</div>'
                f'<a style="color:#8fd3a8" href="{url}">自動で移動しないときはここを押してください</a>'
                '</body>')
        return HTMLResponse(body, headers={'Cache-Control': 'no-store'})
    # 画面を直したときに、参加者の端末に古い版が残らないようにする。
    return FileResponse(WEB_DIR / 'index.html',
                        headers={'Cache-Control': 'no-store'})


@app.get('/api/public_url')
async def api_public_url():
    return JSONResponse({'url': public_url()})


@app.get('/api/sprites')
async def sprites():
    """ゲームで使う絵の名前の一覧。端末はこれを先に全部読み込んでおく。"""
    files = sorted(WEB_GRAPHICS_DIR.glob('*.png'))
    return JSONResponse({
        'names': [p.stem for p in files],
        # 端末に「どれくらい落とすか」を見せるための合計バイト数
        'total_bytes': sum(p.stat().st_size for p in files),
    })


@app.get('/api/options')
async def options():
    """スタート画面で選べる地図・レシピ・注文の組み合わせ。"""
    recipes = []
    for preset, label, desc in RECIPE_CHOICES:
        sets = order_sets_for(preset)
        recipes.append({
            'id': preset, 'label': label, 'desc': desc,
            'combos': [{'case': i, 'label': ' / '.join(recipe_label(r) for r in rs)}
                       for i, rs in enumerate(sets)],
        })
    return JSONResponse({
        'maps': [{'id': m, 'label': label, 'desc': desc} for m, label, desc in MAP_CHOICES],
        'recipes': recipes,
    })


@app.get('/api/config')
async def config():
    a = session.args
    sel = session.selection or {}
    return JSONResponse({
        'selection': sel,
        'map': sel.get('map', a.map),
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
                'facing': list(getattr(a, 'facing', (0, 1))),
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


@app.get('/api/slot')
async def slot():
    """操作枠がいま埋まっているか。順番待ちが詰まったときの確認用。"""
    return JSONResponse({
        'state': session.state,
        'game_id': session.game_id,
        'player_connected': session.player is not None,
        'remaining_s': session.remaining_seconds(),
    })


SURVEY_FIELDS = (['participant_id', 'session', 'timestamp']
                 + [f'coord_{i}' for i in range(1, 7)] + ['coord_mean']
                 + [f'trust_{i}' for i in range(1, 9)]
                 + ['trust_mean', 'trust_dnf_count',
                    'map', 'skip_budget', 'case', 'served', 'makespan_s'])


@app.post('/api/survey')
async def survey(req: Request):
    """セッション直後のアンケートを1行ずつ results/survey.csv に足す。

    協調感(1〜5の6項目)は単純平均。信頼感(0〜7の8項目)は「あてはまらない」
    を欠損として除いた平均。どちらも点数まで CSV に入れる。
    """
    body = await req.json()
    pid = str(body.get('participant_id') or '').strip()
    if not pid:
        return JSONResponse({'ok': False, 'error': '参加者IDがありません'}, status_code=400)

    coord = []
    for i in range(1, 7):
        v = body.get(f'coord_{i}')
        if not isinstance(v, (int, float)) or not (1 <= v <= 5):
            return JSONResponse({'ok': False, 'error': f'協調感の{i}番が未回答です'},
                                status_code=400)
        coord.append(int(v))

    trust = []
    for i in range(1, 9):
        v = body.get(f'trust_{i}')
        if v in (None, '', 'dnf'):
            trust.append(None)          # 「あてはまらない」= 欠損。0点ではない
            continue
        if not isinstance(v, (int, float)) or not (0 <= v <= 7):
            return JSONResponse({'ok': False, 'error': f'信頼感の{i}番が未回答です'},
                                status_code=400)
        trust.append(int(v))
    if body.get('answered_all') is not True and any(
            body.get(f'trust_{i}') is None for i in range(1, 9)):
        return JSONResponse({'ok': False, 'error': '信頼感に未回答があります'},
                            status_code=400)

    got = [v for v in trust if v is not None]
    row = {
        'participant_id': pid, 'session': body.get('session'),
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'coord_mean': round(sum(coord) / len(coord), 2),
        'trust_mean': round(sum(got) / len(got), 2) if got else '',
        'trust_dnf_count': sum(1 for v in trust if v is None),
        'map': body.get('map'), 'skip_budget': body.get('skip_budget'),
        'case': body.get('case'), 'served': body.get('served'),
        'makespan_s': body.get('makespan_s'),
    }
    for i, v in enumerate(coord, 1):
        row[f'coord_{i}'] = v
    for i, v in enumerate(trust, 1):
        row[f'trust_{i}'] = '' if v is None else v
    append_csv(SURVEY_PATH, SURVEY_FIELDS, row)
    print(f"[server] アンケートを保存しました: {pid} session={row['session']} "
          f"協調 {row['coord_mean']} 信頼 {row['trust_mean']} "
          f"あてはまらない {row['trust_dnf_count']}件")
    return JSONResponse({'ok': True})


@app.get('/api/assignment')
async def assignment(participant: str = ''):
    """参加者に割り当てた条件の並びと、次のセッション番号。"""
    pid = participant.strip()
    if not pid:
        return JSONResponse({'ok': False, 'error': '参加者IDがありません'}, status_code=400)
    rec = assignment_for(pid)
    done = int(rec.get('done', 0))
    total = len(rec['order'])
    nxt = rec['order'][min(done, total - 1)]
    label = dict((m, l) for m, l, _ in MAP_CHOICES).get(nxt['map'], nxt['map'])
    return JSONResponse({'ok': True, 'participant_id': pid, 'done': done,
                         'total': total, 'session': min(done + 1, total),
                         'finished': done >= total, 'next_map_label': label})


@app.get('/api/perf')
async def perf():
    """どこが遅いかを切り分けるための実測値。

    env_period_ms が 1/fps(=100ms)より大きいのに env_work_ms が小さければ、
    処理が重いのではなく、他スレッドに邪魔されて環境が回れていない。
    """
    p = dict(session.perf)
    elapsed = max(1e-6, time.time() - p.pop('started'))
    for key in ('client_paint_ms', 'client_recv_fps', 'send_wait_ms'):
        vals = sorted(p.pop(key, []) or [])
        if vals:
            p[key + '_median'] = round(vals[len(vals) // 2], 1)
            p[key + '_max'] = round(vals[-1], 1)
    rtts = sorted(p.pop('client_rtt_ms', []) or [])
    if rtts:
        p['client_rtt_median_ms'] = rtts[len(rtts) // 2]
        p['client_rtt_p90_ms'] = rtts[int(len(rtts) * 0.9)]
        p['client_rtt_max_ms'] = rtts[-1]
        p['client_rtt_samples'] = len(rtts)
    stats = getattr(session.game, 'loop_stats', {}) or {}
    return JSONResponse({
        'elapsed_s': round(elapsed, 1),
        'rendered_fps': round(p['rendered'] / elapsed, 2),
        'encoded_fps': round(p['encoded'] / elapsed, 2),
        'sent_fps': round(p['sent'] / elapsed, 2),
        'env_work_ms': round(stats.get('work_s', 0.0) * 1000, 1),
        'env_period_ms': round(stats.get('period_s', 0.0) * 1000, 1),
        'env_target_ms': round(1000 / max(getattr(session.game, 'fps', 10), 1), 1),
        'cpu_count': os.cpu_count(),
        **p,
    })


@app.get('/api/instructions')
async def instructions():
    """受理した指示と、その時間損失量 L(d) の一覧。

    L(d) = f'(d) - f。f は指示制約なしの最適 makespan、f'(d) は
    「指示タスクの前に同エージェントが実行してよい他タスクは d 個まで」
    という制約ありの最適 makespan。大きいほど段取りから外れた指示。
    """
    env = session.env
    pending = list(getattr(env, '_pending_instructions', []) or []) if env else []
    out = []
    for p in pending:
        task = p.get('task')
        name = None
        if isinstance(task, dict):
            name = task.get('verb') and f"{task['verb']}_{task.get('obj', '')}"
        out.append({
            'id': p.get('id'),
            'task': name or str(task),
            'trigger': p.get('trigger'),
            'accepted_env_time': p.get('accepted_env_time'),
            'status': p.get('status'),
            'time_loss': p.get('time_loss'),
        })
    return JSONResponse({'count': len(out), 'instructions': out})


async def wait_in_line(sock: WebSocket):
    """別の人が遊んでいる間、残り時間を知らせ続ける。空いたら知らせて切る。"""
    try:
        while True:
            if session.slot_free():
                await sock.send_text(json.dumps({'type': 'free'}))
                return
            remaining = session.remaining_seconds()
            await sock.send_text(json.dumps({
                'type': 'busy',
                'state': session.state,
                'remaining': None if remaining is None else round(remaining),
            }))
            await asyncio.sleep(1.0)
    except Exception:
        pass


@app.websocket('/ws')
async def ws(sock: WebSocket):
    await sock.accept()
    token = object()
    if not session.try_acquire(token):
        await wait_in_line(sock)
        return
    # どの経路で来たか(家の中の LAN か、Funnel 経由か)。遅さの切り分けに使う。
    # server_multi.py 経由だと接続元がその受付(127.0.0.1)になるので、
    # 受付が伝えてくる本当の接続元を優先する。
    host = (sock.headers.get('x-forwarded-for', '').split(',')[0].strip()
            or getattr(sock.client, 'host', '') or '')
    via_funnel = ('tailscale-funnel-request' in sock.headers
                  or host in ('127.0.0.1', '::1'))
    session.connection_info = {'route': 'funnel' if via_funnel else 'lan', 'host': host,
                               'user_agent': sock.headers.get('user-agent', '')}
    print(f"[server] #{session.game_id} 接続: {session.connection_info['route']} ({host})")
    my_game = session.game_id
    acquired_at = time.time()
    last_seen = [time.time()]
    # 'draw' = 盤面の描画命令だけを送り、ブラウザで描く(既定)
    # 'png'  = 描いた画像を送る(従来の方式。?mode=png で選べる)
    mode = ['draw']

    loop = asyncio.get_running_loop()
    sent_version = 0
    sent_draw_version = 0
    sent_size = None
    last_state = None
    encoder = DrawEncoder()
    frame_no = [0, 0]      # [送った盤面の番号, 端末が描いたと返事した番号]
    last_rtt = [None]      # 端末が測った往復時間(ms)

    async def pump_input():
        """ブラウザからの入力を pygame イベントへ流し続ける。"""
        while True:
            raw = await sock.receive_text()
            last_seen[0] = time.time()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            kind = msg.get('type')
            if kind == 'key':
                session.post_key(msg.get('code'), up=bool(msg.get('up')))
            elif kind == 'mousemove':
                session.post_mouse_move(msg.get('x', 0), msg.get('y', 0))
            elif kind == 'mousedown':
                session.post_mouse_down(msg.get('x', 0), msg.get('y', 0))
            elif kind == 'start':
                session.start(token, {
                    'map': msg.get('map'), 'preset': msg.get('preset'),
                    'case': msg.get('case'),
                    'instruction': msg.get('instruction'),
                    'skip_budget': msg.get('skip_budget'),
                    'participant': msg.get('participant')})
            elif kind == 'ack':
                try:
                    frame_no[1] = max(frame_no[1], int(msg.get('n', 0)))
                except (TypeError, ValueError):
                    pass
            elif kind == 'instruct':
                session.answer_instruction(msg.get('seq'), msg.get('index'))
            elif kind == 'go':
                session.go(token)
            elif kind == 'hello':
                mode[0] = 'png' if msg.get('mode') == 'png' else 'draw'
                if isinstance(msg.get('net'), dict):
                    # 端末から見た回線の種類(Wi-Fi / モバイル回線など。分かる端末だけ)
                    session.connection_info = dict(session.connection_info or {}, net=msg['net'])
            elif kind == 'ping':
                # RTT 計測用。クライアントの送信時刻をそのまま返す。
                if msg.get('rtt') is not None:
                    session.note_client_rtt(msg.get('rtt'))
                    try:
                        last_rtt[0] = float(msg['rtt'])
                    except (TypeError, ValueError):
                        pass
                session.note_client_stat('client_paint_ms', msg.get('paint_ms'))
                session.note_client_stat('client_recv_fps', msg.get('recv_fps'))
                session.note_timeline({
                    'rtt': msg.get('rtt'), 'paint_ms': msg.get('paint_ms'),
                    'recv_fps': msg.get('recv_fps'),
                    'defs': msg.get('defs'), 'seq': msg.get('seq'),
                    'unacked': frame_no[0] - frame_no[1], 'sent': frame_no[0],
                })
                await sock.send_text(json.dumps({'type': 'pong', 't': msg.get('t')}))

    async def send_text(obj):
        # 送信が戻ってこないことがある(相手が消えた直後)。待ち続けると
        # 枠を握ったまま止まるので、時間を切る。
        await asyncio.wait_for(sock.send_text(json.dumps(obj)), SEND_TIMEOUT_S)

    async def stream():
        """描画ができるたびに画面を送る。状態の変化もここで知らせる。"""
        nonlocal sent_version, sent_draw_version, sent_size, last_state
        while True:
            # 次の描画を待つ。待ち(と画像モードでの PNG 化)は executor 側なので
            # イベントループは塞がらず、入力(ping/キー)は待たされない。
            if mode[0] == 'png':
                await loop.run_in_executor(None, session.wait_and_capture)
            else:
                await loop.run_in_executor(None, session.wait_frame)

            if (session.state == 'ready'
                    and time.time() - acquired_at > START_TIMEOUT_S):
                session.go(token)

            # 枠を取ったままスタートされないと、後ろの人がずっと待たされる。
            if (session.state == 'waiting'
                    and time.time() - acquired_at > START_TIMEOUT_S):
                await send_text({
                    'type': 'kicked',
                    'text': 'しばらくスタートされなかったので、<br>順番を次の人に譲りました'})
                return

            # この人の回が終わった。結果を渡して、次の人に枠を譲る。
            if session.game_id != my_game:
                await send_text({'type': 'status', 'state': 'finished',
                                 'result': session.results.get(my_game)})
                return

            # 次のゲームを組み立てる前(開始待ち)は盤面がない。前のゲームの
            # 盤面を送ろうとして落ち、接続が切れていた。
            game = session.game
            if game is None:
                await send_status_and_notices()
                continue

            draw_version, draw, me = session.latest_draw()
            if mode[0] == 'draw' and draw is not None:
                # 盤面の描画命令だけを送り、ブラウザで描いてもらう。
                base = (game.width, game.height)
                if sent_size != base:
                    sent_size = base
                    # 先読みに要るもの(1マスの大きさ・通れるマスの地図)も一緒に渡す
                    await send_text({'type': 'meta', 'w': base[0], 'h': base[1],
                                     'base_w': base[0], 'base_h': base[1],
                                     'tile': getattr(game, 'scale', 40),
                                     'hz': game_config.INPUT_HZ,
                                     'grid': session.walkable_grid()})
                if (draw_version != sent_draw_version
                        and frame_no[0] - frame_no[1] < unacked_limit(last_rtt[0])):
                    sent_draw_version = draw_version
                    frame_no[0] += 1
                    msg = encoder.encode(draw)
                    msg['n'] = frame_no[0]
                    if me:
                        # 端末が先読みした自分の位置を、サーバーの位置に
                        # 合わせ直すための情報。
                        msg['me'] = me
                    t_send = time.time()
                    await send_text(msg)
                    session.note_client_stat('send_wait_ms', (time.time() - t_send) * 1000)
                    session.perf['sent'] += 1
                await send_status_and_notices()
                continue

            if mode[0] == 'draw':
                # 指示パネルを出している間は、パネルが描画命令を残さないので
                # 画像で送る(ゲームは止まっているので、重さは問題にならない)。
                await loop.run_in_executor(None, session.capture)

            version, data, size = session.latest_frame()

            if size != sent_size and size != (0, 0):
                sent_size = size
                # base_w/base_h はゲーム画面そのものの大きさ。指示パネルを
                # 出すと display はこれより横に広がるが、左側 base_w ぶんは
                # 常にゲーム画面なので、クライアントはそこだけを切り出して
                # 位置を動かさずに描き続けられる。
                await send_text({'type': 'meta', 'w': size[0], 'h': size[1],
                                 'base_w': game.width,
                                 'base_h': game.height})

            if data is not None and version != sent_version:
                sent_version = version
                await asyncio.wait_for(sock.send_bytes(data), SEND_TIMEOUT_S)
                session.perf['sent'] += 1

            await send_status_and_notices()

    sent_instruction_seq = [0]

    async def send_status_and_notices():
        nonlocal last_state
        req = session.instruction_request
        if req and req['seq'] != sent_instruction_seq[0]:
            sent_instruction_seq[0] = req['seq']
            await send_text({'type': 'instruct', **req})
        elif not req and sent_instruction_seq[0]:
            sent_instruction_seq[0] = 0
            await send_text({'type': 'instruct_close'})
        for text in session.take_notices():
            await send_text({'type': 'notice', 'text': text})
        if session.state != last_state:
            last_state = session.state
            sel = session.selection or {}
            await send_text({'type': 'status', 'state': session.state,
                             'result': session.result,
                             'experiment': {
                                 'participant_id': sel.get('participant'),
                                 'session': sel.get('session'),
                                 'sessions_total': sel.get('sessions_total'),
                                 # アンケートに添える条件(画面には出さない)
                                 'map': sel.get('map'), 'case': sel.get('case'),
                                 'skip_budget': sel.get('skip_budget'),
                             } if sel.get('participant') else None,
                             'selection': {
                                 'instruction': sel.get('instruction'),
                                 'skip_budget': sel.get('skip_budget'),
                                 'map': dict((m, l) for m, l, _ in MAP_CHOICES).get(sel.get('map')),
                                 'preset': dict((r, l) for r, l, _ in RECIPE_CHOICES).get(sel.get('preset')),
                                 'orders': [recipe_label(r) for r in sel.get('recipes', [])],
                             } if sel else None})

    async def watchdog():
        """何も言わずに消えた端末(電波切れ等)を見つける。"""
        while True:
            await asyncio.sleep(1.0)
            if time.time() - last_seen[0] > PLAYER_SILENCE_TIMEOUT_S:
                return

    # 入力・画面送り・見張りを別々に回し、どれか1つでも終わったら全部止めて
    # 枠を空ける。1本のループで順に見ていると、送信が戻ってこないだけで
    # 切断の確認までたどり着けず、枠を握ったまま止まる(実測: 切断後も
    # 枠が空かず、次の人がずっと「別の人がプレイ中」のままだった)。
    tasks = [asyncio.create_task(c) for c in (pump_input(), stream(), watchdog())]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        names = {tasks[0]: 'input', tasks[1]: 'stream', tasks[2]: 'silence'}
        for t in done:
            err = t.exception() if not t.cancelled() else None
            why = f"{names[t]}: {type(err).__name__} {err}" if err else f"{names[t]}: 終了"
            if session.player is token:
                session.disconnect_reason = why
                print(f'[server] #{session.game_id} 接続が終わった理由: {why}')
    finally:
        for t in tasks:
            t.cancel()
        # 途中で閉じた(再読み込みした)ときも枠を空ける。遊んでいる途中
        # なら、そのゲームは打ち切り、開き直すと最初からになる。
        session.release(token)
        try:
            await asyncio.wait_for(sock.close(), 2)
        except Exception:
            pass


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
    p.add_argument('--instance', type=int, default=0,
                   help='何番目の入口か(server_multi.py が付ける。記録の名前に入る)')

    p.add_argument('--map', type=str, default='exp_partition',
                   choices=['ring', 'bottleneck', 'partition', 'quick', 'juice', 'experiment',
                            'exp_partition', 'exp_bottleneck', 'exp_ring'],
                   help='スタート画面で選ばれなかったときの地図')
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
    p.add_argument('--input-hz', type=int, default=game_config.INPUT_HZ,
                   help='1秒あたりに行動できる回数(既定 %(default)s)')

    return p.parse_args()


def main():
    global session, INSTANCE_ID

    args = parse_arguments()
    INSTANCE_ID = int(args.instance)
    # 1秒あたりに行動できる回数。ゲームを組み立てる前に決めておく
    # (刻む回数や環境の1手の長さがこれで決まる)。
    game_config.set_input_hz(args.input_hz)
    session = WebGamePlay(args)

    start_server_thread(args.host, args.port)

    shown_host = 'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host
    print('=' * 60)
    print(f'  Overcooked CSP web server')
    print(f'  URL: http://{shown_host}:{args.port}/')
    print(f'  構成: map={args.map} agent0={args.agent0} agent1={args.agent1} '
          f'sc_2agent={args.sc_2agent} orders={args.orders} '
          f'deadline={args.deadline} timing={args.instruction_request_timing} '
          f'input_hz={args.input_hz}')
    print('=' * 60)

    try:
        session.run_forever()
    except KeyboardInterrupt:
        print('\n[server] 中断しました')
    except Exception:
        # 落ちた理由を残す。画面のログは流れて消えるので、ファイルにも書く。
        import traceback
        log = ROOT / 'results' / 'server_crash.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('a', encoding='utf-8') as f:
            f.write('--- ' + datetime.now().isoformat(timespec='seconds') + ' ---' + chr(10))
            traceback.print_exc(file=f)
        traceback.print_exc()
        print(f'[server] 落ちました。理由は {log} に残しました', flush=True)
        raise


if __name__ == '__main__':
    main()
