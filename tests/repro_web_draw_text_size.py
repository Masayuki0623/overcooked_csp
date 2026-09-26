"""Web 版で盤面が描けなくなっていないことの検証。

報告された現象:
    「何故か画面が真っ暗で何も描画されていない」

仕組み:
    ブラウザは画像ではなく描画命令の一覧を受け取って描き直す。文字の
    大きさもその一覧に載せる必要があるが、それは server.py が Game.put_text
    を包んで font ごとに書き足している(_font_px)。

    残り時間を大きくしたとき、Game.put_text に px 引数を足してしまった。
    包んでいる側は px を受け取らないので、描画のたびに TypeError が出て
    描画スレッドごと落ち、画面が真っ暗になった。ローカルの pygame だけで
    試していたため気づけなかった。

ここで確かめること:
    1. Game.put_text の引数の形が、server.py の包みと合っている
    2. 包んだあとに呼んでも落ちず、文字の大きさが命令に載る
    3. 残り時間は大きい字で、盤面の右端からはみ出さない

実行方法:
    python tests/repro_web_draw_text_size.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

import inspect

from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from gym_cooking.play_test import MAP_SETTINGS
from gym_cooking.misc.game.game import Game

import server as srv

results = []


def check(label, ok, detail=""):
    results.append(ok)
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}{(' -> ' + detail) if detail else ''}")


# 1. 引数の形。server.py の包みは (font, text, color, loc) で呼ぶ。
params = list(inspect.signature(Game.put_text).parameters)
check('put_text の引数は self, font, text, color, loc のまま',
      params == ['self', 'font', 'text', 'color', 'loc'],
      str(params))


def build(map_key, now=12.0, limit=90):
    env = OvercookedEnvironment(MapSetting(**dict(MAP_SETTINGS[map_key])))
    env.reset()
    env.arglist.max_num_timesteps = limit
    env.current_time = now
    g = Game(env)
    g.on_init()
    # server.py の WebGamePlay.prepare と同じ包み方
    font_px = {id(g.small_font): 12, id(g.font): 16, id(g.large_font): 40,
               id(g.time_font): g.TIME_PX}
    original = g.put_text

    def put_text(font, text, color, loc):
        original(font, text, color, loc)
        els = g.get_visualization()
        if els and els[-1][0] == 'Text':
            els[-1][1]['px'] = font_px.get(id(font), 12)

    g.put_text = put_text
    g.on_render()
    return g, srv.normalize_elements(g.get_visualization())


# 2 + 3
for map_key in ('exp_ring', 'exp_partition', 'tutorial_salad'):
    g, els = build(map_key)
    check(f'{map_key}: 描画命令がちゃんと出ている', len(els) > 10, f'{len(els)} 個')
    times = [e for e in els if e[0] == 'T' and '残り' in e[1]]
    check(f'{map_key}: 残り時間が1つある', len(times) == 1)
    if not times:
        continue
    e = times[0]
    px, x = e[7], e[5]
    check(f'{map_key}: 大きい字で出る (px={px})', px == g.TIME_PX)
    # ブラウザの Times は測った幅より広い。実測で 1.2 倍ほど。
    right = x + g.time_font.size(e[1])[0] * 1.25
    check(f'{map_key}: 右端からはみ出さない',
          right <= g.width, f'右端 {right:.0f} / 盤面 {g.width}')

ok = sum(1 for r in results if r)
print()
if ok == len(results):
    print(f'[SUCCESS] {ok}/{len(results)} 件が期待どおり')
else:
    print(f'[FAILURE] {ok}/{len(results)} 件のみ期待どおり')
    sys.exit(1)
