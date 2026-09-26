"""エンドレスの回を、あとから同じ注文で再現できることの検証。

背景:
    エンドレスでは注文が片づくたびにくじ引きで補充される。その種を記録に
    残していなかったため、リプレイを回し直すと注文の並びが変わっていた。
    盤面(位置・持ち物)は記録どおりに再現できても、AI が何を考えていたかは
    注文が違えば別物になる。実測で2件の不具合が、この理由で当時の考えまで
    再現できなかった(20260926_153422 と 20260926_161533)。

ここで確かめること:
    1. 同じ種を入れれば、補充される注文の並びが一致する
    2. 種が違えば並びも変わる(種が効いていることの裏取り)
    3. リプレイから作り直す側が、記録した種を使っている

実行方法:
    python tests/repro_endless_replay_reproducible.py
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

results = []


def check(label, ok, detail=""):
    results.append(ok)
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}{(' -> ' + detail) if detail else ''}")


POOL = ('OnionTomatoSalad', 'OnionLettuceSalad', 'TomatoLettuceSoup',
        'OnionLettuceSoup', 'OnionTomatoSoup')


def drawn(seed, n=12):
    """その種で、補充がどの順に出るか。"""
    kw = dict(MAP_SETTINGS['exp_ring_veg'])
    kw.update({'endless_orders': True, 'order_pool': POOL, 'order_seed': seed,
               'max_num_orders': 3, 'max_num_timesteps': 90})
    env = OvercookedEnvironment(MapSetting(**kw))
    env.reset()
    sch = env.order_scheduler
    # 片づいたことにして補充させる。抽選の順だけを見たいので、
    # 盤面は動かさずに注文の入れ替えだけを進める。
    out = list(getattr(sch, 'order_history', []) or [])
    for _ in range(200):
        if len(out) >= n:
            break
        if sch.current_orders:
            sch.current_orders.pop(0)
        sch.update(env.world, 0.2)
        out = list(getattr(sch, 'order_history', []) or [])
    return out[:n]


a = drawn(1234)
b = drawn(1234)
c = drawn(9999)
check('同じ種なら注文の並びが一致する', a == b, f'{a[:5]}')
check('種が違えば並びも変わる', a != c, f'{c[:5]}')

# 記録から作り直す側が種を使っているか
sys.path.insert(0, os.path.join(REPO_ROOT, 'tools'))
import replay_trace
src = inspect.getsource(replay_trace.rebuild)
check('リプレイの作り直しで種を使っている', "sel.get('order_seed')" in src)
check('リプレイの作り直しでエンドレスの設定も入れている',
      "'endless_orders': True" in src)

# 記録する側(サーバー)が種を残しているか
import server as srv
prep = inspect.getsource(srv.WebGamePlay.build)
check('サーバーが種を決めて記録に残している(build)',
      "sel['order_seed'] = order_seed" in prep and "'order_seed': order_seed" in prep)

ok = sum(1 for r in results if r)
print()
if ok == len(results):
    print(f'[SUCCESS] {ok}/{len(results)} 件が期待どおり')
else:
    print(f'[FAILURE] {ok}/{len(results)} 件のみ期待どおり')
    sys.exit(1)
