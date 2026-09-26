"""1回の判断が何秒もかからないことの検証。

報告された現象:
    「最初の方に動かないことがあった」
    (20260926_152900 の回。4.0〜5.6 秒と 6.0〜12.2 秒、AI が
     まったく動かなかった)

原因:
    CP-SAT の探索に上限が入っていなかった。上限を入れる仕組み
    (solve_deterministic_limit)は書いてあったが、既定が None で、
    遊ぶ側では誰も設定していなかった。実測でこの回は探索1回が 1.9 秒、
    1回の判断(複数回解く)が 5.8 秒かかっていた。判断が返るまで AI は
    行動を出せないので、その間ずっと止まって見える。

ここで確かめること:
    1. 遊ぶときの AI には、探索の上限が入っている
    2. 上限は決定性時間で指定する(秒で切ると、PC の速さや他の負荷で
       打ち切る場所が変わり、同じ条件の試行が同じ結果にならない)
    3. L の計測に使う複製では上限を外す(f と f'(d) の両方が最適解で
       ないと意味がないため)

実行方法:
    python tests/repro_solve_time_limit.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

import inspect

from gym_cooking.utils.replay import Replay
from agent.agent.myagent.CSPAgent import CSPAgent

results = []


def check(label, ok, detail=""):
    results.append(ok)
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}{(' -> ' + detail) if detail else ''}")


ai = CSPAgent(10, Replay(), sc_2agent=True, skip_budget=0)
limit = ai.solve_deterministic_limit
check('遊ぶときの AI に探索の上限が入っている',
      limit is not None and limit > 0, f'上限={limit}')

src = inspect.getsource(CSPAgent.solve_csp_scheduling)
check('上限は決定性時間で指定している',
      'max_deterministic_time' in src and 'max_time_in_seconds' not in src)

loss_src = inspect.getsource(CSPAgent._start_time_loss_measurement) \
    if hasattr(CSPAgent, '_start_time_loss_measurement') else ''
if not loss_src:
    # 名前が変わっていても、複製を作っている箇所を探す
    for name, fn in vars(CSPAgent).items():
        if callable(fn):
            try:
                body = inspect.getsource(fn)
            except Exception:
                continue
            if 'probe = _dcopy(self)' in body:
                loss_src = body
                break
check('L の計測に使う複製では上限を外している',
      'probe.solve_deterministic_limit = None' in loss_src)

ok = sum(1 for r in results if r)
print()
if ok == len(results):
    print(f'[SUCCESS] {ok}/{len(results)} 件が期待どおり')
else:
    print(f'[FAILURE] {ok}/{len(results)} 件のみ期待どおり')
    sys.exit(1)
