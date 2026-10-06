"""デバッグ画面に出す「CSP が立てた計画」の検証。

見たいもの(依頼):
    AI と人それぞれの、タスクの選択順番・長さ・開始時点・終了時点。

ここで確かめること:
    1. AI と相手の2人ぶんを返す(AI が先)
    2. 1件ごとに 順番・作業・開始・終了・長さ がそろう
    3. 時刻はゲーム内の秒(フレームではない)
    4. 済んだ作業と、いまやっている作業に印が付く
    5. 計画がまだ無いときは None(画面には何も出さない)
    6. デバッグの回だけ送る(実験の回では送らない)

実行方法:
    python tests/repro_debug_plan_view.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'agent'))

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

import inspect
from types import SimpleNamespace

import server as srv

results = []


def check(label, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}{(' -> ' + detail) if detail else ''}")


class FakeAI:
    fps = 5
    own_agent_idx = 0
    completed_task_ids = {('chop', 'onion', 0)}
    current_task_idx = {0: 1, 1: 0}
    schedule_per_agent = {
        0: [{'id': ('chop', 'onion', 0), 'start': 0, 'end': 10},
            {'id': ('cook', 'onion-tomato soup', 0), 'start': 15, 'end': 25}],
        1: [{'id': ('chop', 'tomato', 0), 'start': 5, 'end': 20}],
    }


def make(ai=FakeAI(), now=4.0):
    s = srv.WebGamePlay.__new__(srv.WebGamePlay)
    s.game = SimpleNamespace(ai=ai)
    s.env = SimpleNamespace(current_time=now)
    return s


snap = make().plan_snapshot()

# 1
check('2人ぶん返る', snap is not None and len(snap['agents']) == 2,
      str(snap and len(snap['agents'])))
check('AI が先', snap['agents'][0]['who'] == 'AI', snap['agents'][0]['who'])
check('相手は「あなた」', snap['agents'][1]['who'] == 'あなた',
      snap['agents'][1]['who'])

# 2 + 3
t = snap['agents'][0]['tasks'][1]
check('順番が付く', t['n'] == 2, str(t['n']))
check('作業が分かる', (t['verb'], t['obj']) == ('cook', 'onion-tomato soup'),
      str((t['verb'], t['obj'])))
check('開始は秒(15フレーム÷5 = 3.0秒)', t['start'] == 3.0, str(t['start']))
check('終了は秒(25フレーム÷5 = 5.0秒)', t['end'] == 5.0, str(t['end']))
check('長さも秒(2.0秒)', t['dur'] == 2.0, str(t['dur']))
check('いまの時刻も秒で入る', snap['now'] == 4.0, str(snap['now']))

# 4
first = snap['agents'][0]['tasks'][0]
check('済んだ作業に印が付く', first['done'] is True, str(first))
check('いまやっている作業に印が付く', t['now'] is True, str(t['now']))
check('相手の1件目もいま扱い', snap['agents'][1]['tasks'][0]['now'] is True)

# 5
class NoPlan:
    schedule_per_agent = None


check('計画が無ければ None', make(NoPlan()).plan_snapshot() is None)

# 6
SRC = inspect.getsource(srv)
check('デバッグの回だけ送る',
      "if session.show_plan and session.state in ('running', 'finished'):" in SRC,
      '実験の回でも送ってしまう')
check('開始時に印を立てている',
      "self.show_plan = bool((choice or {}).get('show_plan'))" in SRC)
check('2人ぶんの割り当ての切り替えも渡している',
      "ai.two_agent_assignment = bool(getattr(self, '_two_agent', False))" in SRC)

print()
ok = sum(1 for r in results if r)
if ok == len(results):
    print(f'[SUCCESS] {ok}/{len(results)} 件すべて期待どおり')
else:
    print(f'[FAILURE] {ok}/{len(results)} 件のみ期待どおり')
    sys.exit(1)
