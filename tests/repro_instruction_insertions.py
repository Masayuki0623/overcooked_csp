"""指示が終わるまでに AI が先に片づけた他の作業の数を、縛りの有無に関わらず数える。

想定(2026-10-04):
    _track_instruction_progress が計画のたびに「前回 AI の担当だった他の作業が
    計画から消えたか」で数える。inf(縛りなし)の回でも同じに数え、結末も
    done になる。以前は縛りの中で数えていたので inf は常に 0・started のまま。

実行方法:
    python tests/repro_instruction_insertions.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from gym_cooking.utils.replay import Replay
from agent.agent.myagent.CSPAgent import CSPAgent

results = []


def check(label, ok, detail=''):
    results.append(bool(ok))
    print(f"  [{'OK  ' if ok else 'NG  '}] {label}" + (f" -> {detail}" if detail else ''))


def task(verb, obj, uid):
    return {'id': (verb, obj, uid), 'verb': verb, 'obj': obj, 'order': uid}


def main():
    ai = CSPAgent(10, Replay(), sc_2agent=True, skip_budget=None)   # inf
    ai.own_agent_idx = 0
    A, B = task('chop', 'lettuce', 2), task('chop', 'apple', 3)
    T1, T2 = task('chop', 'tomato', 0), task('cook', 'onion-tomato soup', 0)
    chain = [('task', 'chop', 'tomato', 0), ('task', 'cook', 'onion-tomato soup', 0)]
    pending = {'id': 'i1', 'status': 'pending', 'skip_budget': None,
               'task': ('cook', {'verb': 'cook', 'obj': 'onion-tomato soup', 'chains': [chain], 'chain': chain, 'count': 1})}
    ai._pending_instructions = [pending]

    class Env:
        _pending_instructions = []

    env = Env()
    print('[1] inf でも、鎖が終わるまでに挟まった作業を数える')
    ai.schedule_per_agent = {0: [A, B, T1, T2], 1: []}
    ai._track_instruction_progress(env, [A, B, T1, T2])
    check('最初の計画: まだ 0', pending.get('_consumed_tasks', 0) == 0)
    ai.schedule_per_agent = {0: [B, T1, T2], 1: []}
    ai._track_instruction_progress(env, [B, T1, T2])          # A を片づけた
    check('A を片づけた -> 1', pending.get('_consumed_tasks') == 1, str(pending.get('_consumed_tasks')))
    ai.completed_task_ids.add(T1['id'])
    ai.schedule_per_agent = {0: [T2, B], 1: []}
    ai._track_instruction_progress(env, [T2, B])              # T1(鎖) を片づけた: 数えない
    check('鎖の工程を片づけても数えない', pending.get('_consumed_tasks') == 1, str(pending.get('_consumed_tasks')))
    ai.schedule_per_agent = {0: [T2], 1: []}
    ai._track_instruction_progress(env, [T2])                 # B を片づけた(鎖の途中)
    check('鎖の途中で B を片づけた -> 2', pending.get('_consumed_tasks') == 2, str(pending.get('_consumed_tasks')))
    ai.completed_task_ids.add(T2['id'])
    ai._track_instruction_progress(env, [])
    check('鎖が全部済んだら done(inf でも)', pending.get('status') == 'done', pending.get('status'))

    print('[2] 相手がやった作業は数えない')
    ai2 = CSPAgent(10, Replay(), sc_2agent=True, skip_budget=None)
    ai2.own_agent_idx = 0
    p2 = {'id': 'i2', 'status': 'pending', 'skip_budget': None,
          'task': ('chop', {'verb': 'chop', 'obj': 'tomato', 'chains': [[chain[0]]], 'chain': [chain[0]], 'count': 1})}
    ai2._pending_instructions = [p2]
    ai2.schedule_per_agent = {0: [T1], 1: [A, B]}
    ai2._track_instruction_progress(env, [A, B, T1])
    ai2.schedule_per_agent = {0: [T1], 1: [B]}
    ai2._track_instruction_progress(env, [B, T1])             # 相手が A を片づけた
    check('相手の分は 0 のまま', p2.get('_consumed_tasks', 0) == 0, str(p2.get('_consumed_tasks')))

    ng = results.count(False)
    if ng:
        print(f'[FAILED] {ng}/{len(results)} 件が期待どおりではありません')
        return 1
    print(f'[SUCCESS] {len(results)}/{len(results)} 件すべて期待どおり')
    return 0


if __name__ == '__main__':
    sys.exit(main())
