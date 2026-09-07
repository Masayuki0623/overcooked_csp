"""参加者が実際にプレイする1セッションを走らせ、客観データを記録する。

シミュレーションと同じ環境・同じ AI を人が相手にする。違うのは、人間側を
方策ではなくキーボードが操作することだけ。

    python tools/play_session.py --participant p01 --case 9 --skip-budget 0

操作: 矢印キーで移動、スペースで持つ/置く/使う。
指示: 開始直後に1回だけ選ぶ(見送り不可)。着手できる工程だけが候補に出る。

本実験で使う注文構成は 9〜17 の9通り。この9通りでは「良い指示」が
スープ専属になる(サラダと具材を共有しない)。それ以外を使うときは
--any-case を付ける。

結果は results/play_sessions.csv に1行ずつ足していく。
"""
import argparse
import csv
import os
import sys
import time
from copy import deepcopy as dcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in ('.', 'agent', 'testbed-cooking', 'tools'):
    sys.path.insert(0, str(ROOT / _p))

# 実験ハーネスは読み込むだけで画面なしに設定するので、その前に取り消す。
for _var in ('SDL_VIDEODRIVER', 'SDL_AUDIODRIVER'):
    if os.environ.get(_var) == 'dummy':
        os.environ.pop(_var)

from gym_cooking.utils.order_preset import enumerate_order_recipes  # noqa: E402
from gym_cooking.utils.replay import Replay  # noqa: E402
from agent.mind.agent import AgentSetting  # noqa: E402
from agent.gameplay import GamePlay, INSTRUCTION_TIMING_ONCE_AT_START  # noqa: E402
from agent.myagent.CSPAgent import CSPAgent  # noqa: E402
import run_human_model_experiment as H  # noqa: E402

# 「良い指示」がスープ専属になる注文構成。サラダとスープが AI 側の具材を
# 共有すると、その指示が「スープを優先させた」と言い切れなくなる。
EXPERIMENT_CASES = tuple(range(9, 18))

FIELDS = ['participant', 'session', 'case', 'orders', 'skip_budget',
          'instruction', 'instruction_verb', 'instruction_obj', 'quality',
          'wait_seconds', 'wait_censored', 'exec_rank', 'natural_rank',
          'rank_gain', 'tasks_before', 'verbs_before',
          'served', 'completed', 'makespan_s', 'wall_seconds']


def dish_kind_of(name):
    for k in ('salad', 'soup', 'juice'):
        if str(name).lower().endswith(k):
            return k
    return None


def current_task_id(ai):
    sched = (ai.schedule_per_agent or {}).get(0) or []
    idx = ai.current_task_idx
    idx = idx.get(0, 0) if isinstance(idx, dict) else (idx or 0)
    return sched[idx].get('id') if 0 <= idx < len(sched) else None


def natural_rank_of(state, verb, obj, skip_budget):
    """指示を出さなかった場合、その作業は AI の何番目になるはずだったか。"""
    ai = H.make_ai(skip_budget, partner_is_external=True)
    ai(dcopy(state))
    sched = (ai.schedule_per_agent or {}).get(0) or []
    for i, t in enumerate(sched, 1):
        tid = t.get('id')
        if tid and tid[0] == verb and tid[1] == obj:
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--participant', required=True, help='参加者ID')
    ap.add_argument('--case', type=int, required=True, help='注文構成 (9〜17)')
    ap.add_argument('--skip-budget', type=int, required=True, choices=[0, 2, 4])
    ap.add_argument('--session', type=int, default=None, help='何回目か(1〜3)')
    ap.add_argument('--any-case', action='store_true',
                    help='9〜17 以外の構成も許す(練習用)')
    ap.add_argument('--max-seconds', type=float, default=100.0)
    ap.add_argument('--out', default=str(ROOT / 'results' / 'play_sessions.csv'))
    args = ap.parse_args()

    sets = enumerate_order_recipes('experiment2')
    if not args.any_case and args.case not in EXPERIMENT_CASES:
        ap.error('case は %s のいずれか(練習なら --any-case)'
                 % ', '.join(str(c) for c in EXPERIMENT_CASES))
    recipes = sets[args.case]

    H.MAX_SECONDS_OVERRIDE = args.max_seconds
    env = H.make_env('experiment', args.case, 'experiment2', recipes)
    ai = H.make_ai(args.skip_budget, partner_is_external=True)

    kind_by_uid = {}
    state0 = H.state_for(env, 0)
    for o in ai._build_order_tasks(dcopy(state0)):
        kind_by_uid[o['order']] = dish_kind_of(o['name'])

    print('注文: ' + ' | '.join(recipes))
    print('skip_budget: %d  /  注文構成: case%d' % (args.skip_budget, args.case))
    print('あなたは右側(1番)です。左側(0番)が AI。')
    print('矢印キーで移動、スペースで持つ/置く/使う。')
    print('開始直後に指示を1つ選んでください(見送りはできません)。')

    # --- 観測 -----------------------------------------------------------
    rec = {'wait': None, 'before': [], 'prev': None, 'target': None}
    orig_call = CSPAgent.__call__

    def spy(self, env_state, *a, **kw):
        out = orig_call(self, env_state, *a, **kw)
        if rec['target'] is None:
            pend = (getattr(env, '_pending_instructions', None)
                    or getattr(self, '_pending_instructions', None) or [])
            for p in pend:
                task = p.get('task') or {}
                if task.get('verb'):
                    rec['target'] = (task['verb'], task['obj'],
                                     tuple(task.get('order_uids') or ()))
                    break
        if rec['target'] is None or rec['wait'] is not None:
            return out
        verb, obj, _uids = rec['target']
        cur = current_task_id(self)
        if cur != rec['prev']:
            if rec['prev'] is not None and not (rec['prev'][0] == verb
                                                and rec['prev'][1] == obj):
                rec['before'].append(rec['prev'])
            rec['prev'] = cur
        if cur and cur[0] == verb and cur[1] == obj:
            rec['wait'] = round(env.current_time, 1)
        return out

    CSPAgent.__call__ = spy
    started = time.time()
    game = GamePlay(env, Replay(), AgentSetting('CSP', speed=10),
                    human_agent_idx=1, ai_agent_idx=0,
                    instruction_request_timing=INSTRUCTION_TIMING_ONCE_AT_START)
    game.ai = ai
    try:
        game.on_execute()
    finally:
        CSPAgent.__call__ = orig_call

    # --- 記録 -----------------------------------------------------------
    sched = env.order_scheduler
    verb = obj = None
    quality = ''
    if rec['target'] is not None:
        verb, obj, uids = rec['target']
        kinds = {kind_by_uid.get(u) for u in uids}
        quality = ('good' if kinds == {'soup'}
                   else 'bad' if kinds == {'juice'}
                   else '/'.join(sorted(k for k in kinds if k)))
    nat = natural_rank_of(state0, verb, obj, args.skip_budget) if verb else None
    exec_rank = len(rec['before']) + 1 if rec['wait'] is not None else None

    row = {
        'participant': args.participant, 'session': args.session,
        'case': args.case, 'orders': '|'.join(recipes),
        'skip_budget': args.skip_budget,
        'instruction': ('%s_%s' % (verb, obj)) if verb else '',
        'instruction_verb': verb or '', 'instruction_obj': obj or '',
        'quality': quality,
        'wait_seconds': (rec['wait'] if rec['wait'] is not None
                         else round(env.current_time, 1)),
        'wait_censored': int(rec['wait'] is None),
        'exec_rank': exec_rank, 'natural_rank': nat,
        'rank_gain': (nat - exec_rank) if (nat and exec_rank) else None,
        'tasks_before': len(rec['before']),
        'verbs_before': '|'.join(t[0] for t in rec['before']),
        'served': sched.successful_orders,
        'completed': int(len(sched.current_orders) == 0),
        'makespan_s': round(env.current_time, 1),
        'wall_seconds': round(time.time() - started, 1),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with out.open('a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        if new:
            w.writeheader()
        w.writerow(row)

    print('\n--- 記録 ---')
    print('  指示: %s (%s)' % (row['instruction'] or 'なし', quality or '-'))
    print('  着手まで: %s秒%s' % (row['wait_seconds'],
                              ' (最後まで着手せず)' if row['wait_censored'] else ''))
    print('  実行順位: %s / 指示なしなら %s 番目 (繰り上がり %s)'
          % (exec_rank, nat, row['rank_gain']))
    print('  提供 %d 品 / 所要 %.1f 秒' % (row['served'], row['makespan_s']))
    print('  -> %s' % out)


if __name__ == '__main__':
    main()
