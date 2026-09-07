"""両者が同時に止まっている時間を洗い出す。

完走していても、二人とも動かない時間があれば協調としては破綻している。
参加者から見れば「二人とも突っ立っている」時間なので、体験を測る前に
消しておきたい。位置と持ち物のどちらも変わらないフレームを「停止」と
数え、それが閾値以上続いた区間を記録する。

    python tools/detect_freeze.py --shard 0/4 --out results/freeze_0.csv
"""
import argparse
import collections
import csv
import os
import random
import sys
import time
from copy import deepcopy as dcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in ('.', 'agent', 'testbed-cooking', 'tools'):
    sys.path.insert(0, str(ROOT / _p))

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

from gym_cooking.utils.order_preset import (  # noqa: E402
    enumerate_order_recipes, experiment_case_indices)
from gym_cooking.utils.replay import Replay  # noqa: E402
import run_human_model_experiment as H  # noqa: E402
from human_models import HumanModel  # noqa: E402
from agent.myagent.TaskAgent import TaskAgent  # noqa: E402

QUALITIES = ('good', 'bad', 'random')
SKIP_BUDGETS = (0, 2, 4)
STEP = 0.1

FIELDS = ['seed', 'orders', 'quality', 'skip_budget', 'human_model', 'served',
          'completed', 'makespan_s', 'freeze_count', 'freeze_max_s',
          'freeze_total_s', 'worst_at_s', 'worst_ai', 'worst_human']


def snapshot(env):
    """両者の位置と持ち物。どちらも変わらなければ「止まっている」。"""
    out = []
    for a in env.sim_agents:
        held = getattr(a, 'holding', None)
        out.append((tuple(a.location), getattr(held, 'full_name', None)))
    return tuple(out)


def pot_is_cooking(state, pot_locs):
    """鍋で調理が進んでいるか。

    調理は15秒かかり、その間は誰も手を出せない。二人が止まっていても
    それは待つのが正しい時間なので、停止として数えない。
    """
    for pos in pot_locs:
        obj = state.pos_obj.get(pos)
        if obj is not None and 'Cooking' in (getattr(obj, 'full_name', '') or ''):
            return True
    return False


def run_trial(case, recipes, quality, budget, model, min_freeze_s):
    env = H.make_env('experiment', case, 'experiment2', recipes)
    ai = H.make_ai(budget, partner_is_external=(model != 'follow_plan'))
    human = HumanModel(model, ai, 1, Replay(), seed=case * 31 + 7)

    state = H.state_for(env, 0)
    orders = ai._build_order_tasks(dcopy(state))
    picked = H.pick_instruction(ai, state, orders, quality,
                                random.Random('experiment-%d-%s' % (case, quality)))
    if picked is not None:
        pend = {'id': float(case), 'task': picked[1], 'target_idx': 0,
                'accepted_env_time': 0.0, 'status': 'pending',
                'skip_budget': budget, 'remaining_skip_budget': budget}
        shared = [pend]
        env._pending_instructions = shared
        ai._pending_instructions = shared

    # 止まっている間、二人が何をしていたつもりなのかを残す
    last_reason = {}
    orig_call = TaskAgent.__call__

    def spy(self, e, **kw):
        action, reason = orig_call(self, e, **kw)
        last_reason[getattr(e, 'agent_idx', None)] = '%s :: %s' % (self.task_name, reason)
        return action, reason

    TaskAgent.__call__ = spy
    freezes = []
    run_start = 0.0
    pot_locs = H.state_for(env, 0).get_pos_by_obj_gs(gs='Pot')
    try:
        prev = snapshot(env)
        run = 0
        run_reason = ('', '')
        for _step in range(1, int(H.MAX_SECONDS_OVERRIDE / STEP) + 1):
            move, _ = ai(dcopy(H.state_for(env, 0)))
            acts = {a.name: (0, 0) for a in env.sim_agents}
            own = move.get('ai_0') if isinstance(move, dict) else move
            if own:
                acts[env.sim_agents[0].name] = own
            if model == 'follow_plan':
                ha = move.get('ai_1') if isinstance(move, dict) else (0, 0)
            else:
                ha, _ = human.act(H.state_for(env, 1), env.sim_agents[0].location)
            acts[env.sim_agents[1].name] = ha or (0, 0)
            human.record(H.state_for(env, 1), ha or (0, 0))
            env.step(acts, passed_time=STEP)

            now = snapshot(env)
            # 調理中は待つのが正しいので、停止として数えない。
            cooking = pot_is_cooking(H.state_for(env, 0), pot_locs)
            if now == prev and not cooking:
                if run == 0:
                    run_reason = (last_reason.get(0, ''), last_reason.get(1, ''))
                    run_start = env.current_time
                run += 1
            else:
                if run * STEP >= min_freeze_s:
                    freezes.append((run * STEP, run_start, run_reason[0], run_reason[1]))
                run = 0
            prev = now
            if not env.order_scheduler.current_orders:
                break
        if run * STEP >= min_freeze_s:
            freezes.append((run * STEP, run_start, run_reason[0], run_reason[1]))
    finally:
        TaskAgent.__call__ = orig_call

    sched = env.order_scheduler
    worst = max(freezes, default=None)
    return {
        'seed': case, 'orders': '|'.join(recipes), 'quality': quality,
        'skip_budget': budget, 'human_model': model,
        'served': sched.successful_orders,
        'completed': int(len(sched.current_orders) == 0),
        'makespan_s': round(env.current_time, 1),
        'freeze_count': len(freezes),
        'freeze_max_s': round(worst[0], 1) if worst else 0.0,
        'freeze_total_s': round(sum(f[0] for f in freezes), 1),
        'worst_at_s': round(worst[1], 1) if worst else '',
        'worst_ai': worst[2] if worst else '',
        'worst_human': worst[3] if worst else '',
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--human-model', default=None,
                    help='省略すると greedy と follow_plan の両方')
    ap.add_argument('--shard', default=None, help='"i/n" 形式')
    ap.add_argument('--cases', default='all', choices=['experiment', 'all'],
                    help="'experiment' は本実験で使う構成だけ")
    ap.add_argument('--max-seconds', type=float, default=100.0)
    ap.add_argument('--min-freeze', type=float, default=3.0,
                    help='これ以上続いた停止を異常として数える(秒)')
    ap.add_argument('--out', default=str(ROOT / 'results' / 'freeze.csv'))
    args = ap.parse_args()

    H.MAX_SECONDS_OVERRIDE = args.max_seconds
    models = [args.human_model] if args.human_model else ['greedy', 'follow_plan']
    sets = enumerate_order_recipes('experiment2')
    cases = (experiment_case_indices('experiment2') if args.cases == 'experiment'
             else list(range(len(sets))))
    combos = [(c, q, d, m) for m in models for c in cases
              for q in QUALITIES for d in SKIP_BUDGETS]
    if args.shard:
        i, n = (int(x) for x in args.shard.split('/'))
        combos = [c for k, c in enumerate(combos) if k % n == i]

    rows = []
    t0 = time.time()
    for k, (case, quality, budget, model) in enumerate(combos, 1):
        rows.append(run_trial(case, sets[case], quality, budget, model, args.min_freeze))
        el = time.time() - t0
        print('  %d/%d 件 (%.0f秒経過, 残り約%.0f秒)'
              % (k, len(combos), el, el / k * (len(combos) - k)), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    bad = [r for r in rows if r['freeze_count']]
    print('\n%d 試行 -> %s' % (len(rows), out))
    print('%.1f秒以上の停止があった試行: %d/%d' % (args.min_freeze, len(bad), len(rows)))
    if bad:
        print('最長 %.1f秒' % max(r['freeze_max_s'] for r in bad))
        c = collections.Counter(r['worst_ai'][:46] for r in bad)
        for k2, v in c.most_common(5):
            print('  AI  %3d件 %s' % (v, k2))
        c = collections.Counter(r['worst_human'][:46] for r in bad)
        for k2, v in c.most_common(5):
            print('  人  %3d件 %s' % (v, k2))


if __name__ == '__main__':
    main()
