"""指示を出した参加者が、客観的にどういう状況に置かれるかを測る。

体験そのものは人に聞くまで分からない。だが「指示したのに、いつまで待たされ、
その間 AI が何をしていたか」は計算できる。本実験の結果を読むときに、
アンケートと突き合わせるための参照データを作る。

ゲーム本体のロジックには触れない。観測を足すだけ。

    python tools/run_instruction_wait_experiment.py --shard 0/8 --out results/wait_0.csv
"""
import argparse
import csv
import os
import random
import sys
import time
from copy import deepcopy as dcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'agent'))
sys.path.insert(0, str(ROOT / 'testbed-cooking'))
sys.path.insert(0, str(ROOT / 'tools'))

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

from gym_cooking.utils.order_preset import enumerate_order_recipes  # noqa: E402
from gym_cooking.utils.replay import Replay  # noqa: E402

import run_human_model_experiment as H  # noqa: E402
from human_models import HumanModel  # noqa: E402

QUALITIES = ('good', 'bad', 'random')
SKIP_BUDGETS = (0, 2, 4)
MAX_STEPS = 1000          # ゲーム内100秒
STEP_SECONDS = 0.1

FIELDS = [
    'seed', 'orders', 'quality', 'skip_budget', 'human_model', 'status',
    'target', 'target_verb', 'target_obj',
    # 3-1 待たされる時間
    'wait_seconds', 'wait_censored',
    # 3-2 待っている間に AI がしていたこと
    'tasks_before', 'verbs_before', 'durations_before',
    'ai_idle_seconds_while_waiting', 'ai_idle_pct_while_waiting',
    # 3-3 実行順位
    'exec_rank', 'natural_rank', 'rank_gain',
    # 指示が AI 以外に割り当たった場合の把握
    'plan_owner', 'wait_any_seconds', 'wait_any_censored', 'done_by_other',
    # 参考
    'served', 'completed', 'makespan_actual_s', 'human_idle_pct', 'wall_seconds',
]


def target_matches(task_id, verb, obj):
    """指示は「玉ねぎを切る」という行動単位。どの注文の分でも一致とみなす。"""
    return bool(task_id) and task_id[0] == verb and task_id[1] == obj


def remaining_matches(ai, verb, obj):
    """その作業がまだ計画に残っているか(どちらの担当かは問わない)。"""
    for agent_idx, sched in (ai.schedule_per_agent or {}).items():
        idx = ai.current_task_idx
        idx = idx.get(agent_idx, 0) if isinstance(idx, dict) else (idx or 0)
        for t in sched[idx:]:
            if target_matches(t.get('id'), verb, obj):
                return True
    return False


def plan_owner_of(ai, verb, obj):
    """その作業を、計画はどちらの担当にしたか。"""
    owners = []
    for agent_idx, sched in sorted((ai.schedule_per_agent or {}).items()):
        if any(target_matches(t.get('id'), verb, obj) for t in sched):
            owners.append('AI' if agent_idx == 0 else '人')
    return '|'.join(owners) if owners else '(計画に無し)'


def current_task_id(ai, agent_idx=0):
    sched = (ai.schedule_per_agent or {}).get(agent_idx) or []
    idx = ai.current_task_idx
    idx = idx.get(agent_idx, 0) if isinstance(idx, dict) else (idx or 0)
    if 0 <= idx < len(sched):
        return sched[idx].get('id')
    return None


def natural_rank_of(env_state, verb, obj, skip_budget):
    """指示を出さなかった場合、その作業は AI の何番目になるはずだったか。

    同じ初期状態を指示なしで解き、AI 側の計画の並びを見る。
    """
    ai = H.make_ai(skip_budget, partner_is_external=True)
    ai(dcopy(env_state))
    sched = (ai.schedule_per_agent or {}).get(0) or []
    for i, t in enumerate(sched, 1):
        if target_matches(t.get('id'), verb, obj):
            return i, len(sched)
    return None, len(sched)


def run_trial(case, recipes, quality, skip_budget, human_model='greedy'):
    started = time.time()
    env = H.make_env('experiment', case, 'experiment2', recipes)
    ai = H.make_ai(skip_budget, partner_is_external=(human_model != 'follow_plan'))
    human_idx = 1
    human = HumanModel(human_model, ai, human_idx, Replay(), seed=case * 31 + 7)

    row = {f: None for f in FIELDS}
    row.update({'seed': case, 'orders': '|'.join(recipes), 'quality': quality,
                'skip_budget': skip_budget, 'human_model': human_model})

    state = H.state_for(env, 0)
    orders = ai._build_order_tasks(dcopy(state))
    if not any(o['tasks'] for o in orders):
        row['status'] = 'no_tasks'
        return row

    rng = random.Random(f'experiment-{case}-{quality}')
    picked = H.pick_instruction(ai, state, orders, quality, rng)
    if picked is None:
        row['status'] = 'no_candidate'
        return row
    display, payload = picked
    verb, obj = payload['verb'], payload['obj']
    row.update({'target': display, 'target_verb': verb, 'target_obj': obj})

    # 指示なしで解いたときの並び(先に取る。指示を積む前の状態で測るため)
    nat_rank, _nat_len = natural_rank_of(state, verb, obj, skip_budget)
    row['natural_rank'] = nat_rank

    pending = {
        'id': float(case), 'task': payload, 'target_idx': 0,
        'accepted_env_time': 0.0, 'status': 'pending',
        'skip_budget': skip_budget, 'remaining_skip_budget': skip_budget,
    }
    env._pending_instructions = [dcopy(pending)]
    ai._pending_instructions = [dcopy(pending)]

    # --- 観測 -----------------------------------------------------------
    wait_seconds = None            # 指示 -> AI が着手 までの経過時間
    wait_any_seconds = None        # 指示 -> 誰かがやり終える までの経過時間
    plan_owner = None              # 計画がその作業をどちらに割り当てたか
    finished_before = []           # 着手までに AI がこなした他タスク
    ai_idle_ticks = 0              # 待っている間、AI が動かなかった回数
    ticks_while_waiting = 0
    prev_task = None
    prev_task_start = 0.0

    for _step in range(1, MAX_STEPS + 1):
        move, _ = ai(dcopy(H.state_for(env, 0)))

        cur = current_task_id(ai, 0)
        now = env.current_time
        if plan_owner is None:
            plan_owner = plan_owner_of(ai, verb, obj)
        if wait_any_seconds is None and not remaining_matches(ai, verb, obj):
            # 計画から消えた = 誰かがやり終えた
            wait_any_seconds = round(now, 1)

        if wait_seconds is None:
            ticks_while_waiting += 1
            if cur != prev_task:
                # 直前までやっていた作業が一区切りついた
                if prev_task is not None and not target_matches(prev_task, verb, obj):
                    finished_before.append((prev_task, round(now - prev_task_start, 1)))
                prev_task, prev_task_start = cur, now
            if target_matches(cur, verb, obj):
                wait_seconds = round(now, 1)

        actions = {a.name: (0, 0) for a in env.sim_agents}
        own = move.get('ai_0') if isinstance(move, dict) else move
        if own:
            actions[env.sim_agents[0].name] = own
        if wait_seconds is None and (not own or tuple(own) == (0, 0)):
            ai_idle_ticks += 1

        if human_model == 'follow_plan':
            h_action = move.get('ai_1') if isinstance(move, dict) else (0, 0)
        else:
            h_action, _tid = human.act(H.state_for(env, human_idx),
                                       env.sim_agents[0].location)
        actions[env.sim_agents[human_idx].name] = h_action or (0, 0)
        human.record(H.state_for(env, human_idx), h_action or (0, 0))

        env.step(actions, passed_time=STEP_SECONDS)
        if not env.order_scheduler.current_orders:
            break

    sched = env.order_scheduler
    row.update({
        'wait_seconds': wait_seconds if wait_seconds is not None else round(env.current_time, 1),
        'wait_censored': int(wait_seconds is None),
        'tasks_before': len(finished_before),
        'verbs_before': '|'.join(t[0][0] for t in finished_before),
        'durations_before': '|'.join(str(t[1]) for t in finished_before),
        'ai_idle_seconds_while_waiting': round(ai_idle_ticks * STEP_SECONDS, 1),
        'ai_idle_pct_while_waiting': (round(100 * ai_idle_ticks / ticks_while_waiting, 1)
                                      if ticks_while_waiting else None),
        'plan_owner': plan_owner,
        'wait_any_seconds': (wait_any_seconds if wait_any_seconds is not None
                             else round(env.current_time, 1)),
        'wait_any_censored': int(wait_any_seconds is None),
        # AI は着手しなかったが、作業自体は片づいた = 相手がやった
        'done_by_other': int(wait_seconds is None and wait_any_seconds is not None),
        'exec_rank': len(finished_before) + 1 if wait_seconds is not None else None,
        'rank_gain': ((nat_rank - (len(finished_before) + 1))
                      if (nat_rank is not None and wait_seconds is not None) else None),
        'served': sched.successful_orders,
        'completed': int(len(sched.current_orders) == 0),
        'makespan_actual_s': round(env.current_time, 1),
        **{'human_idle_pct': human.time_breakdown().get('human_idle_pct')},
        'wall_seconds': round(time.time() - started, 1),
        'status': 'ok',
    })
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--human-model', default='greedy')
    ap.add_argument('--cases', type=int, default=None, help='先頭から何通りだけ使うか(下見用)')
    ap.add_argument('--qualities', default=','.join(QUALITIES))
    ap.add_argument('--budgets', default=','.join(str(d) for d in SKIP_BUDGETS))
    ap.add_argument('--shard', default=None, help='"i/n" 形式')
    ap.add_argument('--out', default=str(ROOT / 'results' / 'instruction_wait.csv'))
    args = ap.parse_args()

    print(f'表示: なし(画面なしで実行します)  '
          f'[SDL_VIDEODRIVER={os.environ.get("SDL_VIDEODRIVER")}]')
    print('様子を目で見たいときは tools/watch_human_model.py を使ってください。')

    sets = enumerate_order_recipes('experiment2')
    if args.cases:
        sets = sets[:args.cases]
    qualities = [q.strip() for q in args.qualities.split(',') if q.strip()]
    budgets = [int(d) for d in args.budgets.split(',') if d.strip()]

    combos = [(c, q, d) for c in range(len(sets)) for q in qualities for d in budgets]
    if args.shard:
        i, n = (int(x) for x in args.shard.split('/'))
        combos = [c for k, c in enumerate(combos) if k % n == i]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for k, (case, quality, budget) in enumerate(combos, 1):
        rows.append(run_trial(case, sets[case], quality, budget, args.human_model))
        if k % 5 == 0 or k == len(combos):
            el = time.time() - t0
            print(f'  {k}/{len(combos)} 件 ({el:.0f}秒経過, '
                  f'残り約{el / k * (len(combos) - k):.0f}秒)', flush=True)

    with out.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f'\n{len(rows)} 試行 -> {out}')
    print(f'所要 {time.time() - t0:.0f} 秒')


if __name__ == '__main__':
    main()
