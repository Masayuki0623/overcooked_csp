"""鎖の指示について、d ごと・鎖の深さごとの効率損失量 L を調べる。

本実験で使う注文構成(地図2種)の開始時点で、指示の候補すべてについて
    f      : 指示なしの makespan
    f'(d)  : d = 0 / 1 / 2 で縛ったときの makespan
    L(d)   = f'(d) - f     (L(0) が即時実行の効率損失量 L0)
を解いて CSV に出す。

実行方法:
    python tools/survey_chain_loss.py [出力CSV] [--maps exp_ring,exp_partition] [--limit N]
"""
import csv
import os
import sys
import time
from copy import deepcopy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from gym_cooking.utils.order_preset import enumerate_order_recipes, experiment_case_indices
from gym_cooking.utils.replay import Replay
from agent.agent.executor.low import EnvState
from agent.agent.myagent.CSPAgent import CSPAgent

PRESET = {'exp_ring': 'experiment1', 'exp_partition': 'experiment2'}
BUDGETS = (None, 0, 1, 2)


def state_of(env):
    i = env.get_ai_info()
    return EnvState(world=i['world'], agents=i['sim_agents'], agent_idx=0,
                    order=i['order_scheduler'], event_history=i['event_history'],
                    time=i['current_time'], chg_grid=i['chg_grid'])


def new_ai(budget):
    ai = CSPAgent(10, Replay(), sc_2agent=True, skip_budget=budget)
    ai.human_counterpart_mode = True
    ai.own_agent_idx = 0
    ai.priority_weights = {}
    ai.gui_text_input = ''
    ai.gui_constraint_input = ''
    ai.active_constraints = []
    ai.solve_deterministic_limit = None
    ai.time_limit_seconds = None
    ai.replay = None
    return ai


def solve(state, payload, budget, orders):
    probe = new_ai(budget)
    st = deepcopy(state)
    pend = {'task': (payload['verb'], deepcopy(payload)), 'status': 'pending', 'id': 'x',
            'skip_budget': budget, 'remaining_skip_budget': budget, 'target_idx': 0}
    probe._pending_instructions = [pend]
    st._pending_instructions = [pend]
    probe.solve_csp_scheduling(st, orders=deepcopy(orders))
    m = dict(getattr(probe, '_last_solve_metrics', {}) or {})
    fr = m.get('makespan_frames')
    return (None if fr is None else fr / float(probe.fps)), m.get('status')


def depth_of(payload):
    groups = payload.get('chains') or [payload.get('chain') or []]
    return max((len(g) for g in groups), default=0)


def main():
    args = [a for a in sys.argv[1:]]
    out_path = next((a for a in args if a.endswith('.csv')), 'survey_chain_loss.csv')
    maps = list(PRESET)
    limit = None
    if '--maps' in args:
        maps = args[args.index('--maps') + 1].split(',')
    if '--limit' in args:
        limit = int(args[args.index('--limit') + 1])
    rows = []
    t0 = time.time()
    jobs = []
    for m in maps:
        cases = experiment_case_indices(PRESET[m]) or []
        if limit:
            cases = cases[:limit]
        jobs += [(m, c) for c in cases]
    for n, (m, case) in enumerate(jobs, 1):
        sets = enumerate_order_recipes(PRESET[m])
        recipes = tuple(sets[case])
        env = OvercookedEnvironment(MapSetting(level=m, order_recipes=recipes))
        env.reset()
        state = state_of(env)
        ai = new_ai(0)
        cands = ai.get_instruction_candidates(deepcopy(state))
        orders = new_ai(None)._build_order_tasks(deepcopy(state))
        f, _ = solve(state, cands[0][1], None, orders) if cands else (None, None)
        print(f'[{n}/{len(jobs)}] {m} case={case} {recipes} 候補={len(cands)} f={f}  '
              f'経過 {time.time() - t0:.0f}s', flush=True)
        for display, payload in cands:
            row = {'地図': m, '注文構成': case, '注文': '+'.join(recipes),
                   '指示': display, '動詞': payload['verb'], '対象': payload['obj'],
                   '個数': payload.get('count', 1), '同じ工程の数': payload.get('total', 1),
                   '今すぐできるか': int(bool(payload.get('startable'))),
                   '鎖の深さ': depth_of(payload), 'f_秒': f}
            for b in (0, 1, 2):
                fb, status = solve(state, payload, b, orders)
                row[f"f'({b})_秒"] = fb
                row[f'L({b})_秒'] = (None if fb is None or f is None else round(fb - f, 2))
                if fb is None:
                    row[f'L({b})_秒'] = f'({status})'
            rows.append(row)
            print(f"    {display:<34} 深さ={row['鎖の深さ']} L0={row['L(0)_秒']} L1={row['L(1)_秒']} L2={row['L(2)_秒']}", flush=True)
    cols = list(rows[0].keys()) if rows else []
    with open(out_path, 'w', encoding='utf-8-sig', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f'書き出し: {out_path} ({len(rows)} 行, {time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
