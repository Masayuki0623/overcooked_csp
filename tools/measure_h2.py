"""H2(効率損失は超過分が正のときだけ生じ、やがて飽和する)を測り直す。

損失 = 指示ありの所要時間 - 指示なしの所要時間
超過分 = max(0, 自然順位 - skip_budget)

指示ありの結果は results/instruction_wait.csv を使う。指示なしの基準値は
ここで測る(注文構成 x 相方 の 36 通り。どちらも決定的なので1回で足りる)。

    python tools/measure_h2.py
"""
import argparse
import collections
import csv
import os
import statistics as st
import sys
from copy import deepcopy as dcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in ('.', 'agent', 'testbed-cooking', 'tools'):
    sys.path.insert(0, str(ROOT / _p))

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

from gym_cooking.utils.order_preset import enumerate_order_recipes  # noqa: E402
from gym_cooking.utils.replay import Replay  # noqa: E402
import run_human_model_experiment as H  # noqa: E402
from human_models import HumanModel  # noqa: E402


def baseline(case, recipes, model):
    """指示を出さずに走らせたときの所要時間。"""
    env = H.make_env('experiment', case, 'experiment2', recipes)
    ai = H.make_ai(0, partner_is_external=(model != 'follow_plan'))
    human = HumanModel(model, ai, 1, Replay(), seed=case * 31 + 7)
    for _step in range(1, 1001):
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
        env.step(acts, passed_time=0.1)
        if not env.order_scheduler.current_orders:
            break
    return env.current_time, len(env.order_scheduler.current_orders) == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='results/instruction_wait.csv')
    ap.add_argument('--out', default='results/h2_loss.csv')
    args = ap.parse_args()

    H.MAX_SECONDS_OVERRIDE = 100.0
    sets = enumerate_order_recipes('experiment2')
    rows = list(csv.DictReader(open(args.csv, encoding='utf-8')))

    base = {}
    for model in ('greedy', 'follow_plan'):
        for case in range(len(sets)):
            ms, done = baseline(case, sets[case], model)
            base[(model, case)] = ms
            if not done:
                print('  ! 基準値が完走せず: case%d %s' % (case, model))
    print('基準値 %d 通りを測定' % len(base))

    out = []
    for r in rows:
        nat = r.get('natural_rank')
        if nat in (None, '', 'None'):
            continue
        d = int(r['skip_budget'])
        excess = max(0, int(float(nat)) - d)
        b = base[(r['human_model'], int(r['seed']))]
        out.append({
            'seed': r['seed'], 'quality': r['quality'], 'skip_budget': d,
            'human_model': r['human_model'], 'natural_rank': int(float(nat)),
            'excess': excess,
            'makespan_baseline_s': round(b, 1),
            'makespan_constrained_s': float(r['makespan_actual_s']),
            'loss_s': round(float(r['makespan_actual_s']) - b, 2),
        })

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print('%d 行 -> %s\n' % (len(out), p))

    by = collections.defaultdict(list)
    for r in out:
        by[r['excess']].append(r['loss_s'])
    print('=== 超過分ごとの損失(秒) ===')
    for e in sorted(by):
        v = by[e]
        print('  超過%d: n=%3d 平均%+6.2f 中央値%+6.2f' % (e, len(v), st.mean(v), st.median(v)))

    zero = [r['loss_s'] for r in out if r['excess'] == 0]
    pos = [r['loss_s'] for r in out if r['excess'] > 0]
    print('\n  超過0   : n=%3d 平均%+6.2f' % (len(zero), st.mean(zero)))
    print('  超過1以上: n=%3d 平均%+6.2f' % (len(pos), st.mean(pos)))


if __name__ == '__main__':
    main()
