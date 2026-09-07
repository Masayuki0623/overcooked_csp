"""少数の試行で、完走と停止をまとめて確かめる。

正式な測定(324試行)は時間がかかるので、実装をいじるたびに回すには重い。
注文構成18通りを1回ずつ、指示なしで走らせて、完走したか・二人とも
止まった時間がなかったかを見る。数分で終わる。

    python tools/quick_check.py                 # 貪欲と最適な相方の両方
    python tools/quick_check.py --model greedy  # 片方だけ
    python tools/quick_check.py --shard 0/4     # 並列に分けて回す
"""
import argparse
import os
import sys
import time
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
from detect_freeze import snapshot, pot_is_cooking  # noqa: E402

STEP = 0.1


def run_trial(case, recipes, model, min_freeze_s):
    env = H.make_env('experiment', case, 'experiment2', recipes)
    ai = H.make_ai(0, partner_is_external=(model != 'follow_plan'))
    human = HumanModel(model, ai, 1, Replay(), seed=case * 31 + 7)
    pot_locs = H.state_for(env, 0).get_pos_by_obj_gs(gs='Pot')

    prev = snapshot(env)
    run = 0
    worst = 0.0
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
        if now == prev and not pot_is_cooking(H.state_for(env, 0), pot_locs):
            run += 1
            worst = max(worst, run * STEP)
        else:
            run = 0
        prev = now
        if not env.order_scheduler.current_orders:
            break

    sched = env.order_scheduler
    return {
        'case': case,
        'served': sched.successful_orders,
        'completed': len(sched.current_orders) == 0,
        'makespan': env.current_time,
        'freeze': worst if worst >= min_freeze_s else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=None, help='省略すると両方')
    ap.add_argument('--shard', default=None, help='"i/n" 形式')
    ap.add_argument('--max-seconds', type=float, default=100.0)
    ap.add_argument('--min-freeze', type=float, default=3.0)
    args = ap.parse_args()

    H.MAX_SECONDS_OVERRIDE = args.max_seconds
    models = [args.model] if args.model else ['greedy', 'follow_plan']
    sets = enumerate_order_recipes('experiment2')
    cases = list(range(len(sets)))
    if args.shard:
        i, n = (int(x) for x in args.shard.split('/'))
        cases = [c for c in cases if c % n == i]

    t0 = time.time()
    for model in models:
        rows = [run_trial(c, sets[c], model, args.min_freeze) for c in cases]
        ok = sum(1 for r in rows if r['completed'])
        frozen = [r for r in rows if r['freeze']]
        print('%s: 完走 %d/%d  停止あり %d件  所要 平均%.1f秒'
              % (model, ok, len(rows), len(frozen),
                 sum(r['makespan'] for r in rows) / len(rows)))
        for r in rows:
            if not r['completed'] or r['freeze']:
                print('    case%2d 提供%d %.1f秒%s'
                      % (r['case'], r['served'], r['makespan'],
                         '  停止%.1f秒' % r['freeze'] if r['freeze'] else ''))
    print('所要 %.0f 秒' % (time.time() - t0))


if __name__ == '__main__':
    main()
