"""刻んだ材料が最初からあちこちの台に置かれている状態から始めて、完走を見る。

参加者は「切ったものを手近な台に置く」。1つの注文の材料が別々の台に散って
いても、まとめて鍋へ入れる・皿に盛るところまで運べなければならない。
実際のプレイで起きるのはこの形の乱雑さなので、回帰試験として残しておく。

    python tools/scatter_check.py                  # 18構成 x 6回
    python tools/scatter_check.py --place 6        # 散らかす数を増やす
    python tools/scatter_check.py --shard 0/8      # 並列に分けて回す
    python tools/scatter_check.py --only 15:1      # 1試行だけ再現する
"""
import argparse
import os
import random
import sys
from copy import deepcopy as dcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in ('.', 'agent', 'testbed-cooking', 'tools'):
    sys.path.insert(0, str(ROOT / _p))

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

from gym_cooking.utils.core import (  # noqa: E402
    Apple, Banana, Lettuce, Object, Onion, Orange, Tomato)
from gym_cooking.utils.order_preset import (  # noqa: E402
    enumerate_order_recipes, experiment_case_indices)
from gym_cooking.utils.replay import Replay  # noqa: E402
import run_human_model_experiment as H  # noqa: E402
from human_models import HumanModel  # noqa: E402
from detect_freeze import snapshot, pot_is_cooking  # noqa: E402

FOODS = {'lettuce': Lettuce, 'onion': Onion, 'tomato': Tomato,
         'apple': Apple, 'orange': Orange, 'banana': Banana}
STEP = 0.1


def run_trial(case, recipes, model, rng, n_place, min_freeze_s):
    env = H.make_env('experiment', case, 'experiment2', recipes)
    ai = H.make_ai(0, partner_is_external=(model != 'follow_plan'))
    human = HumanModel(model, ai, 1, Replay(), seed=case * 31 + 7)

    state = H.state_for(env, 0)
    # 手の届かない台(四方を台に囲まれた角)は、参加者にも置けない。
    free = [c for c in state.get_pos_by_obj_gs(gs='Counter')
            if state.pos_obj.get(c) is None and ai._components_touching(state, tuple(c))]

    ings = []
    for order in ai._build_order_tasks(dcopy(state)):
        ings += list(order['ingredients'])
    rng.shuffle(ings)
    placed = []
    for ing in ings[:n_place]:
        if ing not in FOODS or not free:
            continue
        pos = free.pop(rng.randrange(len(free)))
        env.world.insert(Object(location=pos, contents=[FOODS[ing](state_index=2)]))
        placed.append('%s@%s' % (ing, pos))

    pot_locs = state.get_pos_by_obj_gs(gs='Pot')
    prev = snapshot(env)
    run = 0
    worst = 0.0
    for _step in range(int(H.MAX_SECONDS_OVERRIDE / STEP)):
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
        'served': sched.successful_orders,
        'completed': not sched.current_orders,
        'makespan': env.current_time,
        'freeze': worst if worst >= min_freeze_s else 0.0,
        'placed': placed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='messy', help='人間役の方針')
    ap.add_argument('--trials', type=int, default=6, help='1構成あたりの回数')
    ap.add_argument('--place', type=int, default=4, help='最初に散らかす数')
    ap.add_argument('--shard', default=None, help='"i/n" 形式')
    ap.add_argument('--only', default=None, help='"case:回" で1試行だけ')
    ap.add_argument('--max-seconds', type=float, default=100.0)
    ap.add_argument('--min-freeze', type=float, default=3.0)
    args = ap.parse_args()

    H.MAX_SECONDS_OVERRIDE = args.max_seconds
    sets = enumerate_order_recipes('experiment2')
    combos = [(c, k) for c in experiment_case_indices('experiment2')
              for k in range(args.trials)]
    if args.only:
        case, k = (int(x) for x in args.only.split(':'))
        combos = [(case, k)]
    if args.shard:
        i, n = (int(x) for x in args.shard.split('/'))
        combos = [c for j, c in enumerate(combos) if j % n == i]

    ok = 0
    for case, k in combos:
        r = run_trial(case, sets[case], args.model,
                      random.Random(case * 977 + k), args.place, args.min_freeze)
        ok += int(r['completed'])
        bad = (not r['completed']) or r['freeze']
        print('case%2d #%d 提供%d %s %.1f秒%s  置き=%s%s'
              % (case, k, r['served'], 'OK' if r['completed'] else '未完',
                 r['makespan'], '  停止%.1f秒' % r['freeze'] if r['freeze'] else '',
                 ','.join(r['placed']), '   <<<' if bad else ''), flush=True)
    print('完走 %d/%d' % (ok, len(combos)))


if __name__ == '__main__':
    main()
