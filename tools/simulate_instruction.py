# -*- coding: utf-8 -*-
"""最悪の指示を実際に出して、最後まで遊ばせ、所要時間を測る。

予測値 L ではなく、その指示を守らせた結果どれだけ遅くなったかを見る。

  1. 指示なしで最後まで遊ばせる            -> 基準の所要時間
  2. 指示の時刻まで進め、そこで候補を全部評価  -> L が最大の作業を選ぶ
  3. その指示を付けて残りを遊ばせる          -> 実際の所要時間

    python tools/simulate_instruction.py --orders A,B,C --cook 25 --at 10 \
        --budgets 0,1,2,4
"""
import argparse
import contextlib
import io
import os
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in ('.', 'agent', 'testbed-cooking'):
    sys.path.insert(0, str(ROOT / _p))
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

from gym_cooking.utils.interact import resolve_action  # noqa: E402
from tools.design_probe import set_cook_time  # noqa: E402
from tools.bench_orders import build  # noqa: E402
from tools.selfplay_probe import state_for, make_agent, menu_at  # noqa: E402

MAX_SECONDS = 200.0


def play(env, driver, until=None, pending=None):
    """最後まで(または until 秒まで)遊ばせる。"""
    names = [a.name for a in env.sim_agents]
    n_orders = len(env.order_scheduler.current_orders) + env.order_scheduler.successful_orders
    while env.current_time < (until if until is not None else MAX_SECONDS):
        with contextlib.redirect_stdout(io.StringIO()):
            move, _ = driver(state_for(env, 0))
        acts = {}
        for n, key in ((names[0], 'ai_0'), (names[1], 'ai_1')):
            agent = next(a for a in env.sim_agents if a.name == n)
            acts[n] = tuple(resolve_action(agent, env.world,
                                           tuple(move.get(key) or (0, 0))))
        env.step(acts, passed_time=0.2)
        if env.order_scheduler.successful_orders >= n_orders:
            break
    return env.current_time, env.order_scheduler.successful_orders


def pick_worst(env, budget):
    """その瞬間に出せる指示のうち、L が最大で縛りも成立するもの。"""
    rows = [r for r in menu_at(env, budget, None)
            if r['bound'] and r['loss'] is not None]
    if not rows:
        return None
    return max(rows, key=lambda r: r['loss'])


def run_one(map_name, recipes, cook, at, budget, worst=True):
    set_cook_time(cook)
    env = build(map_name, recipes)
    driver = make_agent(0, False, None, 0.5)
    play(env, driver, until=at)
    rows = [r for r in menu_at(env, budget, None)
            if r['bound'] and r['loss'] is not None]
    if not rows:
        return None
    pick = (max if worst else min)(rows, key=lambda r: r['loss'])
    verb, obj = pick['task'].split(' ', 1)
    pending = {'id': time.time(), 'task': {'verb': verb, 'obj': obj},
               'target_idx': 0, 'status': 'pending',
               'skip_budget': budget, 'remaining_skip_budget': budget,
               'tasks_before_target_log': [], 'execution_logged': False,
               'deadline_constraint_applied': False}
    driver.skip_budget = budget
    driver._pending_instructions = [pending]
    if hasattr(driver, '_mark_reschedule_needed'):
        driver._mark_reschedule_needed('instruction_accepted')
    t, served = play(env, driver)
    return {'task': pick['task'], 'L': pick['loss'], 'time': t, 'served': served}


def run_base(map_name, recipes, cook):
    set_cook_time(cook)
    env = build(map_name, recipes)
    driver = make_agent(0, False, None, 0.5)
    t, served = play(env, driver)
    return {'time': t, 'served': served}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', default='exp_ring')
    ap.add_argument('--sets', default=None,
                    help='注文セットを ";" 区切りで。省略時はスープ2+ジュースを3通り')
    ap.add_argument('--cook', type=int, default=25)
    ap.add_argument('--at', type=float, default=10.0)
    ap.add_argument('--budgets', default='0,1,2,4')
    args = ap.parse_args()

    if args.sets:
        sets = [s.split(',') for s in args.sets.split(';')]
    else:
        sets = [['OnionTomatoSoup', 'TomatoLettuceSoup', 'AppleOrangeJuice'],
                ['OnionTomatoSoup', 'OnionLettuceSoup', 'AppleBananaJuice'],
                ['TomatoLettuceSoup', 'OnionLettuceSoup', 'BananaOrangeJuice']]
    budgets = [int(x) for x in args.budgets.split(',')]

    print('地図=%s / 煮込み=%d秒 / 指示は%.0f秒時点' % (args.map, args.cook, args.at))
    bases = {}
    for rec in sets:
        b = run_base(args.map, rec, args.cook)
        bases[tuple(rec)] = b['time']
        print('  指示なし %-46s %5.1f秒 (提供%d)'
              % ('/'.join(r[:14] for r in rec), b['time'], b['served']))
        sys.stdout.flush()

    print()
    print(' %-3s %-46s %6s %8s %8s' % ('d', '選ばれた最悪の指示', 'L予測', '所要', '基準比'))
    summary = {}
    for d in budgets:
        diffs, Ls, times = [], [], []
        for rec in sets:
            r = run_one(args.map, rec, args.cook, args.at, d, worst=True)
            if r is None:
                print(' %-3d %-46s  (出せる指示なし)' % (d, '/'.join(r0[:10] for r0 in rec)))
                continue
            base = bases[tuple(rec)]
            diffs.append(r['time'] - base)
            Ls.append(r['L'])
            times.append(r['time'])
            print(' %-3d %-46s %6.1f %7.1f秒 %+7.1f秒' %
                  (d, r['task'][:46], r['L'], r['time'], r['time'] - base))
            sys.stdout.flush()
        if diffs:
            summary[d] = (st.mean(Ls), st.mean(times), st.mean(diffs))

    print()
    print(' %-3s %8s %10s %10s' % ('d', 'L予測平均', '所要の平均', '基準との差'))
    for d, (L, t, diff) in sorted(summary.items()):
        print(' %-3d %7.1f秒 %9.1f秒 %+9.1f秒' % (d, L, t, diff))


if __name__ == '__main__':
    main()
