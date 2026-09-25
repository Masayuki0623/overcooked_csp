# -*- coding: utf-8 -*-
"""注文の数を増やしたとき、CSP の計算時間がどうなるかを測る。

計画を組み立てる時間(build)と探索する時間(search)を分けて出す。
AI は 1回の判断ごとにこれを解くので、ここが伸びると手が止まる。

    python tools/bench_orders.py --counts 3,4,5,6 --repeat 3
"""
import argparse
import contextlib
import io
import itertools
import os
import random
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in ('.', 'agent', 'testbed-cooking'):
    sys.path.insert(0, str(ROOT / _p))
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

from tools.design_probe import probe_agent, state_of  # noqa: E402

RECIPES = ['TomatoLettuceSalad', 'OnionTomatoSalad', 'OnionLettuceSalad',
           'TomatoLettuceSoup', 'OnionTomatoSoup', 'OnionLettuceSoup',
           'AppleOrangeJuice', 'AppleBananaJuice', 'BananaOrangeJuice']


def build(map_name, recipes):
    from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
    from gym_cooking.play_test import MAP_SETTINGS
    kw = dict(MAP_SETTINGS[map_name])
    kw['order_recipes'] = tuple(recipes)
    kw['max_num_orders'] = len(recipes)
    env = OvercookedEnvironment(MapSetting(**kw))
    env.reset()
    return env


def one(map_name, recipes, budget=0):
    env = build(map_name, recipes)
    st_ = state_of(env)
    ai = probe_agent(budget)
    with contextlib.redirect_stdout(io.StringIO()):
        ai(st_)                      # 立ち上げ(計測しない)
        orders = ai._build_order_tasks(st_)
        t0 = time.perf_counter()
        ai.solve_csp_scheduling(st_, orders=orders)
        wall = (time.perf_counter() - t0) * 1000
    m = dict(getattr(ai, '_last_solve_metrics', {}) or {})
    ntask = sum(len(o.get('tasks') or []) for o in orders)
    return {'wall': wall, 'build': m.get('build_ms'), 'search': m.get('search_ms'),
            'status': m.get('status'), 'tasks': ntask}


def loss_cost(map_name, recipes, budget=0):
    """指示画面を出すのに要る計算(候補1つあたり)。"""
    env = build(map_name, recipes)
    st_ = state_of(env)
    ai = probe_agent(budget)
    with contextlib.redirect_stdout(io.StringIO()):
        ai(st_)
        cands = []
        for c in ai.get_instruction_candidates(st_):
            p = c[1] if isinstance(c, (list, tuple)) and len(c) >= 2 else c
            if isinstance(p, dict):
                cands.append((str(p.get('verb')), str(p.get('obj'))))
        times = []
        for verb, obj in cands:
            pend = {'id': time.time(), 'task': {'verb': verb, 'obj': obj},
                    'target_idx': 0, 'status': 'pending',
                    'skip_budget': budget, 'remaining_skip_budget': budget}
            t0 = time.perf_counter()
            ai.estimate_instruction_time_loss(st_, pend, skip_budget=budget)
            times.append((time.perf_counter() - t0) * 1000)
    return cands, times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--counts', default='3,4,5,6')
    ap.add_argument('--repeat', type=int, default=3)
    ap.add_argument('--maps', default='exp_ring,exp_partition')
    ap.add_argument('--menu', action='store_true', help='指示画面の計算も測る')
    args = ap.parse_args()
    counts = [int(x) for x in args.counts.split(',')]
    rng = random.Random(0)

    print('%-14s %-4s %-6s %10s %10s %10s  %s'
          % ('地図', '注文', '作業数', '合計(ms)', '組立(ms)', '探索(ms)', '状態'))
    for map_name in args.maps.split(','):
        for n in counts:
            walls, builds, searches, tasks, sts = [], [], [], [], []
            for _ in range(args.repeat):
                rec = rng.sample(RECIPES, n)
                try:
                    r = one(map_name, rec)
                except Exception as e:
                    print('%-14s %-4d 失敗 %s' % (map_name, n, e))
                    break
                print('    %-14s 注文%d 作業%2d  %8.0fms (組立%.0f/探索%.0f) %s'
                      % (map_name, n, r['tasks'], r['wall'],
                         r['build'] or 0, r['search'] or 0, r['status']))
                sys.stdout.flush()
                walls.append(r['wall'])
                if r['build'] is not None:
                    builds.append(r['build'])
                if r['search'] is not None:
                    searches.append(r['search'])
                tasks.append(r['tasks'])
                sts.append(str(r['status']))
            if not walls:
                continue
            print('%-14s %-4d %-6.1f %10.0f %10s %10s  %s'
                  % (map_name, n, st.mean(tasks), st.mean(walls),
                     '%.0f' % st.mean(builds) if builds else '-',
                     '%.0f' % st.mean(searches) if searches else '-',
                     '/'.join(sorted(set(sts)))))
            sys.stdout.flush()

    if args.menu:
        print()
        print('== 指示画面を出すのに要る計算 ==')
        for map_name in args.maps.split(','):
            for n in counts:
                rec = rng.sample(RECIPES, n)
                cands, times = loss_cost(map_name, rec)
                if not times:
                    continue
                print('%-14s 注文%d  候補%d件  1件 %.0fms (最大 %.0f)  全部で %.1f秒'
                      % (map_name, n, len(cands), st.mean(times), max(times),
                         sum(times) / 1000))
                sys.stdout.flush()


if __name__ == '__main__':
    main()
