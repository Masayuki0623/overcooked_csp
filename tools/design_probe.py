# -*- coding: utf-8 -*-
"""ある設定(地図・注文・煮込み時間)が「悪い指示」をどれだけ痛くできるかを測る。

実験を回す前に、CSP 自身に聞いて設計の良し悪しを見る道具。
盤面の開始時点で、指示できる作業をすべて試し、それぞれの
L(d) = f'(d) - f を計算して、その設定の「振れ幅」を出す。

    python tools/design_probe.py --map exp_ring --cook 15
    python tools/design_probe.py --map exp_ring --cook 25 --orders TomatoLettuceSalad,OnionTomatoSoup,AppleOrangeJuice
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in ('.', 'agent', 'testbed-cooking'):
    sys.path.insert(0, str(ROOT / _p))
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')


def set_cook_time(sec):
    """煮込み時間を差し替える。読み込み済みの名前も全部そろえる。"""
    import gym_cooking.utils.config as cfg
    cfg.COOKING_TIME_SECONDS = sec
    for mod in ('gym_cooking.recipe_planner.utils',
                'gym_cooking.envs.overcooked_environment',
                'gym_cooking.misc.game.game',
                'agent.myagent.CSPAgent'):
        try:
            m = __import__(mod, fromlist=['_'])
        except Exception:
            continue
        if hasattr(m, 'COOKING_TIME_SECONDS'):
            setattr(m, 'COOKING_TIME_SECONDS', sec)


def build(map_name, recipes, num_orders):
    from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
    from gym_cooking.play_test import MAP_SETTINGS
    kw = dict(MAP_SETTINGS[map_name])
    if recipes:
        kw['order_recipes'] = tuple(recipes)
        kw['max_num_orders'] = len(recipes)
    elif num_orders:
        kw['max_num_orders'] = num_orders
    env = OvercookedEnvironment(MapSetting(**kw))
    env.reset()
    return env


def probe_agent(skip_budget):
    from gym_cooking.utils.replay import Replay
    from agent.myagent.CSPAgent import CSPAgent
    ai = CSPAgent(10, Replay(), sc_2agent=True, skip_budget=skip_budget)
    ai.human_counterpart_mode = True
    ai.own_agent_idx = 0
    ai.priority_weights = {}
    ai.gui_text_input = ''
    ai.gui_constraint_input = ''
    ai.active_constraints = []
    ai.debug_counter_trace = False
    return ai


def state_of(env):
    from agent.executor.low import EnvState
    i = env.get_ai_info()
    return EnvState(world=i['world'], agents=i['sim_agents'], agent_idx=0,
                    order=i['order_scheduler'], event_history=i['event_history'],
                    time=i['current_time'], chg_grid=i['chg_grid'])


def run(map_name, recipes, num_orders, cook, budgets):
    set_cook_time(cook)
    env = build(map_name, recipes, num_orders)
    st = state_of(env)
    ai = probe_agent(budgets[0])
    ai(st)  # 一度動かして内部を立ち上げる
    orders = ai._build_order_tasks(st)
    tasks = [t for o in orders for t in (o.get('tasks') or [])]
    cands, seen = [], set()
    for t in tasks:
        key = (str(t.get('verb')), str(t.get('obj')))
        if key[0] in ('None', '') or key in seen:
            continue
        seen.add(key)
        cands.append(key)

    print('地図=%s / 注文=%s / 煮込み=%d秒 / 作業数=%d'
          % (map_name, ','.join(recipes or ['(既定)']), cook, len(tasks)))
    print('指示できる作業 %d 種類: %s' % (len(cands), ', '.join('%s %s' % c for c in cands)))
    print()
    print(' %-22s %s' % ('指示する作業', '  '.join('d=%d' % d for d in budgets)))
    best = (None, -1, None)
    rows = []
    for verb, obj in cands:
        cells, f0 = [], None
        for d in budgets:
            # skip_budget は pending 側にも要る。ここが抜けていると
            # 制約そのものが積まれず、L がいつでも 0 になる。
            pending = {'id': time.time(), 'task': {'verb': verb, 'obj': obj},
                       'target_idx': 0, 'status': 'pending',
                       'skip_budget': d, 'remaining_skip_budget': d}
            r = ai.estimate_instruction_time_loss(st, pending, skip_budget=d)
            L = r.get('loss_seconds')
            f0 = f0 if f0 is not None else r.get('baseline_seconds')
            cells.append('%5.1f' % L if L is not None else '  -  ')
            if L is not None and L > best[1]:
                best = ((verb, obj), L, d)
        rows.append(('%s %s' % (verb, obj), cells, f0))
        print(' %-22s %s' % ('%s %s' % (verb, obj), '  '.join(cells)))
    f = rows[0][2] if rows else None
    print()
    print('指示なしの最適 f = %s秒' % ('%.1f' % f if f is not None else '-'))
    if best[0]:
        print('いちばん痛い指示: %s %s (d=%d) で L=%.1f秒  -> f\'=%.1f秒'
              % (best[0][0], best[0][1], best[2], best[1], (f or 0) + best[1]))
    return rows, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', default='exp_ring')
    ap.add_argument('--orders', default=None, help='カンマ区切りのレシピ名')
    ap.add_argument('--num-orders', type=int, default=None)
    ap.add_argument('--cook', type=int, default=15)
    ap.add_argument('--budgets', default='0,1,2')
    args = ap.parse_args()
    recipes = args.orders.split(',') if args.orders else None
    budgets = [int(x) for x in args.budgets.split(',')]
    run(args.map, recipes, args.num_orders, args.cook, budgets)


if __name__ == '__main__':
    main()
