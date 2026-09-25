# -*- coding: utf-8 -*-
"""AI同士に遊ばせて、その途中盤面で「指示メニューと L」を測る。

人を呼ばずに設計案を試すための道具。任意の地図・注文・煮込み時間で
途中盤面が作れるので、「いつ・どの指示なら何秒損させられるか」を
その場で確かめられる。

    python tools/selfplay_probe.py --map exp_ring --cook 25 \
        --orders OnionTomatoSoup,TomatoLettuceSoup,AppleOrangeJuice \
        --at 10,15,20,25,30,35
"""
import argparse
import contextlib
import io
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in ('.', 'agent', 'testbed-cooking'):
    sys.path.insert(0, str(ROOT / _p))
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

from gym_cooking.utils.interact import resolve_action  # noqa: E402
from gym_cooking.utils.replay import Replay  # noqa: E402
from agent.executor.low import EnvState  # noqa: E402
from agent.myagent.CSPAgent import CSPAgent  # noqa: E402
from tools.design_probe import set_cook_time  # noqa: E402
from tools.bench_orders import build  # noqa: E402


def state_for(env, idx=0):
    i = env.get_ai_info()
    return EnvState(world=i['world'], agents=i['sim_agents'], agent_idx=idx,
                    order=i['order_scheduler'], event_history=i['event_history'],
                    time=i['current_time'], chg_grid=i['chg_grid'])


def make_agent(idx, counterpart, budget, limit):
    a = CSPAgent(10, Replay(), sc_2agent=True, skip_budget=budget)
    a.human_counterpart_mode = counterpart
    a.own_agent_idx = idx
    a.priority_weights = {}
    a.gui_text_input = ''
    a.gui_constraint_input = ''
    a.active_constraints = []
    a.debug_counter_trace = False
    a.solve_deterministic_limit = limit
    return a


def menu_at(env, budget, limit):
    """いまメニューに出る作業と L(d)。縛りが成立したかも見る。"""
    st = state_for(env, 0)
    ai = make_agent(0, True, budget, limit)
    with contextlib.redirect_stdout(io.StringIO()):
        ai(st)
    rows = []
    for c in ai.get_instruction_candidates(st):
        p = c[1] if isinstance(c, (list, tuple)) and len(c) >= 2 else c
        if not isinstance(p, dict):
            continue
        verb, obj = str(p.get('verb')), str(p.get('obj'))
        pend = {'id': time.time(), 'task': {'verb': verb, 'obj': obj},
                'target_idx': 0, 'status': 'pending',
                'skip_budget': budget, 'remaining_skip_budget': budget}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = ai.estimate_instruction_time_loss(st, pend, skip_budget=budget)
        rows.append({'task': f'{verb} {obj}', 'loss': r.get('loss_seconds'),
                     'bound': '縛りが効いていません' not in buf.getvalue()})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', default='exp_ring')
    ap.add_argument('--orders', required=True)
    ap.add_argument('--cook', type=int, default=15)
    ap.add_argument('--budget', type=int, default=0)
    ap.add_argument('--at', default='10,15,20,25,30')
    ap.add_argument('--limit', type=float, default=0.5,
                    help='計画を打ち切る決定性時間(自動プレイ側)')
    ap.add_argument('--max-seconds', type=float, default=120.0)
    args = ap.parse_args()

    set_cook_time(args.cook)
    recipes = args.orders.split(',')
    env = build(args.map, recipes)
    driver = make_agent(0, False, None, args.limit)
    marks = [float(x) for x in args.at.split(',')]
    print('地図=%s / 注文=%s / 煮込み=%d秒 / d=%d'
          % (args.map, ','.join(recipes), args.cook, args.budget))

    names = [a.name for a in env.sim_agents]
    mi = 0
    while env.current_time < args.max_seconds:
        if mi < len(marks) and env.current_time >= marks[mi]:
            rows = menu_at(env, args.budget, None)   # L は最適まで解く
            good = [r['loss'] for r in rows if r['bound'] and r['loss'] is not None]
            good.sort(reverse=True)
            print('--- %.0f秒 (候補%d件) 使えるL: %s'
                  % (env.current_time, len(rows),
                     ' / '.join('%.1f' % v for v in good) or 'なし'))
            for r in sorted(rows, key=lambda x: -(x['loss'] if x['loss'] is not None else -1)):
                print('    %-32s L=%5s  %s'
                      % (r['task'],
                         '%.1f' % r['loss'] if r['loss'] is not None else '-',
                         'OK' if r['bound'] else '縛れず'))
            sys.stdout.flush()
            mi += 1
        with contextlib.redirect_stdout(io.StringIO()):
            move, _ = driver(state_for(env, 0))
        # AI は「その物の方向へ進む」で行動を出す。本番と同じく
        # 「向く / 手を出す」へ読み替えてから環境へ渡す。
        acts = {}
        for n, key in ((names[0], 'ai_0'), (names[1], 'ai_1')):
            agent = next(a for a in env.sim_agents if a.name == n)
            acts[n] = tuple(resolve_action(agent, env.world,
                                           tuple(move.get(key) or (0, 0))))
        env.step(acts, passed_time=0.2)
        sched = env.order_scheduler
        if sched.successful_orders >= len(recipes):
            break
    sched = env.order_scheduler
    print('=> %.1f秒で 提供%d件 / 残り%d件'
          % (env.current_time, sched.successful_orders, len(sched.current_orders)))


if __name__ == '__main__':
    main()
