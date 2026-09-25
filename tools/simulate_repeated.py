# -*- coding: utf-8 -*-
"""AIが作業を2つ終えるたびに指示を出し直したら、差はどこまで積めるか。

指示を1回だけにすると、損失は「間違った一手ぶん」で打ち止まる。
出し直せるなら何度も積めるはず、という仮説を確かめる。

    python tools/simulate_repeated.py --cook 20 --chop 64 --every 2
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
from tools.design_probe import set_cook_time  # noqa: E402
from tools.bench_orders import build  # noqa: E402
from tools.selfplay_probe import state_for, make_agent, menu_at  # noqa: E402


def set_chop(n):
    import gym_cooking.utils.config as cfg
    cfg.CHOPPING_NUM_STEPS = n
    cfg.BLENDING_NUM_STEPS = n
    for mod in ('gym_cooking.utils.core', 'gym_cooking.envs.overcooked_environment',
                'gym_cooking.misc.game.game', 'agent.myagent.CSPAgent',
                'agent.myagent.TaskAgent', 'gym_cooking.utils.interact'):
        try:
            m = __import__(mod, fromlist=['_'])
        except Exception:
            continue
        for k in ('CHOPPING_NUM_STEPS', 'BLENDING_NUM_STEPS'):
            if hasattr(m, k):
                setattr(m, k, n)


def run(map_name, recipes, cook, chop, budget, every, mode, limit=0.5, cap=240.0,
        every_seconds=None):
    """mode: 'none' / 'worst' / 'best'"""
    set_cook_time(cook)
    set_chop(chop)
    env = build(map_name, recipes)
    driver = make_agent(0, False, None, limit)
    names = [a.name for a in env.sim_agents]
    done_at_last = 0
    last_t = [0.0]
    log = []
    # 「作業をいくつ終えたか」は completed_task_ids では数えられない。
    # chop と serve は「置くと決めた瞬間」であって実行確認ではないため
    # 記録されず、cook だけが入る(実測: 1ゲームで2件しか増えない)。
    # 代わりに、AI の担当作業が切り替わった回数を数える。
    seen = {'tid': None, 'n': 0}

    def switches():
        sc = (getattr(driver, 'schedule_per_agent', None) or {}).get(0) or []
        idx = (getattr(driver, 'current_task_idx', None) or {}).get(0, 0)
        cur = sc[idx]['id'] if idx < len(sc) else None
        if cur is not None and cur != seen['tid']:
            seen['tid'] = cur
            seen['n'] += 1
        return seen['n']

    def instruct():
        rows = [r for r in menu_at(env, budget, None)
                if r['bound'] and r['loss'] is not None]
        if not rows:
            return None
        pick = (max if mode == 'worst' else min)(rows, key=lambda r: r['loss'])
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
        log.append((round(env.current_time, 1), pick['task'], pick['loss']))
        return pick

    if mode != 'none':
        instruct()
    while env.current_time < cap:
        with contextlib.redirect_stdout(io.StringIO()):
            move, _ = driver(state_for(env, 0))
        acts = {}
        for n, key in ((names[0], 'ai_0'), (names[1], 'ai_1')):
            agent = next(a for a in env.sim_agents if a.name == n)
            acts[n] = tuple(resolve_action(agent, env.world,
                                           tuple(move.get(key) or (0, 0))))
        env.step(acts, passed_time=0.2)
        if env.order_scheduler.successful_orders >= len(recipes):
            break
        if mode != 'none':
            if every_seconds:
                if env.current_time - last_t[0] >= every_seconds:
                    last_t[0] = env.current_time
                    instruct()
            else:
                ndone = switches()
                if ndone - done_at_last >= every:
                    done_at_last = ndone
                    instruct()
    return {'time': env.current_time,
            'served': env.order_scheduler.successful_orders,
            'log': log}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', default='exp_partition')
    ap.add_argument('--orders',
                    default='OnionTomatoSoup,TomatoLettuceSoup,AppleOrangeJuice')
    ap.add_argument('--cook', type=int, default=20)
    ap.add_argument('--chop', type=int, default=64)
    ap.add_argument('--budget', type=int, default=0)
    ap.add_argument('--every', type=int, default=2)
    ap.add_argument('--every-seconds', type=float, default=None)
    args = ap.parse_args()
    recipes = args.orders.split(',')
    trig = ('%.0f秒ごとに指示' % args.every_seconds) if args.every_seconds else ('%d作業ごとに指示' % args.every)
    print('地図=%s / 煮込み%d秒 / 切る%d / d=%d / %s'
          % (args.map, args.cook, args.chop, args.budget, trig))
    out = {}
    for mode in ('none', 'best', 'worst'):
        r = run(args.map, recipes, args.cook, args.chop, args.budget,
                args.every, mode, every_seconds=args.every_seconds)
        out[mode] = r
        label = {'none': '指示なし', 'best': '毎回いちばん良い指示',
                 'worst': '毎回いちばん悪い指示'}[mode]
        print('  %-22s %6.1f秒 提供%d件  指示%d回  L合計%.1f'
              % (label, r['time'], r['served'], len(r['log']),
                 sum(x[2] for x in r['log'])))
        for t, task, L in r['log']:
            print('        %5.1f秒 %-28s L=%.1f' % (t, task, L))
        sys.stdout.flush()
    if 'best' in out and 'worst' in out:
        print('  => 良い指示と悪い指示の差 %.1f秒'
              % (out['worst']['time'] - out['best']['time']))


if __name__ == '__main__':
    main()
