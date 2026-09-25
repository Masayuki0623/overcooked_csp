# -*- coding: utf-8 -*-
"""実際の回を再生しながら「その瞬間に指示できる作業と、その L」を並べる。

指示メニューは前提の済んだ作業しか出さないので、いつ指示を出すかで
候補も L もまるごと変わる。どの瞬間なら狙った L の段が作れるかを探す道具。

    python tools/instruction_menu_probe.py <リプレイ> --cook 25 --at 15,18,20,22
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

from tools.design_probe import set_cook_time, probe_agent, state_of  # noqa: E402
from gym_cooking.utils.replay import Replay  # noqa: E402


def rebuild(sel):
    from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
    from gym_cooking.play_test import MAP_SETTINGS
    name = sel['map']
    rec = sel.get('recipes') or []
    if not any(f in r for r in rec for f in ('Apple', 'Orange', 'Banana', 'Juice')):
        name = f'{name}_veg'
    kw = dict(MAP_SETTINGS[name])
    kw['order_recipes'] = tuple(rec)
    env = OvercookedEnvironment(MapSetting(**kw))
    env.reset()
    return env


def menu_at(env, budget):
    """いまメニューに出る作業と、その L(d)。縛りが成立したかも返す。"""
    st = state_of(env)
    ai = probe_agent(budget)
    with contextlib.redirect_stdout(io.StringIO()):
        ai(st)
    out = []
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
        out.append({'task': f'{verb} {obj}', 'loss': r.get('loss_seconds'),
                    'bound': '縛りが効いていません' not in buf.getvalue()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('replay')
    ap.add_argument('--cook', type=int, default=15)
    ap.add_argument('--budget', type=int, default=0)
    ap.add_argument('--at', default='0,5,10,15,20,25')
    args = ap.parse_args()

    set_cook_time(args.cook)
    rep = Replay.from_file(args.replay)
    sel = rep['web_selection']
    steps = [h for h in list(rep) if h['name'] == 'env.step']
    env = rebuild(sel)
    marks = [float(x) for x in args.at.split(',')]
    print('地図=%s / 注文=%s / 煮込み=%d秒 / d=%d'
          % (sel['map'], ','.join(sel['recipes']), args.cook, args.budget))

    def report(t):
        rows = menu_at(env, args.budget)
        good = [r['loss'] for r in rows if r['bound'] and r['loss'] is not None]
        print('--- %.0f秒 (候補%d件) 信用できる最大L=%.1f秒'
              % (t, len(rows), max(good) if good else 0.0))
        for r in sorted(rows, key=lambda x: -(x['loss'] if x['loss'] is not None else -1)):
            print('    %-32s L=%5s  %s'
                  % (r['task'], '%.1f' % r['loss'] if r['loss'] is not None else '-',
                     'OK' if r['bound'] else '縛れず'))
        sys.stdout.flush()

    mi = 0
    if marks and marks[0] <= 0:
        report(0.0)
        mi = 1
    for h in steps:
        acts = {a.name: tuple(h['args']['action_dict'].get(a.name) or (0, 0))
                for a in env.sim_agents}
        env.step(acts, passed_time=h['args'].get('passed_time', 0.2))
        if mi < len(marks) and env.current_time >= marks[mi]:
            report(env.current_time)
            mi += 1
        if mi >= len(marks):
            break


if __name__ == '__main__':
    main()
