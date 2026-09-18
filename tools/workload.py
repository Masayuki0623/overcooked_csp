"""二人の仕事量を測る。

1試合を走らせて、それぞれが「仕事をしていた」時間を数える。
  働いた   : 位置か持ち物が変わった、または器具を使った(刻む・混ぜる等)
  手待ち   : 何もしていない(行動が (0,0))
同じ地図・同じ注文で、AI と人間の働いた時間の割合を比べる。

    python tools/workload.py                         # 実験の地図、12構成、貪欲な相方
    python tools/workload.py --level experiment --model follow_plan
"""
import argparse
import os
import sys
from copy import deepcopy as dcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in ('.', 'agent', 'testbed-cooking', 'tools'):
    sys.path.insert(0, str(ROOT / _p))

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting  # noqa: E402
from gym_cooking.utils.order_preset import (  # noqa: E402
    enumerate_order_recipes, experiment_case_indices)
from gym_cooking.utils.replay import Replay  # noqa: E402
import run_human_model_experiment as H  # noqa: E402
from human_models import HumanModel  # noqa: E402
from detect_freeze import snapshot, pot_is_cooking  # noqa: E402

STEP = 0.1


def make_env(level, recipes, max_seconds=100.0):
    env = OvercookedEnvironment(MapSetting(
        level=level, max_num_orders=3, max_num_timesteps=max_seconds,
        order_recipes=list(recipes)))
    env.reset()
    return env


def worked(before, after, action):
    """このフレームで仕事をしたか。

    動いた・持ち物が変わったに加えて、器具の前で操作している(刻む・
    混ぜる)フレームも数える。その間は位置も持ち物も変わらないが、
    仕事はしている。壁に向かって歩き続けているだけのフレームは、
    行動はあっても何も変わらないので、器具を使ったかで区別する。
    """
    if action == (0, 0):
        return False
    if before != after:
        return True
    return 'using'


def run_trial(level, recipes, model, seed=0, max_seconds=100.0):
    env = make_env(level, recipes, max_seconds)
    ai = H.make_ai(0, partner_is_external=(model != 'follow_plan'))
    human = HumanModel(model, ai, 1, Replay(), seed=seed)

    work = [0, 0]
    idle = [0, 0]
    frames = 0
    pot_locs = H.state_for(env, 0).get_pos_by_obj_gs(gs='Pot')
    prev_snap = snapshot(env)
    run = 0
    worst_freeze = 0.0
    for _step in range(int(max_seconds / STEP)):
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

        before = [(tuple(a.location), getattr(a.holding, 'full_name', None))
                  for a in env.sim_agents]
        progress_before = _appliance_progress(env)
        env.step(acts, passed_time=STEP)
        progress_after = _appliance_progress(env)
        frames += 1
        for i, a in enumerate(env.sim_agents):
            action = acts[a.name] or (0, 0)
            after = (tuple(a.location), getattr(a.holding, 'full_name', None))
            w = worked(before[i], after, action)
            if w == 'using':
                # 器具の前で操作した(刻む/混ぜる)なら、器具の進み具合が変わる
                w = progress_before != progress_after
            if w:
                work[i] += 1
            elif action == (0, 0):
                idle[i] += 1
        # 二人とも止まっている時間(調理中は除く)
        snap = snapshot(env)
        if snap == prev_snap and not pot_is_cooking(H.state_for(env, 0), pot_locs):
            run += 1
            worst_freeze = max(worst_freeze, run * STEP)
        else:
            run = 0
        prev_snap = snap
        if not env.order_scheduler.current_orders:
            break

    sched = env.order_scheduler
    return {
        'freeze': worst_freeze,
        'served': sched.successful_orders,
        'completed': not sched.current_orders,
        'makespan': env.current_time,
        'work_s': [w * STEP for w in work],
        'idle_s': [x * STEP for x in idle],
        'frames': frames,
    }


def _appliance_progress(env):
    """まな板・ミキサーの上の物の状態(刻む・混ぜるの進み具合を含む)。"""
    out = []
    for gs in ('Cutboard', 'Blender'):
        for obj in env.world.objects.get(gs, []):
            loc = obj.location
            for o in env.world.get_object_list():
                if getattr(o, 'location', None) == loc and hasattr(o, 'contents'):
                    for c in o.contents:
                        st = getattr(c, 'state', None)
                        out.append((loc, getattr(c, 'full_name', ''),
                                    getattr(st, '_rest_steps', None)))
    return tuple(sorted(out, key=str))


def summarize(rows):
    n = len(rows)
    ai = sum(r['work_s'][0] for r in rows)
    hu = sum(r['work_s'][1] for r in rows)
    return {
        'n': n,
        'completed': sum(1 for r in rows if r['completed']),
        'makespan': sum(r['makespan'] for r in rows) / n,
        'ai_work': ai / n,
        'human_work': hu / n,
        'ai_share': ai / max(ai + hu, 1e-9),
        'ai_idle': sum(r['idle_s'][0] for r in rows) / n,
        'human_idle': sum(r['idle_s'][1] for r in rows) / n,
        'frozen': sum(1 for r in rows if r['freeze'] >= 3.0),
        'worst_freeze': max(r['freeze'] for r in rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--level', default='experiment')
    ap.add_argument('--model', default='greedy')
    args = ap.parse_args()

    H.MAX_SECONDS_OVERRIDE = 100.0
    sets = enumerate_order_recipes('experiment2')
    rows = []
    for case in experiment_case_indices('experiment2'):
        r = run_trial(args.level, sets[case], args.model, seed=case * 31 + 7)
        rows.append(r)
        print('case%2d 提供%d %.1f秒  働いた AI %.1f秒 / 人 %.1f秒' % (
            case, r['served'], r['makespan'], r['work_s'][0], r['work_s'][1]), flush=True)
    s = summarize(rows)
    print('\n%s / %s: 完走 %d/%d  所要 平均%.1f秒' % (
        args.level, args.model, s['completed'], s['n'], s['makespan']))
    print('  働いた時間  AI %.1f秒  人 %.1f秒   → AI の割合 %.0f%%' % (
        s['ai_work'], s['human_work'], 100 * s['ai_share']))
    print('  手待ち      AI %.1f秒  人 %.1f秒' % (s['ai_idle'], s['human_idle']))
    print('  3秒以上の停止 %d件 (最長 %.1f秒)' % (s['frozen'], s['worst_freeze']))


if __name__ == '__main__':
    main()
