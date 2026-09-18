"""人が間違えて材料をくっつけたときに、AI が立て直せるかを見る。

ゲームの途中、共有台に切った材料が1つだけ乗った瞬間に、どの注文にも
使わない組み合わせになる材料を重ねる(例: 玉ねぎトマトの料理が無いのに
トマトに玉ねぎを重ねる)。環境はレシピとして存在する組み合わせなら
重ねるのを許すので、人が普通にやってしまう間違い。重ねた山は分けられず、
この地図にはゴミ箱も無いので、その山は放っておき、材料を用意し直すしかない。

    python tools/mistake_check.py greedy          # 12構成 x 3つの時点
    python tools/mistake_check.py messy 0/8       # 並列に分けて回す
    python tools/mistake_check.py greedy 6:3      # 1試行だけ(case:秒)
"""
import os
import sys
from copy import deepcopy as dcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in ('.', 'agent', 'testbed-cooking', 'tools'):
    sys.path.insert(0, str(ROOT / _p))

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

from gym_cooking.utils.core import (  # noqa: E402
    Apple, Banana, Lettuce, Object, Onion, Orange, Tomato, mergeable)
from gym_cooking.utils.order_preset import (  # noqa: E402
    enumerate_order_recipes, experiment_case_indices)
from gym_cooking.utils.replay import Replay  # noqa: E402
import run_human_model_experiment as H  # noqa: E402
from human_models import HumanModel  # noqa: E402
from detect_freeze import snapshot, pot_is_cooking  # noqa: E402

FOODS = {'Lettuce': Lettuce, 'Onion': Onion, 'Tomato': Tomato,
         'Apple': Apple, 'Orange': Orange, 'Banana': Banana}
DELAYS = (3.0, 10.0, 20.0)   # 間違いを起こす時点(ゲーム内の秒)
STEP = 0.1


def needed_combos(env):
    out = []
    for o in env.order_scheduler.current_orders:
        n = o[0].full_name
        out.append({f for f in FOODS if f in n})
    return out


def make_mistake(env):
    """1つだけ乗った切った材料に、どの注文も使わない組み合わせを作る。"""
    st = H.state_for(env, 0)
    combos = needed_combos(env)
    for pos in sorted(st.get_pos_by_obj_gs(gs='Counter')):
        obj = st.pos_obj.get(pos)
        if obj is None or getattr(obj, 'is_held', False) or len(obj.contents) != 1:
            continue
        name = obj.full_name
        if not name.startswith('Chopped'):
            continue
        base = name[len('Chopped'):]
        for other in FOODS:
            if other == base:
                continue
            if any({base, other} <= c for c in combos):
                continue      # 何かの注文に使える組み合わせは間違いではない
            extra = Object(location=pos, contents=[FOODS[other](state_index=2)])
            if not mergeable(obj, extra):
                continue
            world_obj = env.world.get_object_at(pos, None, find_held_objects=False)
            env.world.remove(world_obj)
            world_obj.merge(extra)
            env.world.insert(world_obj)
            return '%s に %s を重ねた @%s' % (name, other, pos)
    return None


def run_trial(case, recipes, model, delay, min_freeze_s=3.0):
    env = H.make_env('experiment', case, 'experiment2', recipes)
    ai = H.make_ai(0, partner_is_external=(model != 'follow_plan'))
    human = HumanModel(model, ai, 1, Replay(), seed=case * 31 + 7)
    pot_locs = H.state_for(env, 0).get_pos_by_obj_gs(gs='Pot')
    prev = snapshot(env)
    run = 0
    worst = 0.0
    mistake = None
    for _step in range(int(H.MAX_SECONDS_OVERRIDE / STEP)):
        if mistake is None and env.current_time >= delay:
            m = make_mistake(env)
            if m:
                mistake = 't=%.1f %s' % (env.current_time, m)
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
        'mistake': mistake,
    }


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else 'greedy'
    sel = sys.argv[2] if len(sys.argv) > 2 else None
    H.MAX_SECONDS_OVERRIDE = 100.0
    sets = enumerate_order_recipes('experiment2')
    combos = [(c, d) for c in experiment_case_indices('experiment2') for d in DELAYS]
    if sel and ':' in sel:
        c, d = sel.split(':')
        combos = [(int(c), float(d))]
    elif sel:
        i, n = (int(x) for x in sel.split('/'))
        combos = [x for k, x in enumerate(combos) if k % n == i]

    ok = injected = 0
    for case, delay in combos:
        r = run_trial(case, sets[case], model, delay)
        ok += int(r['completed'])
        injected += int(bool(r['mistake']))
        bad = (not r['completed']) or r['freeze']
        print('case%2d 時点%4.1f 提供%d %s %.1f秒%s | %s%s' % (
            case, delay, r['served'], 'OK' if r['completed'] else '未完', r['makespan'],
            '  停止%.1f秒' % r['freeze'] if r['freeze'] else '',
            r['mistake'] or '(間違いを作れず)', '   <<<' if bad else ''), flush=True)
    print('完走 %d/%d (間違いを入れた %d)' % (ok, len(combos), injected))


if __name__ == '__main__':
    main()
