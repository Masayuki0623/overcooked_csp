"""材料の足りない皿を提供口へ出さないことの検証。

想定シナリオ(バグ報告 20260924_180040 / 20260924_180746 の場面):
    注文は「玉ねぎ・トマトのサラダ」。AI はスープ用に空の皿を持っていて、
    そのまま置き場の刻んだ玉ねぎに触れると、玉ねぎが皿に乗って手元に残る。

    そこで AI は「皿に刻んだ材料が乗っている = サラダの完成品」と見なし、
    玉ねぎ1つだけの皿を提供口へ運んでしまった。提供口は注文に合わない物
    でも受け取るので、皿ごと材料が消えて作り直しになる。

実行方法:
    python tests/repro_partial_plate_not_served.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from gym_cooking.utils.core import Onion, Tomato, Plate, Object, FoodState
from gym_cooking.utils.interact import resolve_action
from gym_cooking.utils.replay import Replay
from agent.agent.executor.low import EnvState
from agent.agent.myagent.CSPAgent import CSPAgent

ORDERS = ('OnionTomatoSalad', 'OnionTomatoSalad', 'TomatoLettuceSoup')
results = []


def check(label, ok, detail=""):
    results.append(ok)
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}{(' -> ' + detail) if detail else ''}")


def chopped(cls):
    f = cls()
    f.set_state(FoodState.CHOPPED)
    return f


def state_of(env):
    i = env.get_ai_info()
    return EnvState(world=i['world'], agents=i['sim_agents'], agent_idx=0,
                    order=i['order_scheduler'], event_history=i['event_history'],
                    time=i['current_time'], chg_grid=i['chg_grid'])


def main():
    env = OvercookedEnvironment(MapSetting(level='exp_ring_veg',
                                           order_recipes=ORDERS))
    env.reset()

    me = env.sim_agents[0]
    held = Object(location=tuple(me.location), contents=[chopped(Onion), Plate()])
    env.world.insert(held)
    me.acquire(held)
    print(f'[SETUP] 注文={ORDERS}')
    print(f'[SETUP] AI は {me.get_holding()} を持っている(材料が1つ足りない皿)')

    # 足りないトマトは置き場にある
    st = state_of(env)
    counter = next(c for c in st.get_pos_by_obj_gs(gs='Counter')
                   if st.pos_obj.get(tuple(c)) is None)
    counter = tuple(counter)
    obj = Object(location=counter, contents=[chopped(Tomato)])
    env.world.insert(obj)
    env.world.get_gridsquare_at(counter).acquire(obj)
    print(f'[SETUP] 置き場 {counter} に 刻んだトマト')

    ai = CSPAgent(10, Replay(), sc_2agent=True, skip_budget=2)
    ai.human_counterpart_mode = True
    ai.own_agent_idx = 0
    ai.priority_weights = {}
    ai.gui_text_input = ''
    ai.gui_constraint_input = ''
    ai.active_constraints = []

    served_at = None
    # 60秒ぶん。見たいのは「止まらずに足りない材料を足して出せるか」で、
    # 何秒で出せるかではない。人間の実座標を距離に使うようにしてから、
    # この場面では切る順番が変わって 30.2秒 -> 38.2秒 になった。
    for _step in range(1, 301):
        move, _reason = ai(state_of(env))
        own = move.get('ai_0') if isinstance(move, dict) else move
        acts = {a.name: (0, 0) for a in env.sim_agents}
        if own:
            acts[env.sim_agents[0].name] = own
        for a in env.sim_agents:
            acts[a.name] = resolve_action(a, env.world, acts[a.name])
        env.step(acts, passed_time=0.2)
        if env.order_scheduler.successful_orders and served_at is None:
            served_at = env.current_time
            break

    log = list(getattr(env, 'delivery_log', None) or [])
    bad = [d for d in log if not d.get('ok')]
    for d in log:
        print('  %5.1f秒 %-38s %s' % (d['time'], d['dish'],
                                      '注文どおり' if d.get('ok') else '注文に無い'))
    check('注文に無い皿を出さない', not bad,
          '出してしまった: ' + ', '.join(d['dish'] for d in bad) if bad else '')
    check('足りない材料を足してサラダを出せる', served_at is not None,
          f'{served_at:.1f}秒' if served_at else '60秒回しても提供できず')

    print()
    print(f"[{'SUCCESS' if all(results) else 'FAIL'}] "
          f"{results.count(True)}/{len(results)} 件が期待どおり")
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
