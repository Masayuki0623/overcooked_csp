"""空の皿を持ったまま、取っては戻すを繰り返さないことの検証。

想定シナリオ(バグ報告 20260925_004543「AIが皿を持ったままうろちょろして
いた」の場面):
    サラダの材料が共有台にそろっている。皿先取りで回ると決まっているので、
    AI はまず皿を取り、その山へ触れに行く。

    ところが「指定テーブルに材料が全部そろっているなら、いま持っている
    ものは余り」という決まりが、空の皿にも当てはまってしまっていた。
    皿を取る → 余りとみなして置きに行く → また取る、を繰り返す。

実行方法:
    python tests/repro_plate_loop_on_salad.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from gym_cooking.utils.core import (Banana, Lettuce, Object, Onion, Orange,
                                    Plate, Tomato, FoodState)
from gym_cooking.utils.interact import resolve_action
from gym_cooking.utils.replay import Replay
from agent.agent.executor.low import EnvState
from agent.agent.myagent.CSPAgent import CSPAgent

ORDERS = ('TomatoLettuceSalad', 'OnionTomatoSoup', 'BananaOrangeJuice')
PILE = (6, 5)
JUICE_PILE = (6, 4)
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
    env = OvercookedEnvironment(MapSetting(level='exp_partition',
                                           order_recipes=ORDERS))
    env.reset()

    def place(pos, contents):
        o = Object(location=pos, contents=contents)
        env.world.insert(o)
        env.world.get_gridsquare_at(pos).acquire(o)
        return o

    # 報告のあった場面をそのまま作る。
    #   スープは鍋で煮ている最中 / サラダとジュースの材料は共有台にそろい、
    #   AI はスープ用に取った空の皿を持っている。
    st = state_of(env)
    pot = st.get_pos_by_obj_gs(gs='Pot')[0]
    soup = place(tuple(pot), [chopped(Onion), chopped(Tomato)])
    soup.cook(0.1)
    place(PILE, [chopped(Lettuce), chopped(Tomato)])
    place(JUICE_PILE, [chopped(Banana), chopped(Orange)])

    me = env.sim_agents[0]
    plate = Object(location=tuple(me.location), contents=[Plate()])
    env.world.insert(plate)
    me.acquire(plate)

    print(f'[SETUP] 注文={ORDERS}')
    print(f'[SETUP] 鍋 {tuple(pot)} でスープを煮ている')
    print(f'[SETUP] 共有台 {PILE} に 刻んだレタス+トマト (サラダはすぐ出せる)')
    print(f'[SETUP] 共有台 {JUICE_PILE} に 刻んだバナナ+オレンジ')
    print(f'[SETUP] AI は空の皿を持っている')

    ai = CSPAgent(10, Replay(), sc_2agent=True, skip_budget=1)
    ai.human_counterpart_mode = True
    ai.own_agent_idx = 0
    ai.priority_weights = {}
    ai.gui_text_input = ''
    ai.gui_constraint_input = ''
    ai.active_constraints = []

    # 相手が動き回ると、そのたびに計画を立て直す。2通りの回り方の差は
    # 1手ぶんしかないので、覚えておかないとそこで入れ替わる。
    partner = env.sim_agents[1]
    walk = [(0, 1), (0, -1)]

    reasons = []
    routes = []
    served_at = None
    for step in range(1, 151):
        move, reason = ai(state_of(env))
        reasons.append(str(reason))
        sched = (ai.schedule_per_agent or {}).get(0) or []
        salad = next((t for t in sched
                      if (t.get('id') or (None,))[0] == 'serve_salad'), None)
        if salad and salad.get('serve_route'):
            routes.append(salad['serve_route'])
        own = move.get('ai_0') if isinstance(move, dict) else move
        acts = {a.name: (0, 0) for a in env.sim_agents}
        if own:
            acts[me.name] = own
        acts[partner.name] = walk[(step // 2) % len(walk)]
        for a in env.sim_agents:
            acts[a.name] = resolve_action(a, env.world, acts[a.name])
        env.step(acts, passed_time=0.2)
        if env.order_scheduler.successful_orders:
            served_at = env.current_time
            break

    drops = sum(1 for r in reasons if 'サラダの材料が揃っている' in r)
    for r in reasons[:4]:
        print(f'  reason={r}')
    check('空の皿を「余り」として戻しに行かない', drops == 0,
          f'{drops}/{len(reasons)} フレームで戻そうとした')
    flips = sum(1 for a, b in zip(routes, routes[1:]) if a != b)
    check('回り方が途中で入れ替わらない', flips == 0,
          f'{flips} 回入れ替わった ({sorted(set(routes))})')
    check('サラダを出せる', served_at is not None,
          f'{served_at:.1f}秒' if served_at else '30秒回しても出せず')

    print()
    print(f"[{'SUCCESS' if all(results) else 'FAIL'}] "
          f"{results.count(True)}/{len(results)} 件が期待どおり")
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
