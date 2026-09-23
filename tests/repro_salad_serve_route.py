"""サラダの提供を、材料先取り/皿先取りの早いほうで回ることの検証。

サラダだけは、提供までの回り方が2通りある。

    ingredient : 材料を取る → 皿タイルで皿に乗せる → 提供口
    plate      : 皿を取る   → 材料のある台で皿に乗せる → 提供口

鍋やミキサーの中身は持ち上げられず、必ず皿(コップ)を持って取りに行く
しかないので、この選択があるのはサラダだけ。どちらが早いかは立ち位置と
台の場所で入れ替わるため、CSP に makespan で決めさせている。

実行方法:
    python tests/repro_salad_serve_route.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from gym_cooking.utils.core import Onion, Tomato, Object, FoodState
from gym_cooking.utils.interact import resolve_action
from gym_cooking.utils.replay import Replay
from agent.agent.executor.low import EnvState
from agent.agent.myagent.CSPAgent import CSPAgent

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


def make_ai():
    ai = CSPAgent(10, Replay(), sc_2agent=True, skip_budget=2)
    ai.human_counterpart_mode = True
    ai.own_agent_idx = 0
    ai.priority_weights = {}
    ai.gui_text_input = ''
    ai.gui_constraint_input = ''
    ai.active_constraints = []
    return ai


def build(pile):
    env = OvercookedEnvironment(MapSetting(level='exp_partition',
                                           order_recipes=('OnionTomatoSalad',)))
    env.reset()
    obj = Object(location=pile, contents=[chopped(Onion), chopped(Tomato)])
    env.world.insert(obj)
    env.world.get_gridsquare_at(pile).acquire(obj)
    return env


def faster_route(ai, st, pile, ai_pos):
    """移動込みで、どちらが早いかを素直に計算する(CSP の答え合わせ用)。"""
    res = ai._get_resources(st)
    plate = ai._pick_plate(st, res, res['delivery'])
    choices = ai._salad_route_choices(st, pile, plate, res['delivery'])
    totals = {}
    for name, c in choices.items():
        approach = ai.astar_distance(st, ai_pos, c['start_pos'])
        totals[name] = approach + c['dur']
    return min(totals, key=lambda k: totals[k]), totals, plate


def run_case(label, pile):
    print(f'=== {label}: 材料の山 {pile} ===')
    env = build(pile)
    ai_pos = tuple(env.sim_agents[0].location)
    ai = make_ai()
    st = state_of(env)
    want, totals, plate = faster_route(ai, st, pile, ai_pos)
    print('  AI %s / 皿タイル %s / 移動込みの所要: %s'
          % (ai_pos, plate, {k: int(v) for k, v in totals.items()}))

    ai(state_of(env))
    serve = next((t for t in ((ai.schedule_per_agent or {}).get(0) or [])
                  if (t.get('id') or (None,))[0] == 'serve_salad'), None)
    if serve is None:
        check(f'{label}: 提供工程が AI の担当になる', False,
              f"AI の担当={[t.get('id') for t in (ai.schedule_per_agent or {}).get(0) or []]}")
        return None
    got = serve.get('serve_route')
    # 同着のときはどちらを選んでも最短。所要で見比べる。
    check(f'{label}: 早いほうの回り方を選ぶ',
          got in totals and totals[got] == min(totals.values()),
          f'選んだ={got}({totals.get(got)}) / 最短={want}({min(totals.values())})')

    # 実際に提供まで行けるか。皿先取りなら、山に触れる前に皿を持っている。
    took_plate_first = None
    served_at = None
    for _step in range(1, 201):
        move, _reason = ai(state_of(env))
        own = move.get('ai_0') if isinstance(move, dict) else move
        acts = {a.name: (0, 0) for a in env.sim_agents}
        if own:
            acts[env.sim_agents[0].name] = own
        for a in env.sim_agents:
            acts[a.name] = resolve_action(a, env.world, acts[a.name])
        env.step(acts, passed_time=0.2)
        held = getattr(env.sim_agents[0].holding, 'full_name', '') or ''
        if took_plate_first is None and held:
            took_plate_first = (held == 'Plate')
        if env.order_scheduler.successful_orders:
            served_at = env.current_time
            break
    check(f'{label}: サラダを提供できる', served_at is not None,
          f'{served_at:.1f}秒' if served_at else '40秒回しても提供できず')
    if got == 'plate':
        check(f'{label}: 先に皿を持つ', took_plate_first is True,
              f'最初に持った物={"皿" if took_plate_first else "材料"}')
    else:
        check(f'{label}: 先に材料を持つ', took_plate_first is False,
              f'最初に持った物={"皿" if took_plate_first else "材料"}')
    print()
    return got


def find_ingredient_first_counter():
    """材料を先に取るほうが確実に早くなる置き場を1つ探す。"""
    env = build((6, 5))
    ai = make_ai()
    st = state_of(env)
    ai_pos = tuple(env.sim_agents[0].location)
    for pos in st.get_pos_by_obj_gs(gs='Counter'):
        pos = tuple(pos)
        if st.pos_obj.get(pos) is not None:
            continue
        try:
            want, totals, _plate = faster_route(ai, st, pos, ai_pos)
        except Exception:
            continue
        if want == 'ingredient' and totals['ingredient'] < totals['plate']:
            return pos
    return None


def main():
    # 共有台(提供口から遠く、皿タイルは AI の目の前)。皿を先に取るほうが早い。
    a = run_case('共有台に置いてある', (6, 5))
    # 材料を先に取るほうが早くなる置き場。選ぶ答えが入れ替わることを見る。
    pos = find_ingredient_first_counter()
    if pos is None:
        check('材料先取りが早くなる置き場がある', False, 'この地図では見つからず')
        b = None
    else:
        b = run_case('材料のほうが近い台', pos)
    check('置き場によって選び方が変わる', a != b,
          f'共有台={a} / 材料のほうが近い台={b}')

    print(f"[{'SUCCESS' if all(results) else 'FAIL'}] "
          f"{results.count(True)}/{len(results)} 件が期待どおり")
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
