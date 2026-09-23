"""間違った材料が鍋に入ったとき、鍋を空けて作り直せることの検証。

想定シナリオ:
    注文は「レタス・玉ねぎのスープ」なのに、鍋には玉ねぎ・トマトが入って
    しまった。鍋の中身は足すことも入れ替えることもできないので、そのままだと
    スープは永久に作れない。

    空の皿で取り出せるのは煮上がってからなので、煮上がるのを待って皿に取り、
    近くの空いた台へ置く。それだけでよい(提供口へは持って行かない)。

実行方法:
    python tests/repro_clear_wrong_pot.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from gym_cooking.utils import config as game_config
from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from gym_cooking.utils.core import Onion, Tomato, Object, FoodState
from gym_cooking.utils.interact import resolve_action
from gym_cooking.utils.replay import Replay
from agent.agent.executor.low import EnvState
from agent.agent.myagent.CSPAgent import CSPAgent

ORDERS = ('OnionLettuceSoup', 'OnionTomatoSalad')
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

    # 注文と違う中身を鍋に入れてしまった状態を作る(AI に見せる前に)
    pot = state_of(env).get_pos_by_obj_gs(gs='Pot')[0]
    wrong = Object(location=pot, contents=[chopped(Onion), chopped(Tomato)])
    env.world.insert(wrong)
    env.world.get_gridsquare_at(pot).acquire(wrong)
    wrong.cook(0.1)
    print(f'[SETUP] 注文={ORDERS}')
    print(f'[SETUP] 鍋 {pot} に 玉ねぎ+トマト(注文にない組み合わせ)が入っている')

    ai = CSPAgent(10, Replay(), sc_2agent=True, skip_budget=2)
    ai.human_counterpart_mode = True
    ai.own_agent_idx = 0
    ai.priority_weights = {}
    ai.gui_text_input = ''
    ai.gui_constraint_input = ''
    ai.active_constraints = []
    ai(state_of(env))

    sched = (ai.schedule_per_agent or {}).get(0) or []
    verbs = [t['id'][0] for t in sched if t.get('id')]
    print('  AI の計画:', [t.get('id') for t in sched])
    check('鍋を空ける工程が計画に入る', 'clear_pot' in verbs, f'{verbs}')
    if 'clear_pot' in verbs and 'cook' in verbs:
        check('鍋を空けてから入れる順番になっている',
              verbs.index('clear_pot') < verbs.index('cook'),
              f"clear_pot={verbs.index('clear_pot')} cook={verbs.index('cook')}")

    # 煮上がるまで進めてから、実際に空けられるかを見る
    cook_s = game_config.COOKING_TIME_SECONDS
    idle = {a.name: (0, 0) for a in env.sim_agents}
    for _ in range(int((cook_s + 1) / 0.2)):
        env.step(dict(idle), passed_time=0.2)
    obj = state_of(env).pos_obj.get(pot)
    print(f'  {env.current_time:.1f}秒: 鍋の中身={getattr(obj, "full_name", None)}'
          f' (煮上がり={getattr(obj, "is_cooked", lambda: None)()})')

    cleared_at = None
    for step in range(1, 121):
        move, reason = ai(state_of(env))
        own = move.get('ai_0') if isinstance(move, dict) else move
        acts = {a.name: (0, 0) for a in env.sim_agents}
        if own:
            acts[env.sim_agents[0].name] = own
        for a in env.sim_agents:
            acts[a.name] = resolve_action(a, env.world, acts[a.name])
        env.step(acts, passed_time=0.2)
        if state_of(env).pos_obj.get(pot) is None:
            cleared_at = env.current_time
            break
    check('鍋を空けられる', cleared_at is not None,
          f'{cleared_at:.1f}秒に空になった' if cleared_at
          else '120フレーム回しても空にならず')

    if cleared_at is not None:
        held = getattr(env.sim_agents[0].holding, 'full_name', None)
        print(f'  空にした直後の持ち物: {held}')

    print()
    print(f"[{'SUCCESS' if all(results) else 'FAIL'}] "
          f"{results.count(True)}/{len(results)} 件が期待どおり")
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
