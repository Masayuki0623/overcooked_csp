"""提供できない側がサラダを受け渡すとき、皿を持ってきて山の場所で盛ることの検証。

想定シナリオ(報告: 4/6 の回):
    仕切りの地図。AI は右側にいて、提供口は左(人間の側)。刻んだ山が
    受け渡し台に乗っている。以前は「材料を先に取る」回り方だったので、
    AI が山を持ち上げて皿の場所へ行き、盛ってから受け渡し台へ戻していた。
    参加者から見て「持って行って戻す」不可解な動き。
    皿を先に取り、山のところで盛れば、そのまま受け渡し台に置ける。

実行方法:
    python tests/repro_salad_handover_plate_first.py
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


def check(label, ok, detail=''):
    results.append(bool(ok))
    print(f"  [{'OK  ' if ok else 'NG  '}] {label}" + (f" -> {detail}" if detail else ''))


def state_of(env):
    i = env.get_ai_info()
    return EnvState(world=i['world'], agents=i['sim_agents'], agent_idx=0,
                    order=i['order_scheduler'], event_history=i['event_history'],
                    time=i['current_time'], chg_grid=i['chg_grid'])


def chopped(cls):
    f = cls()
    f.set_state(FoodState.CHOPPED)
    return f


def main():
    env = OvercookedEnvironment(MapSetting(level='exp_partition',
                                           order_recipes=('OnionTomatoSalad',)))
    env.reset()
    pile_pos = (6, 5)     # 中央の受け渡し台
    obj = Object(location=pile_pos, contents=[chopped(Onion)])
    obj.contents.append(chopped(Tomato))
    env.world.insert(obj)
    env.world.get_gridsquare_at(pile_pos).acquire(obj)
    print(f'[SETUP] 受け渡し台 {pile_pos} に ChoppedOnion-ChoppedTomato / AI は右側、提供口は左側')

    ai = CSPAgent(10, Replay(), sc_2agent=True, skip_budget=0)
    ai.human_counterpart_mode = True
    ai.own_agent_idx = 0
    ai.priority_weights = {}
    ai.gui_text_input = ''
    ai.gui_constraint_input = ''
    ai.active_constraints = []

    first_hold = None
    pile_lifted_bare = False
    plated_on_counter_at = None
    for _ in range(200):
        st = state_of(env)
        move, _reason = ai(st)
        own = move.get('ai_0') if isinstance(move, dict) else move
        acts = {a.name: (0, 0) for a in env.sim_agents}
        if own:
            acts[env.sim_agents[0].name] = own
        for a in env.sim_agents:
            acts[a.name] = resolve_action(a, env.world, acts[a.name])
        env.step(acts, passed_time=0.2)
        held = getattr(env.sim_agents[0].holding, 'full_name', '') or ''
        if held and first_hold is None:
            first_hold = held
        if held and 'Chopped' in held and 'Plate' not in held:
            pile_lifted_bare = True
        for pos, o in state_of(env).pos_obj.items():
            name = getattr(o, 'full_name', '') or ''
            if (pos[0] == 6 and 'ChoppedOnion' in name and 'ChoppedTomato' in name
                    and 'Plate' in name):
                plated_on_counter_at = plated_on_counter_at or env.current_time
        if plated_on_counter_at and env.current_time - plated_on_counter_at > 2.0:
            break

    check('最初に手に取るのは皿', first_hold == 'Plate', f'最初の持ち物={first_hold}')
    check('山を皿無しで持ち上げない(持って行って戻さない)', not pile_lifted_bare)
    check('盛った皿が受け渡し台に置かれる', plated_on_counter_at is not None,
          f'{plated_on_counter_at:.1f}秒' if plated_on_counter_at else '40秒回しても置かず')

    ng = results.count(False)
    if ng:
        print(f'[FAILED] {ng}/{len(results)} 件が期待どおりではありません')
        return 1
    print(f'[SUCCESS] {len(results)}/{len(results)} 件すべて期待どおり')
    return 0


if __name__ == '__main__':
    sys.exit(main())
