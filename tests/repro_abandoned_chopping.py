"""まな板に切りかけの材料を残したまま止まらないことの検証。

想定シナリオ(バグ報告 20260923_215425 の場面):
    仕切りの地図。レタスは人間側にしか無いが、人が生のレタスを共有台へ
    置いてくれたので、AI はそれを取って自分のまな板に乗せ、切り始めた。

    ところが計画を立て直した瞬間、「レタスを切る」工程は人間の担当に
    振られてしまう。切りかけの物は Fresh でも Chopped でもないため
    計画からは見えず、レタスは供給口(人間側)から取ってくるものだと
    見なされるからである。切りかけはこちら側のまな板の上にあって人間の
    手は届かないので、誰も手を付けないまま試合が終わる(実測47秒停止)。

実行方法:
    python tests/repro_abandoned_chopping.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from gym_cooking.utils.core import Lettuce, Object, FoodState
from gym_cooking.utils.interact import resolve_action
from gym_cooking.utils.replay import Replay
from agent.agent.executor.low import EnvState
from agent.agent.myagent.CSPAgent import CSPAgent

ORDERS = ('TomatoLettuceSoup', 'OnionTomatoSalad')
results = []


def check(label, ok, detail=""):
    results.append(ok)
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}{(' -> ' + detail) if detail else ''}")


def state_of(env):
    i = env.get_ai_info()
    return EnvState(world=i['world'], agents=i['sim_agents'], agent_idx=0,
                    order=i['order_scheduler'], event_history=i['event_history'],
                    time=i['current_time'], chg_grid=i['chg_grid'])


def main():
    env = OvercookedEnvironment(MapSetting(level='exp_partition',
                                           order_recipes=ORDERS))
    env.reset()

    st = state_of(env)
    ai_pos = tuple(env.sim_agents[0].location)
    boards = [tuple(b) for b in st.get_pos_by_obj_gs(gs='Cutboard')]
    # AI 側(仕切りのこちら側)のまな板を選ぶ
    board = min(boards, key=lambda b: abs(b[0] - ai_pos[0]) + abs(b[1] - ai_pos[1]))

    half = Lettuce()
    half.set_state(FoodState.CHOPPING)
    obj = Object(location=board, contents=[half])
    env.world.insert(obj)
    env.world.get_gridsquare_at(board).acquire(obj)
    print(f'[SETUP] 注文={ORDERS}')
    print(f'[SETUP] AI 側のまな板 {board} に 切りかけのレタス '
          f'({obj.full_name}) が残っている')
    print(f'[SETUP] レタスの供給口は人間側にしかない')

    ai = CSPAgent(10, Replay(), sc_2agent=True, skip_budget=2)
    ai.human_counterpart_mode = True
    ai.own_agent_idx = 0
    ai.priority_weights = {}
    ai.gui_text_input = ''
    ai.gui_constraint_input = ''
    ai.active_constraints = []
    ai(state_of(env))

    mine = [t.get('id') for t in ((ai.schedule_per_agent or {}).get(0) or [])]
    other = [t.get('id') for t in ((ai.schedule_per_agent or {}).get(1) or [])]
    print('  AI の担当:', mine)
    print('  人の担当  :', other)
    chop_lettuce = [tid for tid in mine if tid and tid[0] == 'chop' and tid[1] == 'lettuce']
    check('切りかけのレタスは AI の担当になる', bool(chop_lettuce),
          f'人の担当={other}')

    # 実際に切り終えられるか
    done_at = None
    for step in range(1, 121):
        move, _reason = ai(state_of(env))
        own = move.get('ai_0') if isinstance(move, dict) else move
        acts = {a.name: (0, 0) for a in env.sim_agents}
        if own:
            acts[env.sim_agents[0].name] = own
        for a in env.sim_agents:
            acts[a.name] = resolve_action(a, env.world, acts[a.name])
        env.step(acts, passed_time=0.2)
        names = [getattr(o, 'full_name', '') or ''
                 for o in state_of(env).pos_obj.values() if o is not None]
        held = getattr(env.sim_agents[0].holding, 'full_name', '') or ''
        if any('ChoppedLettuce' in n for n in names) or 'ChoppedLettuce' in held:
            done_at = env.current_time
            break
    check('切りかけのレタスを切り終える', done_at is not None,
          f'{done_at:.1f}秒' if done_at else '24秒回しても切り終えず')

    print()
    print(f"[{'SUCCESS' if all(results) else 'FAIL'}] "
          f"{results.count(True)}/{len(results)} 件が期待どおり")
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
