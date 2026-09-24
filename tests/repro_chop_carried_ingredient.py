"""相手が運んできた材料を、こちら側で刻めることの検証。

想定シナリオ(実測 20260924_173219 の場面):
    仕切りの地図。バナナの供給口は人間側にしかないが、ミキサーは AI 側に
    しかない。人間が生のバナナを共有台まで持ってきてくれたら、AI には
    「バナナを刻む」という仕事が増えるはず。

    ところが AI は手を付けなかった。「刻めるかどうか」を供給口が自分の側に
    あるかだけで判定していたため、目の前の台にバナナが置いてあっても
    「自分には刻めない材料」と見なしていた。

実行方法:
    python tests/repro_chop_carried_ingredient.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from gym_cooking.utils.core import Banana, Object
from gym_cooking.utils.interact import resolve_action
from gym_cooking.utils.replay import Replay
from agent.agent.executor.low import EnvState
from agent.agent.myagent.CSPAgent import CSPAgent

ORDERS = ('BananaOrangeJuice',)
SHARED_COUNTER = (6, 4)
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

    obj = Object(location=SHARED_COUNTER, contents=[Banana()])
    env.world.insert(obj)
    env.world.get_gridsquare_at(SHARED_COUNTER).acquire(obj)
    print(f'[SETUP] 注文={ORDERS}')
    print(f'[SETUP] 共有台 {SHARED_COUNTER} に 生のバナナ (人間が運んできた分)')
    print('[SETUP] バナナの供給口は人間側のみ / ミキサーは AI 側のみ')

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
    check('運ばれてきたバナナを刻むのは AI の担当になる',
          any(tid and tid[0] == 'chop' and tid[1] == 'banana' for tid in mine),
          f'人の担当={other}')

    chopped_at = None
    for _step in range(1, 151):
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
        if any('ChoppedBanana' in n for n in names) or 'ChoppedBanana' in held:
            chopped_at = env.current_time
            break
    check('AI が実際にバナナを刻む', chopped_at is not None,
          f'{chopped_at:.1f}秒' if chopped_at else '30秒回しても刻まず')

    print()
    print(f"[{'SUCCESS' if all(results) else 'FAIL'}] "
          f"{results.count(True)}/{len(results)} 件が期待どおり")
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
