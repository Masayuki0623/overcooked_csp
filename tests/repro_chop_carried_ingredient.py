"""相手が運んできた材料を、こちら側で刻めることの検証。

想定シナリオ(実測 20260924_173219 の場面を、開始位置の入れ替え(2026-09-28、
AI が右側)に合わせて鏡写しにしたもの):
    仕切りの地図。玉ねぎの供給口は人間側(左)にしかない。人間が生の玉ねぎを
    共有台まで持ってきてくれたら、AI(右側。まな板あり)には「玉ねぎを刻む」
    という仕事ができるはず。

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
from gym_cooking.utils.core import Onion, Object
from gym_cooking.utils.interact import resolve_action
from gym_cooking.utils.replay import Replay
from agent.agent.executor.low import EnvState
from agent.agent.myagent.CSPAgent import CSPAgent

ORDERS = ('OnionTomatoSalad',)
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

    obj = Object(location=SHARED_COUNTER, contents=[Onion()])
    env.world.insert(obj)
    env.world.get_gridsquare_at(SHARED_COUNTER).acquire(obj)
    print(f'[SETUP] 注文={ORDERS}')
    print(f'[SETUP] 共有台 {SHARED_COUNTER} に 生の玉ねぎ (人間が運んできた分)')
    print('[SETUP] 玉ねぎの供給口は人間側(左)のみ / AI は右側')

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
    # どちらが刻むかは所要時間で決まる。ここで見るのは「AI にも刻める
    # 材料と判定されること」(以前は供給口が自分の側に無いだけで
    # 「刻めない」と見なしていた)。
    st = state_of(env)
    orders = ai._build_order_tasks(st)
    chop = next((t for o in orders for t in o.get('tasks', [])
                 if t['id'][0] == 'chop' and t['id'][1] == 'onion'), None)
    allowed = ai._assignable_agents(st, chop) if chop else set()
    check('運ばれてきた玉ねぎは AI にも刻める材料と判定される', 0 in allowed,
          f'刻める人={sorted(allowed)} AI の担当={mine} 人の担当={other}')
    ai_has_it = any(tid and tid[0] == 'chop' and tid[1] == 'onion' for tid in mine)

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
        if any('ChoppedOnion' in n for n in names) or 'ChoppedOnion' in held:
            chopped_at = env.current_time
            break
    if ai_has_it:
        check('AI が実際に玉ねぎを刻む', chopped_at is not None,
              f'{chopped_at:.1f}秒' if chopped_at else '30秒回しても刻まず')
    else:
        print('  (この盤面では刻むのが人の担当になったので、実際に刻む確認は飛ばす)')

    print()
    print(f"[{'SUCCESS' if all(results) else 'FAIL'}] "
          f"{results.count(True)}/{len(results)} 件が期待どおり")
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
