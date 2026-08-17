"""置き場(マージ地点)の割り当てが「必要なときだけ」保持されることを検証する。

置き場は刻んだ食材を1か所に集めて鍋へ運ぶための一時的な作業場所であり、
注文が永久に所有するものではない。次の遷移を確認する:

  1. 集めている途中          -> 置き場が割り当たっている
  2. 材料が全部そろって手に取られた -> 割り当てが解除される
  3. またテーブルに置き直した       -> その場所へ割り当て直される
  4. 鍋に入った                     -> 割り当ては解除されたまま

実行方法:
    python tests/repro_merge_point_lifecycle.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from gym_cooking.utils.core import Lettuce, Onion, Tomato, Object, FoodState
from agent.agent.executor.low import EnvState
from agent.agent.myagent.CSPAgent import CSPAgent

FOOD_CLASS = {'onion': Onion, 'lettuce': Lettuce, 'tomato': Tomato}
results = []


def chopped(name):
    food = FOOD_CLASS[name]()
    food.set_state(FoodState.CHOPPED)
    return food


def check(label, actual, expect_none=None, expect_pos=None):
    if expect_none:
        ok = actual is None
        want = "解除されている"
    else:
        ok = actual == expect_pos
        want = f"{expect_pos} に割り当て"
    results.append(ok)
    mark = "OK  " if ok else "FAIL"
    print(f"  [{mark}] {label}: assigned_counter={actual} (期待: {want})")


def main():
    map_set = MapSetting(level="new1", order_file="sample")
    env = OvercookedEnvironment(map_set)
    env.reset()

    agent = CSPAgent(speed=10, sc_2agent=True)
    agent.human_counterpart_mode = True
    agent.own_agent_idx = 0
    agent.debug_counter_trace = False

    def build():
        info = env.get_ai_info()
        return EnvState(
            world=info['world'], agents=info['sim_agents'], agent_idx=0,
            order=info['order_scheduler'], event_history=info['event_history'],
            time=info['current_time'], chg_grid=info['chg_grid'],
        )

    def reschedule():
        agent._mark_reschedule_needed('test_merge_point_lifecycle')
        agent(build())

    agent(build())

    # 材料が2つ以上の注文を対象にする(マージが必要な注文でないと意味がない)
    target_uid, ings = None, None
    for entry in agent.active_order_entries:
        goal = env.order_scheduler.current_orders[entry['order_idx']][0]
        name = getattr(goal, 'full_name', '').lower()
        found = [i for i in ('lettuce', 'onion', 'tomato') if i in name]
        if len(found) >= 2:
            target_uid, ings = entry['uid'], found
            break

    if target_uid is None:
        print("[FAIL] 材料が2つ以上の注文が見つからなかった")
        return

    print(f"[SETUP] order_uid={target_uid} ingredients={ings}")

    # --- 1. 集めている途中 -> 置き場が割り当たっている ---
    counter = agent.counter_policy_by_order.get(target_uid, {}).get('counter')
    check("1. 集めている途中", counter, expect_pos=counter)
    if counter is None:
        print("[FAIL] 初期状態で置き場が割り当たっていない")
        return

    # --- 2. 材料が全部そろって手に取られた -> 解除 ---
    ai_agent = env.sim_agents[0]
    held = Object(location=ai_agent.location, contents=[chopped(i) for i in ings])
    env.world.insert(held)
    ai_agent.acquire(held)
    reschedule()
    check("2. 全部そろって手に取られた",
          agent.counter_policy_by_order.get(target_uid, {}).get('counter'),
          expect_none=True)

    # --- 3. またテーブルに置き直した -> その場所へ割り当て直される ---
    es = build()
    resources = agent._get_resources(es)
    put_back = None
    for pos in resources.get('counters', []):
        if es.pos_obj.get(pos) is None:
            put_back = pos
            break
    ai_agent.release()
    env.world.remove(held)
    placed = Object(location=put_back, contents=[chopped(i) for i in ings])
    env.world.insert(placed)
    env.world.get_gridsquare_at(put_back).acquire(placed)
    reschedule()
    check(f"3. またテーブル {put_back} に置き直した",
          agent.counter_policy_by_order.get(target_uid, {}).get('counter'),
          expect_pos=put_back)

    # --- 4. 鍋に入った -> 解除されたまま ---
    env.world.remove(placed)
    env.world.get_gridsquare_at(put_back).release()
    pot = resources.get('pots', [])[0]
    soup = Object(location=pot, contents=[chopped(i) for i in ings])
    env.world.insert(soup)
    env.world.get_gridsquare_at(pot).acquire(soup)
    reschedule()
    check("4. 鍋に入った",
          agent.counter_policy_by_order.get(target_uid, {}).get('counter'),
          expect_none=True)

    print()
    if all(results):
        print(f"[SUCCESS] 置き場のライフサイクル {len(results)}件すべて期待どおり")
    else:
        print(f"[FAIL] {results.count(False)}/{len(results)} 件が期待と異なる")


if __name__ == '__main__':
    main()
