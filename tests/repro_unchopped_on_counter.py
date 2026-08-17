"""人間が「切っていない食材」を注文の置き場(assigned_counter)に置いた場合に、
AI が置けない操作を延々と繰り返して停止しないことを検証するスクリプト。

ゲームエンジンの mergeable() は ChoppedX 同士でなければマージを許さないが、
CSP 側の食材名は Fresh/Chopped の区別を落とす(どちらも 'onion')ため、
未カット食材が置き場にあると「必要な食材が既に置いてある」と誤認したまま
AI がマージできない置く操作を繰り返して固まる不具合があった。

置き場が解除され、別の使えるカウンターへ retarget されることを確認する。

実行方法:
    python tests/repro_unchopped_on_counter.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from gym_cooking.utils.core import Onion, Lettuce, Tomato, Object, mergeable
from agent.agent.executor.low import EnvState
from agent.agent.myagent.CSPAgent import CSPAgent

FRESH_CLASS = {'onion': Onion, 'lettuce': Lettuce, 'tomato': Tomato}


def main():
    map_set = MapSetting(level="new1", order_file="sample")
    env = OvercookedEnvironment(map_set)
    env.reset()

    csp_agent = CSPAgent(speed=10, sc_2agent=True)
    csp_agent.human_counterpart_mode = True
    csp_agent.own_agent_idx = 0
    csp_agent.debug_counter_trace = False

    def build_env_state():
        info = env.get_ai_info()
        return EnvState(
            world=info['world'], agents=info['sim_agents'], agent_idx=0,
            order=info['order_scheduler'], event_history=info['event_history'],
            time=info['current_time'], chg_grid=info['chg_grid'],
        )

    csp_agent(build_env_state())

    # 材料が2つ以上ある注文を選ぶ(マージが必要な注文でないと再現しない)
    target_uid, target_ings = None, None
    for entry in csp_agent.active_order_entries:
        order_idx = entry.get('order_idx')
        goal = env.order_scheduler.current_orders[order_idx][0]
        name = getattr(goal, 'full_name', '').lower()
        ings = [i for i in ('lettuce', 'onion', 'tomato') if i in name]
        if len(ings) >= 2:
            target_uid, target_ings = entry['uid'], ings
            break

    if target_uid is None:
        print("[FAIL] 材料が2つ以上の注文が見つからなかった")
        return

    blocked_counter = csp_agent.counter_policy_by_order.get(target_uid, {}).get('counter')
    if blocked_counter is None:
        print("[FAIL] 対象注文に assigned_counter が割り当たっていない")
        return

    fresh_ing = target_ings[0]
    print(f"[SETUP] order_uid={target_uid} ingredients={target_ings}")
    print(f"[SETUP] assigned_counter={blocked_counter} に未カットの {fresh_ing} を置く")

    # 人間が「切っていない食材」をその置き場に置いた状況を再現する
    fresh_food = FRESH_CLASS[fresh_ing]()
    obj = Object(location=blocked_counter, contents=[fresh_food])
    env.world.insert(obj)
    env.world.get_gridsquare_at(blocked_counter).acquire(obj)

    # 前提の確認: この状態では実際にマージできない(=AIは永久に置けない)
    chopped_other = FRESH_CLASS[target_ings[1]]()
    from gym_cooking.utils.core import FoodState
    chopped_other.set_state(FoodState.CHOPPED)
    held = Object(location=(0, 0), contents=[chopped_other])
    print(f"[SETUP] mergeable(Chopped{target_ings[1].capitalize()}, "
          f"Fresh{fresh_ing.capitalize()}) = {mergeable(held, obj)}")

    # 解除には2秒のヒステリシスがあるため、時間を進めながら数回まわす
    action_dict = {agent.name: (0, 0) for agent in env.sim_agents}
    for _ in range(60):
        env.step(action_dict, passed_time=0.1)
        csp_agent._mark_reschedule_needed('test_unchopped_on_counter')
        csp_agent(build_env_state())
        new_counter = csp_agent.counter_policy_by_order.get(target_uid, {}).get('counter')
        if new_counter != blocked_counter:
            break

    new_counter = csp_agent.counter_policy_by_order.get(target_uid, {}).get('counter')
    print(f"[RESULT] order_uid={target_uid} assigned_counter: "
          f"{blocked_counter} -> {new_counter}")

    if new_counter == blocked_counter:
        print("[FAIL] 未カット食材が乗ったままの置き場が解除されなかった "
              "(AIはここで永久に置こうとし続ける)")
    elif new_counter is None:
        print("[INFO] 置き場は解除されたが、新しい置き場がまだ決まっていない")
    elif csp_agent._get_counter_blocking_food_names(build_env_state(), new_counter):
        print(f"[FAIL] retarget 先 {new_counter} にも未カット食材が乗っている")
    else:
        print(f"[SUCCESS] 使える別の置き場 {new_counter} へ retarget された")


if __name__ == '__main__':
    main()
