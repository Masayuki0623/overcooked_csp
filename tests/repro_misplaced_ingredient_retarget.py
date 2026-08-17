"""人間が刻んだ食材を「割り当てられていない別のカウンター」に誤って置いた場合、
CSPAgent がその注文の assigned_counter を実際の置き場所へ追従(retarget)するかを
検証するスクリプト。

同期的に1回だけスケジューリングを回し、対象注文の assigned_counter が
誤配置先のカウンターに更新されることを確認する。

実行方法:
    python tests/repro_misplaced_ingredient_retarget.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)

from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from gym_cooking.utils.core import Onion, Object, FoodState
from agent.agent.executor.low import EnvState
from agent.agent.myagent.CSPAgent import CSPAgent


def main():
    map_set = MapSetting(level="new1", order_file="sample")
    env = OvercookedEnvironment(map_set)
    env.reset()

    csp_agent = CSPAgent(speed=10, sc_2agent=True)
    csp_agent.human_counterpart_mode = True
    csp_agent.own_agent_idx = 0

    def build_env_state():
        info = env.get_ai_info()
        return EnvState(
            world=info['world'], agents=info['sim_agents'], agent_idx=0,
            order=info['order_scheduler'], event_history=info['event_history'],
            time=info['current_time'], chg_grid=info['chg_grid'],
        )

    # 1回目: 初期スケジューリングを走らせる
    csp_agent(build_env_state())

    # onion を含む注文の order_uid を、実際に割り当てられたラベルから逆引きする
    es = build_env_state()
    current_orders = env.order_scheduler.current_orders
    target_order_uid = None
    for order_idx, order_tuple in enumerate(current_orders):
        goal = order_tuple[0]
        name = getattr(goal, 'full_name', '').lower()
        if 'onion' in name:
            # _build_order_tasks 内の order_uid 決定と同じロジック(active_order_entries)を使う
            for entry in csp_agent.active_order_entries:
                if entry.get('order_idx') == order_idx:
                    target_order_uid = entry['uid']
                    break
            break

    if target_order_uid is None:
        print("[FAIL] onion を含む注文が見つからなかった")
        return

    original_counter = csp_agent.counter_policy_by_order.get(target_order_uid, {}).get('counter')
    print(f"[SETUP] order_uid={target_order_uid} original_assigned_counter={original_counter}")

    # 割り当てられていない未使用カウンターを探す
    es2 = build_env_state()
    resources = csp_agent._get_resources(es2)
    used = {e.get('counter') for e in csp_agent.counter_policy_by_order.values() if e.get('counter')}
    misplace_pos = None
    for c in resources.get('counters', []):
        if c not in used and es2.pos_obj.get(c) is None:
            misplace_pos = c
            break

    if misplace_pos is None:
        print("[FAIL] 誤配置先にできる空きカウンターが見つからなかった")
        return

    print(f"[SETUP] misplace_target_counter={misplace_pos}")

    # 人間がその注文向けの ChoppedOnion を、割り当て先ではない misplace_pos に置いたことを再現する
    onion = Onion()
    onion.set_state(FoodState.CHOPPED)
    obj = Object(location=misplace_pos, contents=[onion])
    env.world.insert(obj)
    counter_gs = env.world.get_gridsquare_at(misplace_pos)
    counter_gs.acquire(obj)

    # 2回目: 再スケジューリングを強制して、assigned_counter が追従するか確認する
    csp_agent._mark_reschedule_needed('test_misplaced_ingredient')
    csp_agent(build_env_state())

    new_entry = csp_agent.counter_policy_by_order.get(target_order_uid, {})
    new_counter = new_entry.get('counter')
    print(f"[RESULT] order_uid={target_order_uid} new_assigned_counter={new_counter}")

    if new_counter == misplace_pos:
        print("[SUCCESS] 誤配置先へ retarget された")
    elif new_counter == original_counter:
        print("[FAIL] 誤配置後も元の(空の)counterのまま変化しなかった")
    else:
        print(f"[INFO] 誤配置先とは異なる場所へ割り当てられた: {new_counter}")


if __name__ == '__main__':
    main()
