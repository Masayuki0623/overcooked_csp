"""人間プレイヤーなしで CSPAgent(agent0) の挙動を同期的に再現するスクリプト。

gameplay.py のスレッド/キュー構造(_run_ai と _run_env の非同期性)を排除し、
1ステップずつ同期的に CSPAgent -> env.step() を呼び出すことで、
「オニオン取得中にうろちょろする」問題が TaskAgent/CSPAgent 自体のロジックに
起因するのか、それとも gameplay.py のスレッド構造に起因するのかを切り分ける。

agent1 は常に (0,0) (何もしない) に固定し、人間/他エージェントの干渉を排除する。

実行方法:
    python tests/repro_onion_oscillation.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)

from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from agent.agent.executor.low import EnvState
from agent.agent.myagent.CSPAgent import CSPAgent


def main(num_steps=300):
    map_set = MapSetting(level="new1", order_file="sample")
    env = OvercookedEnvironment(map_set)
    env.reset()

    csp_agent = CSPAgent(speed=10, sc_2agent=True)

    prev_pos = None
    prev_hold = None
    direction_changes = 0
    last_action = None

    for step in range(num_steps):
        info = env.get_ai_info()
        e = EnvState(
            world=info['world'],
            agents=info['sim_agents'],
            agent_idx=0,
            order=info['order_scheduler'],
            event_history=info['event_history'],
            time=info['current_time'],
            chg_grid=info['chg_grid'],
        )

        move, reason = csp_agent(e)
        action0 = move.get('ai_0', (0, 0)) if isinstance(move, dict) else move

        agent0 = env.sim_agents[0]
        pos = agent0.location
        hold = agent0.holding.full_name if agent0.holding is not None else None

        if pos != prev_pos or hold != prev_hold:
            print(f"step={step:4d} time={info['current_time']:.2f} pos={pos} hold={hold} action={action0}")

        if last_action is not None and action0 != (0, 0) and last_action != (0, 0):
            # 直前と正反対の方向に切り替わったら「揺れ」としてカウントする
            if action0[0] == -last_action[0] and action0[1] == -last_action[1] and action0 != last_action:
                direction_changes += 1

        if action0 != (0, 0):
            last_action = action0

        prev_pos = pos
        prev_hold = hold

        action_dict = {agent0.name: action0}
        for other in env.sim_agents[1:]:
            action_dict[other.name] = (0, 0)

        env.step(action_dict, passed_time=0.1)

        if hold is not None and 'Onion' in (hold or ''):
            print(f"\n[SUCCESS] agent0 picked up an onion-related item at step={step}: hold={hold}")
            break
    else:
        print(f"\n[FAIL] agent0 did not pick up an onion within {num_steps} steps")

    print(f"\n反転(逆方向への切り替え)回数: {direction_changes}")


if __name__ == '__main__':
    main()
