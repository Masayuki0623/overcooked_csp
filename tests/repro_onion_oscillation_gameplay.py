"""実際の GamePlay._run_ai / _run_env をそのまま使って再現・検証するスクリプト。

pygame の描画・キー入力(on_execute の _run_human)は使わず、
_run_env と _run_ai の2スレッドだけを直接起動する。
agent1 は疑似人間として往復移動させる。

実行方法:
    python tests/repro_onion_oscillation_gameplay.py
"""
import os
import sys
import threading
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from gym_cooking.utils.replay import Replay
from agent.agent.gameplay import GamePlay
from agent.agent.mind.agent import AgentSetting
from agent.agent.myagent.CSPAgent import CSPAgent

MAX_WALL_SECONDS = 12.0


def main():
    map_set = MapSetting(level="new1", order_file="sample")
    env = OvercookedEnvironment(map_set)
    env.reset()

    replay = Replay()
    agent_set = AgentSetting("CSP", speed=10)

    game = GamePlay(env, replay, agent_set, debug_mode=True,
                     human_agent_idx=1, ai_agent_idx=0, sc_2agent=True)
    csp_agent = CSPAgent(agent_set.speed, replay, sc_2agent=True)
    csp_agent.human_counterpart_mode = True
    csp_agent.own_agent_idx = 0
    game.ai = csp_agent

    if game.on_init() is False:
        print("on_init failed (pygame/display issue)")
        return

    t_env = threading.Thread(target=game._run_env, daemon=True)
    t_ai = threading.Thread(target=game._run_ai, daemon=True)
    t_env.start()
    t_ai.start()

    # 疑似人間: agent1(sim_agents[1]) を往復させる
    def human_driver():
        pattern = [(-1, 0)] * 4 + [(1, 0)] * 4
        i = 0
        start = time.time()
        while time.time() - start < MAX_WALL_SECONDS:
            game._q_env.put(('Action', {"agent": "human", "action": pattern[i % len(pattern)]}))
            i += 1
            time.sleep(0.1)

    t_human = threading.Thread(target=human_driver, daemon=True)
    t_human.start()

    start = time.time()
    picked = False
    while time.time() - start < MAX_WALL_SECONDS:
        if env.sim_agents[0].holding is not None and 'Onion' in env.sim_agents[0].holding.full_name:
            picked = True
            break
        time.sleep(0.05)

    elapsed = time.time() - start
    if picked:
        print(f"\n[RESULT] SUCCESS: picked up onion after {elapsed:.2f}s (wall)")
    else:
        print(f"\n[RESULT] FAIL: did not pick up onion within {MAX_WALL_SECONDS}s")


if __name__ == '__main__':
    main()
