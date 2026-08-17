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

    # 最初の食材を手にするまでの経路を記録する。
    # 検証したいのは「どの食材を取るか」ではなく「目的地まで揺れずに真っ直ぐ向かうか」。
    # どの食材から着手するかはスケジューラが makespan を見て決める内部的な判断であり、
    # ここで特定の食材に固定すると、正当なスケジュール変更で誤検知してしまう。
    start = time.time()
    picked_name = None
    path = []
    while time.time() - start < MAX_WALL_SECONDS:
        agent = env.sim_agents[0]
        if not path or path[-1] != agent.location:
            path.append(agent.location)
        if agent.holding is not None:
            picked_name = agent.holding.full_name
            break
        time.sleep(0.05)

    elapsed = time.time() - start

    if picked_name is None:
        print(f"\n[RESULT] FAIL: 食材を {MAX_WALL_SECONDS}s 以内に取得できなかった")
        return

    # 同じマスへ戻っていたら「揺れ」とみなす(本来の不具合の症状)
    revisited = [p for p in set(path) if path.count(p) > 1]
    print(f"  取得したもの: {picked_name}  経路: {path}")
    if revisited:
        print(f"\n[RESULT] FAIL: 取得までに同じマスへ戻った(揺れ) -> {revisited}")
    else:
        print(f"\n[RESULT] SUCCESS: 揺れずに {elapsed:.2f}s で {picked_name} を取得 "
              f"({len(path)} マス移動)")


if __name__ == '__main__':
    main()
