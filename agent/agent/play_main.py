from agent.mind.agent import AgentSetting, get_agent
from agent.gameplay import GamePlay

from gym_cooking.utils.gui import *
from gym_cooking.utils.replay import Replay
from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from gym_cooking.play_test import MAP_SETTINGS
from agent.TSP.Random_Agent import RandomAgent  # 追加
from agent.TSP.TSPSolverAgent import TSPSolverAgent  # 追加
from agent.executor.low import EnvState  # 追加
from copy import deepcopy

import os
import argparse
from datetime import datetime
from pathlib import Path
from agent.myagent.GreedyAgent import GreedyAgent  # 追加
from agent.myagent.CSPAgent import CSPAgent  # 追加



def parse_arguments():
    parser = argparse.ArgumentParser("Overcooked argument parser")

    parser.add_argument(
        "--map", type=str,
        choices=['ring', 'bottleneck', 'partition', 'quick'], default='ring'
    )
    parser.add_argument(
        "--agent", type=str,
        choices=['HLA', 'SMOA', 'FMOA', 'NEA','Random', 'TSPSolver', 'Greedy', 'CSP', 'Task'], default='TSPSolver'  # CSP, Taskを追加
    )
    parser.add_argument(
        "--task", type=str, default=None, help="Task to execute for TaskAgent (e.g. chop_tomato)"
    )
    parser.add_argument(
        "--no_reschedule", action='store_true', help="Disable rescheduling in CSPAgent"
    )
    parser.add_argument(
        "--debug", action='store_true', help="Enable debug mode with overlay"
    )

    return parser.parse_args()


def init_env_replay(map_name, agent_name, task_name=None, no_reschedule=False, debug_mode=False):
    map_set = MapSetting(**MAP_SETTINGS[map_name])
    # agent_set = AgentSetting(agent_name, speed=2.5 if map_name != 'quick' else 3.5)
    agent_set = AgentSetting(agent_name, speed=10)
    replay = Replay()

    env = OvercookedEnvironment(map_set)
    env.reset()

    # ここで初期状態のEnvStateを作成
    init_env_state = EnvState(env.world, env.sim_agents, 0, env.order_scheduler, [], env.chg_grid, env.current_time)
    # ここでエージェントを初期化（距離計算も済ませる）
    if agent_name == "TSPSolver":
        ai = TSPSolverAgent(agent_set.speed, replay)
        ai._compute_all_distances(init_env_state)
        ai.extract_tasks_from_current_orders(init_env_state)
        # グラフの出力
        graph = ai.generate_task_graph(init_env_state)
        print("=== タスクグラフ（ノード, コスト） ===")
        for node, cost in graph:
            print(node, ":", cost)
        print("===============================")
        # タスク間遷移コストの出力
        print("=== タスク間遷移コスト ===")
        ai.print_task_transition_costs(init_env_state)
        print("===============================")
    elif agent_name == "Greedy":
        ai = GreedyAgent(agent_set.speed, replay)
        ai._compute_all_distances(init_env_state)
        ai.extract_tasks_from_current_orders(init_env_state)
    elif agent_name == "CSP":
        ai = CSPAgent(agent_set.speed, replay, no_reschedule=no_reschedule)
        # CSPエージェントの初期化（将来的に制約の構築などを行う）
        try:
            from agent.myagent.gui import configure_agent_settings
            print("Opening Agent Configuration GUI...")
            settings = configure_agent_settings(env)
            ai.priority_weights = settings['weights']
            ai.gui_text_input = settings['text_input'] # 将来の使用のために保存
            print("Settings configured:", settings)
        except Exception as e:
            print(f"Failed to configure settings via GUI: {e}")
    elif agent_name == "Task":
        from agent.myagent.TaskAgent import TaskAgent
        ai = TaskAgent(agent_set.speed, replay, task_name=task_name)
    else:
        ai = get_agent(agent_set, replay)
    game = GamePlay(env, replay, agent_set, debug_mode=debug_mode)
    game.ai = ai
    replay['set_map'] = deepcopy(map_set)
    replay['set_agent'] = deepcopy(agent_set)
    replay['order_rand'] = deepcopy(env.order_scheduler.rand_recipe_list)
    replay['chg_rand'] = deepcopy(env.chg_rand_list)

    return game, env, replay

if __name__ == '__main__':
    arglist = parse_arguments()

    # initialize replay
    game, env, replay = init_env_replay(arglist.map, arglist.agent, arglist.task, arglist.no_reschedule, arglist.debug)

    try:
        # play
        ok = game.on_execute()
        
        if ok is True:
            print("Game End!")
        else:
            print("Game Failed!")

    except KeyboardInterrupt:
        print("\nGame interrupted by user.")

    finally:
        print(replay['order_result'])
        repdir = Path(__file__).resolve().parent / 'replay'
        replay.save(repdir / f'{arglist.map}-{arglist.agent}-{datetime.now().strftime("%Y%m%d_%H%M%S")}.rep')
        print(f"Replay saved to {repdir}")
