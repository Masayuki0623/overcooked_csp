from agent.mind.agent import AgentSetting, get_agent
from agent.gameplay import GamePlay

from gym_cooking.utils.gui import *
from gym_cooking.utils.replay import Replay
from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from gym_cooking.play_test import MAP_SETTINGS
from agent.TSP.Random_Agent import RandomAgent
from agent.TSP.TSPSolverAgent import TSPSolverAgent
from agent.executor.low import EnvState
from copy import deepcopy

import os
import argparse
from datetime import datetime
from pathlib import Path
from agent.myagent.GreedyAgent import GreedyAgent
from agent.myagent.CSPAgent import CSPAgent
from agent.myagent.BreadthFirstSearchAgent import BreadthFirstSearchAgent
from agent.executor.low import EnvState


def parse_arguments():
    parser = argparse.ArgumentParser("Overcooked argument parser")

    parser.add_argument(
        "--map", type=str,
        choices=['ring', 'bottleneck', 'partition', 'quick'], default='ring'
    )
    parser.add_argument(
        "--agent", type=str,
        choices=['HLA', 'SMOA', 'FMOA', 'NEA','Random', 'TSPSolver', 'Greedy', 'CSP', 'Task', 'BFS'], default='TSPSolver'
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
    agent_set = AgentSetting(agent_name, speed=10)
    replay = Replay()

    env = OvercookedEnvironment(map_set)
    env.reset()

    # aiのインスタンス化はここで行う
    ai = None
    if agent_name == "TSPSolver":
        init_env_state = EnvState(env.world, env.sim_agents, 0, env.order_scheduler, [], env.chg_grid, env.current_time)
        ai = TSPSolverAgent(agent_set.speed, replay)
        ai._compute_all_distances(init_env_state)
        ai.extract_tasks_from_current_orders(init_env_state)
        graph = ai.generate_task_graph(init_env_state)
        print("=== タスクグラフ（ノード, コスト） ===")
        for node, cost in graph:
            print(node, ":", cost)
        print("===============================")
        print("=== タスク間遷移コスト ===")
        ai.print_task_transition_costs(init_env_state)
        print("===============================")
    elif agent_name == "Greedy":
        init_env_state = EnvState(env.world, env.sim_agents, 0, env.order_scheduler, [], env.chg_grid, env.current_time)
        ai = GreedyAgent(agent_set.speed, replay)
        ai._compute_all_distances(init_env_state)
        ai.extract_tasks_from_current_orders(init_env_state)
    elif agent_name == "CSP":
        ai = CSPAgent(agent_set.speed, replay, no_reschedule=no_reschedule)
        try:
            from agent.myagent.gui import configure_agent_settings
            print("Opening Agent Configuration GUI...")
            settings = configure_agent_settings(env)
            ai.priority_weights = settings['weights']
            ai.gui_text_input = settings['text_input']
            ai.gui_constraint_input = settings.get('constraint_input', "")
            ai.active_constraints = settings.get('constraints', [])
            print("Settings configured:", settings)
        except Exception as e:
            print(f"Failed to configure settings via GUI: {e}")
    elif agent_name == "BFS":
        ai = BreadthFirstSearchAgent(agent_set.speed, replay)
    elif agent_name == "Task":
        from agent.myagent.TaskAgent import TaskAgent
        ai = TaskAgent(agent_set.speed, replay, task_name=task_name)
    else:
        ai = get_agent(agent_set, replay)

    game = GamePlay(env, replay, agent_set, debug_mode=debug_mode)
    game.ai = ai  # GamePlayのaiをここで設定
    replay['set_map'] = deepcopy(map_set)
    replay['set_agent'] = deepcopy(agent_set)
    replay['order_rand'] = deepcopy(env.order_scheduler.rand_recipe_list)
    replay['chg_rand'] = deepcopy(env.chg_rand_list)

    return game, env, replay

if __name__ == '__main__':
    arglist = parse_arguments()

    game, env, replay = init_env_replay(arglist.map, arglist.agent, arglist.task, arglist.no_reschedule, arglist.debug)

    # BFSエージェントの場合、ゲーム開始前にプランニングを行う
    if arglist.agent == 'BFS':
        print("\nBFSエージェントの事前プランニングを開始します...")
        
        info = env.get_ai_info()
        initial_env_state = EnvState(
            world=info['world'],
            agents=info['sim_agents'],
            agent_idx=0, 
            order=info['order_scheduler'],
            event_history=info['event_history'],
            time=info['current_time'],
            chg_grid=info['chg_grid']
        )
        
        game.ai.current_plan = game.ai.bfs(initial_env_state)

        if game.ai.current_plan:
            print(f"事前プランニングが完了しました。ステップ数: {len(game.ai.current_plan)}\n")
        else:
            print("プランが見つかりませんでした。\n")
            game.ai.current_plan = []
        
        # __call__が呼ばれたときに再度bfsを実行しないように、ステップを0に設定
        game.ai.current_step = 0

    try:
        ok = game.on_execute()
        
        if ok is True:
            print("Game End!")
        else:
            print("Game Failed!")

    except KeyboardInterrupt:
        print("\nGame interrupted by user.")

    finally:
        # order_resultが生成される前にエラー終了する場合があるためチェック
        if 'order_result' in replay._d['dict']:
            print(replay['order_result'])
        
        repdir = Path(__file__).resolve().parent / 'replay'
        if not repdir.exists():
            repdir.mkdir()
        replay.save(repdir / f'{arglist.map}-{arglist.agent}-{datetime.now().strftime("%Y%m%d_%H%M%S")}.rep')
        print(f"Replay saved to {repdir}")
