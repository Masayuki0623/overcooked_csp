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
    # Add arguments for number of AI and Human agents
    parser.add_argument(
        "--num_ai", type=int, default=1, help="Number of AI agents (default: 1)"
    )
    parser.add_argument(
        "--num_human", type=int, default=1, help="Number of Human agents (default: 1)"
    )
    # Support for the specific flag format mentioned by the user
    # We'll handle this in the main block by checking sys.argv or just add a catch-all if needed
    # But usually better to just use standard args. 
    # Let's add a convenience argument to parse string like "ai_2-h_0"
    parser.add_argument(
        "--agent_config", type=str, default=None, help="Agent config string (e.g., ai_2-h_0)"
    )

    return parser.parse_args()


def init_env_replay(map_name, agent_name, task_name=None, no_reschedule=False, debug_mode=False, num_ai=1, num_human=1):
    map_set = MapSetting(**MAP_SETTINGS[map_name])
    map_set.num_agents = num_ai + num_human  # Update total agents
    
    # agent_set = AgentSetting(agent_name, speed=2.5 if map_name != 'quick' else 3.5)
    agent_set = AgentSetting(agent_name, speed=10)
    replay = Replay()

    env = OvercookedEnvironment(map_set)
    env.reset()

    # ここで初期状態のEnvStateを作成
    init_env_state = EnvState(env.world, env.sim_agents, 0, env.order_scheduler, [], env.chg_grid, env.current_time)
    
    # Initialize Agent
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
        # Pass num_ai to CSPAgent so it knows how many agents to schedule for
        ai = CSPAgent(agent_set.speed, replay, no_reschedule=no_reschedule, num_agents=num_ai)
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

    # Pass agent counts to GamePlay
    game = GamePlay(env, replay, agent_set, debug_mode=debug_mode, num_ai=num_ai, num_human=num_human)
    game.ai = ai
     
    replay['set_map'] = deepcopy(map_set)
    replay['set_agent'] = deepcopy(agent_set)
    replay['order_rand'] = deepcopy(env.order_scheduler.rand_recipe_list)
    replay['chg_rand'] = deepcopy(env.chg_rand_list)

    return game, env, replay

if __name__ == '__main__':
    arglist = parse_arguments()
    
    # helper for parsing agent_config like "ai_2-h_0"
    if arglist.agent_config:
        import re
        match = re.search(r'ai_(\d+)-h_(\d+)', arglist.agent_config)
        if match:
            arglist.num_ai = int(match.group(1))
            arglist.num_human = int(match.group(2))
            print(f"Parsed agent config: {arglist.num_ai} AI, {arglist.num_human} Human")

    # initialize replay
    game, env, replay = init_env_replay(
        arglist.map, arglist.agent, arglist.task, arglist.no_reschedule, arglist.debug,
        num_ai=arglist.num_ai, num_human=arglist.num_human
    )

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
        if 'order_result' in replay._d['dict']:
            print(replay['order_result'])
        
        repdir = Path(__file__).resolve().parent / 'replay'
        if not repdir.exists():
            repdir.mkdir()
        replay.save(repdir / f'{arglist.map}-{arglist.agent}-{datetime.now().strftime("%Y%m%d_%H%M%S")}.rep')
        print(f"Replay saved to {repdir}")
