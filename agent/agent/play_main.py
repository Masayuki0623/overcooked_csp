from agent.mind.agent import AgentSetting, get_agent
from agent.gameplay import GamePlay, INSTRUCTION_TIMINGS, INSTRUCTION_TIMING_FREE

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
import random
from datetime import datetime
from pathlib import Path
from agent.myagent.GreedyAgent import GreedyAgent  # 追加
from agent.myagent.ChopOnlyAgent import ChopOnlyAgent  # 追加
from agent.myagent.DualAgentController import DualAgentController  # 追加
from agent.myagent.CSPAgent import CSPAgent  # 追加
from gym_cooking.utils.order_preset import generate_order_recipes, is_order_preset, preset_names
from gym_cooking.utils.order_schedule import resolve_order_path



def parse_arguments():
    parser = argparse.ArgumentParser("Overcooked argument parser")

    parser.add_argument(
        "--map", type=str,
        choices=['ring', 'bottleneck', 'partition', 'quick', 'juice', 'experiment'], default='ring'
    )
    parser.add_argument(
        "--agent", type=str,
        choices=['HLA', 'SMOA', 'FMOA', 'NEA','Random', 'TSPSolver', 'Greedy', 'CSP', 'Task', 'choponly'], default='TSPSolver'  # CSP, Taskを追加
    )
    parser.add_argument(
        "--agent0", type=str,
        choices=['human', 'HLA', 'SMOA', 'FMOA', 'NEA', 'Random', 'TSPSolver', 'Greedy', 'CSP', 'Task', 'choponly'],
        default=None,
        help="Agent for player 0. If omitted, falls back to --agent for backward compatibility."
    )
    parser.add_argument(
        "--agent1", type=str,
        choices=['human', 'HLA', 'SMOA', 'FMOA', 'NEA', 'Random', 'TSPSolver', 'Greedy', 'CSP', 'Task', 'choponly'],
        default=None,
        help="Agent for player 1. If omitted, defaults to human unless --agent0/--agent are used."
    )
    parser.add_argument(
        "--task", type=str, default=None, help="Task to execute for TaskAgent (e.g. chop_tomato)"
    )
    parser.add_argument(
        "--no_reschedule", action='store_true', help="Disable rescheduling in CSPAgent"
    )
    parser.add_argument(
        "--sc_2agent", action='store_true', help="Enable two AI agents via CSP"
    )
    parser.add_argument(
        "--debug", action='store_true', help="Enable debug mode with overlay"
    )
    parser.add_argument(
        "--orders", "--order", dest='orders', type=str, default='sample.txt',
        help=(
            "Orders to serve. Either an order preset name "
            f"({', '.join(preset_names())}) which randomizes the ingredients, "
            "or an order definition file name/path (e.g. sample.txt)."
        )
    )
    parser.add_argument(
        "--order-seed", type=int, default=None,
        help="Random seed used when --orders names a preset (for reproducible experiments)"
    )
    parser.add_argument(
        "--instruction_request_timing", type=str,
        choices=list(INSTRUCTION_TIMINGS), default=INSTRUCTION_TIMING_FREE,
        help=(
            "When the human may instruct the AI. "
            "free: anytime via the Space key (default). "
            "enable_cook: automatically prompt the moment a cook task becomes "
            "immediately startable (Space is disabled). "
            "no_instruction: instructions are disabled. "
            "once_at_start: prompt once right after the session starts (cannot be skipped)."
        )
    )
    parser.add_argument(
        "--deadline", type=float, default=None,
        help="Hard deadline d in seconds to apply to instructed tasks (single value)"
    )

    return parser.parse_args()


def _create_single_agent(agent_name, speed, replay, init_env_state, env, task_name=None, no_reschedule=False):
    if agent_name == "TSPSolver":
        ai = TSPSolverAgent(speed, replay)
        ai._compute_all_distances(init_env_state)
        ai.extract_tasks_from_current_orders(init_env_state)
        graph = ai.generate_task_graph(init_env_state)
        # print("=== タスクグラフ（ノード, コスト） ===")
        # for node, cost in graph:
        #     print(node, ":", cost)
        # print("===============================")
        # print("=== タスク間遷移コスト ===")
        ai.print_task_transition_costs(init_env_state)
        # print("===============================")
        return ai
    if agent_name == "Greedy":
        ai = GreedyAgent(speed, replay)
        ai._compute_all_distances(init_env_state)
        ai.extract_tasks_from_current_orders(init_env_state)
        return ai
    if agent_name == "Task":
        from agent.myagent.TaskAgent import TaskAgent
        return TaskAgent(speed, replay, task_name=task_name)
    if agent_name == "choponly":
        return ChopOnlyAgent(speed, replay)
    return get_agent(AgentSetting(agent_name, speed=speed), replay)


def _iter_task_agents(ai):
    """AI が内部で持つ TaskAgent を列挙する(1体モード/2体モードの両方)。"""
    task_agent = getattr(ai, 'task_agent', None)
    if task_agent is not None:
        yield task_agent
    task_agents = getattr(ai, 'task_agents', None)
    if isinstance(task_agents, dict):
        for task_agent in task_agents.values():
            if task_agent is not None:
                yield task_agent


def resolve_orders(orders, order_seed=None):
    """--orders の値を MapSetting へ渡す形に解決する。

    プリセット名なら、ここで材料の組み合わせを選んで具体的なレシピ名まで
    確定させる。env 側でランダム化しないのは、リプレイに残るのが
    「プリセット名」ではなく「実際に出た注文」になるようにするため
    (プリセット名のまま保存すると、再生のたびに別の注文になってしまう)。
    """
    if orders is None:
        return {}
    if is_order_preset(orders):
        rng = random.Random(order_seed) if order_seed is not None else random
        recipes = generate_order_recipes(orders, rng=rng)
        seed_note = f" (seed={order_seed})" if order_seed is not None else ""
        print(f"[Orders] preset '{orders}'{seed_note}: {', '.join(recipes)}")
        return {'order_recipes': tuple(recipes)}

    # プリセット名でなければ注文ファイル。ここで存在を確かめておかないと、
    # プリセット名の打ち間違いが「ファイルが無い」という分かりにくい失敗になる。
    order_path = resolve_order_path(orders)
    if not order_path.exists():
        raise FileNotFoundError(
            f"--orders '{orders}' is neither an order preset "
            f"({', '.join(preset_names())}) nor an existing order file ({order_path})")
    return {'order_file': orders}


def init_env_replay(map_name, agent0_name, agent1_name, task_name=None, no_reschedule=False, debug_mode=False, orders=None, order_seed=None, instruction_request_timing=INSTRUCTION_TIMING_FREE):
    map_kwargs = dict(MAP_SETTINGS[map_name])
    map_kwargs.update(resolve_orders(orders, order_seed))
    map_set = MapSetting(**map_kwargs)
    replay = Replay()

    env = OvercookedEnvironment(map_set)
    env.reset()

    # ここで初期状態のEnvStateを作成
    init_env_state = EnvState(env.world, env.sim_agents, 0, env.order_scheduler, [], env.chg_grid, env.current_time)

    legacy_mode = agent1_name is None
    if legacy_mode:
        agent1_name = "human"

    if agent0_name == "human" and agent1_name == "human":
        raise ValueError("At least one of --agent0 / --agent1 must be an AI agent.")

    primary_agent_name = agent1_name if agent1_name not in (None, "human") else agent0_name
    agent_set = AgentSetting(primary_agent_name, speed=10)

    ai = None
    ai_idx = None
    human_idx = None

    use_two_agent_mode = bool(arglist.sc_2agent)

    if agent1_name == "CSP":
        if agent0_name == "choponly":
            chop_agent = ChopOnlyAgent(agent_set.speed, replay)
            csp_agent = CSPAgent(agent_set.speed, replay, no_reschedule=no_reschedule, sc_2agent=use_two_agent_mode, skip_budget=int(arglist.deadline) if arglist.deadline is not None else None)
            ai = DualAgentController(chop_agent, csp_agent)
            ai_idx = None
            human_idx = None
            try:
                # print("Skipping agent configuration GUI for CSP startup")
                csp_agent.priority_weights = {}
                csp_agent.gui_text_input = ""
                csp_agent.gui_constraint_input = ""
                csp_agent.active_constraints = []
            except Exception as e:
                # print(f"Failed to initialize default CSP settings: {e}")
                pass
        else:
            ai = CSPAgent(agent_set.speed, replay, no_reschedule=no_reschedule, sc_2agent=use_two_agent_mode, deadline_seconds=arglist.deadline, skip_budget=int(arglist.deadline) if arglist.deadline is not None else None)
            ai_idx = 1
            if agent0_name == "human":
                human_idx = 0
                ai.human_counterpart_mode = use_two_agent_mode
                ai.own_agent_idx = 1
            elif agent0_name == "CSP":
                human_idx = None
            else:
                raise NotImplementedError("--agent1 CSP currently supports only agent0 human/CSP/choponly.")
            try:
                # print("Skipping agent configuration GUI for CSP startup")
                ai.priority_weights = {}
                ai.gui_text_input = ""
                ai.gui_constraint_input = ""
                ai.active_constraints = []
            except Exception as e:
                # print(f"Failed to initialize default CSP settings: {e}")
                pass
    elif agent0_name == "CSP":
        ai = CSPAgent(agent_set.speed, replay, no_reschedule=no_reschedule, sc_2agent=use_two_agent_mode, deadline_seconds=arglist.deadline, skip_budget=int(arglist.deadline) if arglist.deadline is not None else None)
        ai_idx = 0
        if agent1_name == "human":
            human_idx = 1
            ai.human_counterpart_mode = use_two_agent_mode
            ai.own_agent_idx = 0
        else:
            raise NotImplementedError("--agent0 CSP currently supports only agent1 human.")
        try:
            # print("Skipping agent configuration GUI for CSP startup")
            ai.priority_weights = {}
            ai.gui_text_input = ""
            ai.gui_constraint_input = ""
            ai.active_constraints = []
        except Exception as e:
            # print(f"Failed to initialize default CSP settings: {e}")
            pass
    else:
        if agent0_name != "human" and agent1_name == "human":
            ai = _create_single_agent(agent0_name, agent_set.speed, replay, init_env_state, env, task_name=task_name, no_reschedule=no_reschedule)
            ai_idx = 0
            human_idx = 1
        elif agent0_name == "human" and agent1_name != "human":
            ai = _create_single_agent(agent1_name, agent_set.speed, replay, init_env_state, env, task_name=task_name, no_reschedule=no_reschedule)
            ai_idx = 1
            human_idx = 0
        else:
            raise NotImplementedError("This mode currently supports one AI and one human, or CSP on agent1/agent0.")

    game = GamePlay(env, replay, agent_set, debug_mode=debug_mode, human_agent_idx=human_idx, ai_agent_idx=ai_idx,
                    instruction_request_timing=instruction_request_timing)
    game.ai = ai
    # 詳細トレースは --debug のときだけ出す。
    # これらは1回の判断ごとに数十行を出力するため、常時ONだと実コンソールへの
    # 書き込みだけで判断1回が数百msかかり(実測: 8.5ms -> 約350ms)、
    # AI が毎フレーム動けなくなる。
    if hasattr(ai, 'debug_counter_trace'):
        ai.debug_counter_trace = debug_mode
    for task_agent in _iter_task_agents(ai):
        task_agent.debug_trace = debug_mode
    replay['set_map'] = deepcopy(map_set)
    replay['set_agent'] = deepcopy(agent_set)
    replay['order_rand'] = deepcopy(env.order_scheduler.rand_recipe_list)
    replay['chg_rand'] = deepcopy(env.chg_rand_list)

    return game, env, replay

if __name__ == '__main__':
    arglist = parse_arguments()

    agent0_name = arglist.agent0 if arglist.agent0 is not None else arglist.agent
    agent1_name = arglist.agent1

    # initialize replay
    game, env, replay = init_env_replay(arglist.map, agent0_name, agent1_name, arglist.task,
                                       arglist.no_reschedule, arglist.debug,
                                       arglist.orders, arglist.order_seed,
                                       arglist.instruction_request_timing)

    try:
        # play
        ok = game.on_execute()
        
        if ok is True:
            # print("Game End!")
            pass
        else:
            # print("Game Failed!")
            pass

    except KeyboardInterrupt:
        # print("\nGame interrupted by user.")
        pass

    finally:
        # print(replay['order_result'])
        repdir = Path(__file__).resolve().parent / 'replay'
        replay.save(repdir / f'{arglist.map}-{agent0_name}-{agent1_name or "human"}-{datetime.now().strftime("%Y%m%d_%H%M%S")}.rep')
        # print(f"Replay saved to {repdir}")
