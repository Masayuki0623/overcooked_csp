from agent.mind.agent import AgentSetting
from agent.gameplay import GamePlay

from gym_cooking.misc.game.gamereplay import GamePlayReply
from gym_cooking.utils.gui import *
from gym_cooking.utils.replay import Replay
from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from agent.myagent.CSPAgent import CSPAgent

import argparse
from datetime import datetime
from pathlib import Path



def parse_arguments():
    parser = argparse.ArgumentParser("Overcooked argument parser")

    parser.add_argument("--replay", type=str)
    parser.add_argument("--calc-skillemi", action='store_true', help="Calculate skill estimation from saved raw logs")

    return parser.parse_args()


def calculate_skill_estimation_from_replay(replay):
    if 'skill_estimation_log' not in replay._d['dict']:
        raise ValueError("This replay does not contain skill estimation raw logs.")

    meta = replay._d['dict'].get('skill_estimation_meta', {})
    calculator = CSPAgent(speed=10, replay=None, sc_2agent=True, skill_emi=False)
    calculator.skill_estimation_alpha = meta.get('alpha', 0.3)
    result = calculator.calculate_skill_estimation_from_log(
        skill_estimation_log=replay['skill_estimation_log'],
        emit_logs=True,
    )
    replay['skill_estimation'] = result
    return result


def init_env_replay(replay):
    map_set = replay['set_map']
    agent_set = replay['set_agent']
    
    print(replay['order_result'])

    env = OvercookedEnvironment(map_set)
    env.reset()
    env.order_scheduler.assign_rand_recipe_list(replay['order_rand'])
    env.assign_chg_rand_list(replay['chg_rand'])

    game = GamePlayReply(env, replay)

    return game, env, replay


if __name__ == '__main__':
    arglist = parse_arguments()
    
    repdir = Path(__file__).resolve().parent / 'replay'
    replay_path = Path(arglist.replay)
    if not replay_path.is_absolute() and not replay_path.exists():
        replay_path = repdir / replay_path.name
    replay = Replay.from_file(replay_path)

    if arglist.calc_skillemi:
        result = calculate_skill_estimation_from_replay(replay)
        replay.save(replay_path)
        print(result['summary'])
        raise SystemExit(0)

    # initialize replay
    game, env, replay = init_env_replay(replay)

    # play
    ok = game.on_execute()
    
    