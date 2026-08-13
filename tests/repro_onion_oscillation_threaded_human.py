"""repro_onion_oscillation_threaded.py に、agent1 がオニオン供給口付近を
うろつく「疑似人間」の動きを追加したバージョン。

人間の物理的な干渉(to_grid_a によるブロック、dynamic_obstacles ペナルティ、
頻繁な reschedule トリガー)が揺れの再現に必要かどうかを検証する。

実行方法:
    python tests/repro_onion_oscillation_threaded_human.py
"""
import os
import sys
import threading
import queue
import time
from copy import deepcopy as dcopy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)

from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from agent.agent.executor.low import EnvState
from agent.agent.myagent.CSPAgent import CSPAgent


FPS = 10
FPS_AI = 10
MAX_WALL_SECONDS = 12.0

q_env = queue.Queue()
q_ai = queue.Queue()
stop_flag = threading.Event()

trace_lock = threading.Lock()
trace = []


def log_trace(kind, **payload):
    with trace_lock:
        trace.append((time.time(), kind, payload))


# 疑似人間: agent1 を (5,1) <-> (2,1) の間で往復させ、供給口周辺(x=2..5, y=1)を通過させる
HUMAN_PATTERN = [(-1, 0)] * 4 + [(1, 0)] * 4


def run_env(env):
    seconds_per_step = 1 / FPS
    action_dict = {agent.name: None for agent in env.sim_agents}
    last_t = time.time()
    human_step = 0

    info = env.get_ai_info()
    e = EnvState(world=info['world'], agents=info['sim_agents'], agent_idx=0,
                 order=info['order_scheduler'], event_history=info['event_history'],
                 time=info['current_time'], chg_grid=info['chg_grid'])
    q_ai.put(('Env', {"EnvState": e}))

    start = time.time()
    while not stop_flag.is_set():
        if time.time() - start > MAX_WALL_SECONDS:
            stop_flag.set()
            break

        while not q_env.empty():
            event_type, args = q_env.get_nowait()
            if event_type == 'Action' and args['agent'] == 'ai_0':
                action_dict[env.sim_agents[0].name] = args['action']

        human_action = HUMAN_PATTERN[human_step % len(HUMAN_PATTERN)]
        human_step += 1
        if len(env.sim_agents) > 1:
            action_dict[env.sim_agents[1].name] = human_action

        ad = {k: v if v is not None else (0, 0) for k, v in action_dict.items()}
        pos_before = env.sim_agents[0].location
        env.step(ad, passed_time=seconds_per_step)
        pos_after = env.sim_agents[0].location
        hold_after = env.sim_agents[0].holding.full_name if env.sim_agents[0].holding else None
        human_pos = env.sim_agents[1].location if len(env.sim_agents) > 1 else None

        log_trace('env_step', pos_before=pos_before, action0=ad[env.sim_agents[0].name],
                   pos_after=pos_after, hold_after=hold_after,
                   human_action=human_action, human_pos=human_pos)

        if hold_after and 'Onion' in hold_after:
            stop_flag.set()

        info = env.get_ai_info()
        e = EnvState(world=info['world'], agents=info['sim_agents'], agent_idx=0,
                     order=info['order_scheduler'], event_history=info['event_history'],
                     time=info['current_time'], chg_grid=info['chg_grid'])
        q_ai.put(('Env', {"EnvState": dcopy(e)}))
        action_dict = {agent.name: None for agent in env.sim_agents}

        sleep_time = max(seconds_per_step - (time.time() - last_t), 0)
        last_t = time.time()
        time.sleep(sleep_time)


def run_ai(csp_agent):
    time_per_step = 1 / FPS_AI
    time_last = time.time()
    env = None
    env_update = False

    while not stop_flag.is_set():
        try:
            event = q_ai.get(timeout=0.5)
        except queue.Empty:
            continue
        while True:
            event_type, args = event
            if event_type == 'Env':
                env = args['EnvState']
                env_update = True
            if not q_ai.empty():
                event = q_ai.get_nowait()
            else:
                break

        if env_update and env is not None:
            move, _ = csp_agent(env)
            action0 = move.get('ai_0', (0, 0)) if isinstance(move, dict) else move

            sleep_time = max(time_per_step - (time.time() - time_last), 0)
            time.sleep(sleep_time)
            time_last = time.time()

            q_env.put(('Action', {"agent": "ai_0", "action": action0}))
            env_update = False


def main():
    map_set = MapSetting(level="new1", order_file="sample")
    env = OvercookedEnvironment(map_set)
    env.reset()

    csp_agent = CSPAgent(speed=10, sc_2agent=True)
    # 実際の --agent0 CSP --agent1 human --sc_2agent と同じ設定にする。
    # これがないと CSPAgent が agent1(人間) の分までタスクを計画してしまい、
    # 実プレイと異なる内部状態になる。
    csp_agent.human_counterpart_mode = True
    csp_agent.own_agent_idx = 0

    t_env = threading.Thread(target=run_env, args=(env,), daemon=True)
    t_ai = threading.Thread(target=run_ai, args=(csp_agent,), daemon=True)
    t_env.start()
    t_ai.start()
    t_env.join(timeout=MAX_WALL_SECONDS + 2)
    stop_flag.set()
    t_ai.join(timeout=2)

    with trace_lock:
        events = sorted(trace, key=lambda x: x[0])
    t0 = events[0][0] if events else time.time()
    prev_pos = None
    for t, kind, payload in events:
        if payload.get('pos_after') != prev_pos:
            print(f"t={t - t0:6.3f} {kind:10s} {payload}")
            prev_pos = payload.get('pos_after')


if __name__ == '__main__':
    main()
