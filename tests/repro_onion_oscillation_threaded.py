"""gameplay.py の _run_ai / _run_env と同じスレッド+キュー構造だけを抽出し、
pygame/人間操作なしで再現するスクリプト。

同期版(repro_onion_oscillation.py)では揺れが再現しなかったため、
gameplay.py の非同期スレッド構造そのものが原因かどうかを検証する。
agent1 は常に (0,0) に固定して人間の影響を排除する。

実行方法:
    python tests/repro_onion_oscillation_threaded.py
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


FPS = 10          # _run_env の速度 (gameplay.py と同じ)
FPS_AI = 10        # _run_ai の速度 (play_main.py の agent_set.speed=10 と同じ)
MAX_WALL_SECONDS = 8.0

q_env = queue.Queue()
q_ai = queue.Queue()
stop_flag = threading.Event()

# 診断用ログ: (wall_time, kind, payload) を時系列で溜める
trace_lock = threading.Lock()
trace = []


def log_trace(kind, **payload):
    with trace_lock:
        trace.append((time.time(), kind, payload))


def run_env(env):
    seconds_per_step = 1 / FPS
    action_dict = {agent.name: None for agent in env.sim_agents}
    last_t = time.time()

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
                log_trace('env_recv_action', action=args['action'])

        ad = {k: v if v is not None else (0, 0) for k, v in action_dict.items()}
        pos_before = env.sim_agents[0].location
        log_trace('env_step_begin', pos_before=pos_before, action_applied=ad[env.sim_agents[0].name])
        env.step(ad, passed_time=seconds_per_step)
        pos_after = env.sim_agents[0].location
        hold_after = env.sim_agents[0].holding.full_name if env.sim_agents[0].holding else None
        log_trace('env_step_end', pos_after=pos_after, hold_after=hold_after)

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
            pos_seen = env.self_pos
            log_trace('ai_decide_begin', pos_seen=pos_seen)
            move, _ = csp_agent(env)
            action0 = move.get('ai_0', (0, 0)) if isinstance(move, dict) else move
            log_trace('ai_decide_end', pos_seen=pos_seen, action=action0)

            sleep_time = max(time_per_step - (time.time() - time_last), 0)
            time.sleep(sleep_time)
            time_last = time.time()

            q_env.put(('Action', {"agent": "ai_0", "action": action0}))
            log_trace('ai_push_action', pos_seen=pos_seen, action=action0)
            env_update = False


def main():
    map_set = MapSetting(level="new1", order_file="sample")
    env = OvercookedEnvironment(map_set)
    env.reset()

    csp_agent = CSPAgent(speed=10, sc_2agent=True)

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
    for t, kind, payload in events:
        print(f"t={t - t0:6.3f} {kind:16s} {payload}")


if __name__ == '__main__':
    main()
