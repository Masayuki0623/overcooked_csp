"""実際の GamePlay スレッドを使ってゲームを何度も実行し、
「同じ場所への往復移動」「同じ物の拾う/置くの繰り返し」のような
無駄な重複行動を event_history から自動検出するハーネス。

agent0=CSP, agent1=疑似人間(スクリプトでランダムに動く)で、
sc_2agent モードで複数回まわし、検出されたパターンをレポートする。

実行方法:
    python tests/detect_redundant_actions.py [試行回数] [1試行あたりの秒数]
"""
import os
import sys
import threading
import time
import random

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from gym_cooking.utils.replay import Replay
from agent.agent.gameplay import GamePlay
from agent.agent.mind.agent import AgentSetting
from agent.agent.myagent.CSPAgent import CSPAgent


def detect_move_oscillation(event_history, window=20, min_repeats=3):
    """短い時間窓の中で同じ2〜3マスを何度も往復しているパターンを検出する。"""
    moves = [(e.time, e.playerA, e.location) for e in event_history if e.event == 'Move']
    findings = []
    by_agent = {}
    for t, player, loc in moves:
        by_agent.setdefault(player, []).append((t, loc))

    for player, seq in by_agent.items():
        i = 0
        while i < len(seq):
            t0, loc0 = seq[i]
            # 同じ場所へ min_repeats 回以上戻ってきているか、window 秒以内を見る
            revisits = [j for j in range(i + 1, len(seq)) if seq[j][0] - t0 <= window and seq[j][1] == loc0]
            if len(revisits) >= min_repeats:
                findings.append({
                    'type': 'move_oscillation',
                    'player': player,
                    'location': loc0,
                    'start_time': t0,
                    'revisit_count': len(revisits),
                    'end_time': seq[revisits[-1]][0],
                })
                i = revisits[-1] + 1
            else:
                i += 1
    return findings


def detect_pickup_putdown_loop(event_history, window=20, min_repeats=2):
    """同じ食材を同じ場所で拾う/置くを繰り返しているパターンを検出する。"""
    relevant = [e for e in event_history if e.event.startswith(('Pickup_', 'Put_'))]
    by_agent = {}
    for e in relevant:
        by_agent.setdefault(e.playerA, []).append(e)

    findings = []
    for player, seq in by_agent.items():
        i = 0
        while i < len(seq) - 1:
            e0 = seq[i]
            kind0 = 'Pickup' if e0.event.startswith('Pickup_') else 'Put'
            # 対応するアイテム名(Pickup_X_from_Y / Put_X_on_Y)を大まかに取り出す
            item0 = e0.event.split('_')[1] if '_' in e0.event else e0.event

            repeats = 0
            j = i + 1
            last_t = e0.time
            while j < len(seq) and seq[j].time - e0.time <= window:
                ej = seq[j]
                kindj = 'Pickup' if ej.event.startswith('Pickup_') else 'Put'
                itemj = ej.event.split('_')[1] if '_' in ej.event else ej.event
                if itemj == item0 and ej.location == e0.location and kindj != kind0:
                    repeats += 1
                    last_t = ej.time
                    kind0 = kindj
                j += 1

            if repeats >= min_repeats:
                findings.append({
                    'type': 'pickup_putdown_loop',
                    'player': player,
                    'item': item0,
                    'location': e0.location,
                    'start_time': e0.time,
                    'end_time': last_t,
                    'toggle_count': repeats,
                })
                i = j
            else:
                i += 1
    return findings


def detect_stuck_holding(snapshots, stuck_seconds=3.0):
    """agent0 が同じ物を持ったまま座標が変化しない(=置けずに詰まっている)区間を検出する。"""
    findings = []
    i = 0
    n = len(snapshots)
    while i < n:
        t0, pos0, hold0 = snapshots[i]
        if hold0 is None:
            i += 1
            continue
        j = i + 1
        while j < n and snapshots[j][1] == pos0 and snapshots[j][2] == hold0:
            j += 1
        t_end = snapshots[j - 1][0]
        if t_end - t0 >= stuck_seconds:
            findings.append({
                'type': 'stuck_holding',
                'player': 'agent-1',
                'location': pos0,
                'holding': hold0,
                'start_time': t0,
                'end_time': t_end,
                'duration': t_end - t0,
            })
        i = j
    return findings


def detect_stuck_idle(snapshots, stuck_seconds=5.0):
    """agent0 が手ぶらのまま同じ場所から動かない区間を検出する。

    human_counterpart_mode では、CSP が人間スロットへ割り当てたタスクは誰も
    実行しない。自分のタスクの前提がそこにあると AI は永久に待ち続けて停止する。
    detect_stuck_holding は「何かを持ったまま詰まる」ケースしか見ないため、
    この「手ぶらで固まる」デッドロックを拾えるよう別途検出する。
    """
    findings = []
    i = 0
    n = len(snapshots)
    while i < n:
        t0, pos0, hold0 = snapshots[i]
        if hold0 is not None:
            i += 1
            continue
        j = i + 1
        while j < n and snapshots[j][1] == pos0 and snapshots[j][2] is None:
            j += 1
        t_end = snapshots[j - 1][0]
        if t_end - t0 >= stuck_seconds:
            findings.append({
                'type': 'stuck_idle',
                'player': 'agent-1',
                'location': pos0,
                'start_time': t0,
                'end_time': t_end,
                'duration': t_end - t0,
            })
        i = j
    return findings


def run_one_trial(trial_idx, duration_seconds, seed=None):
    if seed is not None:
        random.seed(seed)

    map_set = MapSetting(level="new1", order_file="sample", max_num_timesteps=10_000)
    env = OvercookedEnvironment(map_set)
    env.reset()

    replay = Replay()
    agent_set = AgentSetting("CSP", speed=10)

    debug_mode = os.environ.get('DETECT_DEBUG') == '1'
    game = GamePlay(env, replay, agent_set, debug_mode=debug_mode,
                     human_agent_idx=1, ai_agent_idx=0, sc_2agent=True)
    csp_agent = CSPAgent(agent_set.speed, replay, sc_2agent=True)
    csp_agent.human_counterpart_mode = True
    csp_agent.own_agent_idx = 0
    # 詳細トレースは既定 OFF(常時ONだと出力だけで判断1回が数百msかかる)。
    # DETECT_DEBUG=1 のときだけ、実アプリの --debug と同じように有効化する。
    csp_agent.debug_counter_trace = debug_mode
    # DETECT_NO_HUMAN_MODEL=1 で人間タスクの推測を切る(A/B比較用)
    if os.environ.get('DETECT_NO_HUMAN_MODEL') == '1':
        csp_agent.use_predicted_human_model = False
    for task_agent in list(getattr(csp_agent, 'task_agents', {}).values()):
        task_agent.debug_trace = debug_mode
    game.ai = csp_agent

    if game.on_init() is False:
        print(f"[trial {trial_idx}] on_init failed")
        return []

    t_env = threading.Thread(target=game._run_env, daemon=True)
    t_ai = threading.Thread(target=game._run_ai, daemon=True)
    t_env.start()
    t_ai.start()

    def human_driver():
        rnd = random.Random(trial_idx * 1000 + (seed or 0))
        start = time.time()
        while time.time() - start < duration_seconds:
            action = rnd.choice([(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0), (0, 0)])
            game._q_env.put(('Action', {"agent": "human", "action": action}))
            time.sleep(0.1)

    # DETECT_NO_HUMAN=1 で疑似人間を止める。ランダムに動く人間は通路を塞ぐことが
    # あるため、残った停止が「人間の通せんぼ」由来か「AI 側のロジック」由来かを
    # 切り分けるのに使う。
    if os.environ.get('DETECT_NO_HUMAN') != '1':
        t_human = threading.Thread(target=human_driver, daemon=True)
        t_human.start()

    snapshots = []
    stop_poll = threading.Event()

    def poller():
        start = time.time()
        while not stop_poll.is_set() and time.time() - start < duration_seconds:
            a0 = env.sim_agents[0]
            hold_name = a0.holding.full_name if a0.holding is not None else None
            snapshots.append((time.time() - start, a0.location, hold_name))
            time.sleep(0.2)

    t_poll = threading.Thread(target=poller, daemon=True)
    t_poll.start()

    time.sleep(duration_seconds + 0.5)
    stop_poll.set()

    event_history = env._event_history
    findings = detect_move_oscillation(event_history) + detect_pickup_putdown_loop(event_history)
    # agent-1 = sim_agents[0] = CSP が操作する AI。agent-2(疑似人間)はランダムに動くのが
    # 仕様なので除外し、CSP 側の重複行動だけを対象にする。
    findings = [f for f in findings if f.get('player') == 'agent-1']
    findings += detect_stuck_holding(snapshots)
    findings += detect_stuck_idle(snapshots)
    print(f"[trial {trial_idx}] event_history len={len(event_history)} findings={len(findings)}")
    if findings and os.environ.get('DETECT_DUMP_EVENTS') == '1':
        print("  --- raw agent-1 Pickup_/Put_ events ---")
        for e in event_history:
            if e.playerA == 'agent-1' and e.event.startswith(('Pickup_', 'Put_')):
                print(f"    t={e.time:.2f} event={e.event} location={e.location}")
    if findings and os.environ.get('DETECT_DUMP_SNAPSHOTS') == '1':
        print("  --- agent-1 snapshots (t, pos, holding) ---")
        for t, pos, hold in snapshots:
            print(f"    t={t:5.2f} pos={pos} hold={hold}")
    for f in findings:
        print(f"    {f}")
    return findings


def main():
    # 複数試行をまとめて同一プロセス内で回すと、pygame/デーモンスレッドの終了処理が
    # 干渉して低レベルのクラッシュ(_enter_buffered_busy)が起きることがあったため、
    # 1回の起動につき1試行だけ実行する。複数試行はシェル側でプロセスを分けて回す。
    trial_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else trial_idx

    findings = run_one_trial(trial_idx, duration, seed=seed)
    print(f"\n=== TOTAL: {len(findings)} findings in trial {trial_idx} ===")


if __name__ == '__main__':
    main()
