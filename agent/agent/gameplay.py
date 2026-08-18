# modules for game
from gym_cooking.misc.game.game import Game
from gym_cooking.misc.game.utils import *
from gym_cooking.utils.gui import popup_text, popup_task_choice
from gym_cooking.utils.replay import Replay
from agent.executor.low import EnvState
from agent.mind.agent import get_agent, AgentSetting
from agent.instruction_panel import InstructionPanel

# helpers
import pygame
import threading
import queue
import time
import logging
import sys
import os
from datetime import datetime

from copy import deepcopy as dcopy

# import vosk
# from vosk import Model, KaldiRecognizer
# import pyaudio
# import os


# def speak(text: str):
#     def _speak(text: str):
#         from win32com.client import Dispatch
#         speaker = Dispatch("SAPI.SpVoice")
#         speaker.Rate = 5
#         speaker.Speak(text)
#         pass
#
#     threading.Thread(target=_speak, args=(text,)).start()


class _TeeWriter:
    """標準出力とファイルの両方に書き出すラッパー"""
    def __init__(self, stdout, logfile):
        self.stdout = stdout
        self.logfile = logfile

    def write(self, text):
        self.stdout.write(text)
        self.logfile.write(text)

    def flush(self):
        self.stdout.flush()
        self.logfile.flush()


class _SelectiveWriter:
    """特定プレフィックスの行だけを通すラッパー"""
    def __init__(self, stream, allowed_prefixes):
        self.stream = stream
        self.allowed_prefixes = allowed_prefixes
        self._buffer = ""

    def _should_emit(self, line):
        stripped = line.lstrip()
        return any(stripped.startswith(prefix) for prefix in self.allowed_prefixes)

    def write(self, text):
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if self._should_emit(line):
                self.stream.write(line + "\n")

    def flush(self):
        if self._buffer:
            if self._should_emit(self._buffer):
                self.stream.write(self._buffer)
            self._buffer = ""
        self.stream.flush()


# AI は「送った行動が適用された」ことを次の状態pushで確認してから次を送るため、
# 1行動につき判断時間+往復ぶんが必要になる。環境の1ステップをAIの判断時間の
# 何倍まで引き伸ばすか(--debug で進行を遅くするときにだけ使う)。
AI_PACE_ROUNDTRIP_FACTOR = 2.0

# 入力イベントを拾うループの待ち時間(秒)。500Hz。
HUMAN_POLL_INTERVAL = 0.002

# 指示を出せるタイミング(--instruction_request_timing)。
#   free           : 従来どおり、人間が好きなときに Space で指示できる
#   enable_cook    : 調理タスクに今すぐ着手できる状態になった瞬間、自動で指示画面を出す。
#                    タイミングを実験条件として固定するため Space での任意呼び出しは無効
#   no_instruction : 指示を出せない
INSTRUCTION_TIMING_FREE = 'free'
INSTRUCTION_TIMING_ENABLE_COOK = 'enable_cook'
INSTRUCTION_TIMING_NO_INSTRUCTION = 'no_instruction'
INSTRUCTION_TIMINGS = (
    INSTRUCTION_TIMING_FREE,
    INSTRUCTION_TIMING_ENABLE_COOK,
    INSTRUCTION_TIMING_NO_INSTRUCTION,
)


class GamePlay(Game):
    def __init__(self, env, replay: Replay, agent_set: AgentSetting, debug_mode: bool = False, human_agent_idx: int | None = 1, ai_agent_idx: int | None = 0, sc_2agent: bool = False, instruction_request_timing: str = INSTRUCTION_TIMING_FREE):
        Game.__init__(self, env, play=True)
        self.replay = replay
        self.agent_set = agent_set
        self.debug_mode = debug_mode
        if instruction_request_timing not in INSTRUCTION_TIMINGS:
            raise ValueError(
                f"Unknown instruction_request_timing: {instruction_request_timing} "
                f"(available: {', '.join(INSTRUCTION_TIMINGS)})")
        self.instruction_request_timing = instruction_request_timing
        self.human_agent_idx = human_agent_idx
        self.ai_agent_idx = ai_agent_idx
        self.sc_2agent = sc_2agent

        # デバッグモード時はログをファイルに出力
        if debug_mode:
            log_dir = os.path.join(os.path.dirname(__file__), 'logs')
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_path = os.path.join(log_dir, f'debug_{timestamp}.log')
            # 標準出力をファイルとターミナル両方に出力する
            self._log_file = open(log_path, 'w', encoding='utf-8', buffering=1)
            self._original_stdout = sys.stdout
            sys.stdout = _TeeWriter(sys.stdout, self._log_file)
            print(f'=== Debug log: {log_path} ===')
        else:
            self._log_file = None

        # fps of human and ai
        self.fps = 10
        self.fps_ai = agent_set.speed

        # --debug では詳細トレースの出力だけで AI の判断1回が数百msかかる。
        # 環境をそのまま 1/fps 秒ごとに進めると AI が数フレームに1回しか動けず、
        # 「AIだけが極端に遅い」状態になってしまう。
        # そこで AI の実際の判断時間を測り、環境がそれより速く進まないようにする。
        # env.step へ渡すゲーム内時間(passed_time)は変えないので、調理時間などの
        # ゲームロジックは一切変わらず、実時間の進行だけが遅くなる。
        self.pace_env_to_ai = debug_mode
        # AI の判断時間(指数移動平均, 秒)。_run_ai が更新し _run_env が読む。
        self._ai_decide_seconds = 0.0
        # 遅くする上限(1ステップが 1/fps 秒の何倍まで伸びてよいか)。
        # これを超えると体感で止まって見えるため頭打ちにする。
        self.max_pace_stretch = 8.0

        # human_agent_idx は None の場合もある（両方AIの旧モードなど）
        self.idx_human = human_agent_idx
        self.ai = get_agent(self.agent_set, self.replay)

        # concurrent control variables
        self._q_control = queue.Queue()  # receive
        self._q_env = queue.Queue()
        self._q_ai = queue.Queue()
        self._success = False
        self._finalized = False
        self._latest_env_state = None
        # 指示パネル表示中は _run_env スレッド側の描画を止める。
        # on_render は screen.fill -> display.flip まで行うため、パネルの描画と
        # 交互に画面全体を上書きし合って激しく点滅してしまう。
        self._instruction_panel_active = False

        # 環境ループ1周の実測(Web版の /api/perf から読む)。
        self.loop_stats = {'work_s': 0.0, 'period_s': 0.0, 'last_top': None}
        # enable_cook の立ち上がり検出用。「前回見たときに着手可能だった cook タスク」。
        # 同じタスクで何度も指示画面を出さないよう、集合の差分でだけ発火させる。
        self._seen_ready_cook_actions = set()

    def _get_unexecuted_task_candidates(self):
        env_state = self._latest_env_state
        if env_state is None:
            return []

        try:
            snapshot = dcopy(env_state)
            if hasattr(self.ai, 'get_instruction_candidates'):
                return self.ai.get_instruction_candidates(snapshot)
        except Exception as e:
            print(f"[GamePlay] 候補タスク取得失敗: {e}")

        return []

    def _build_env_summary(self):
        """指示パネルの帯に出す情報。

        座標のような細かい数値ではなく「どちらのキャラクターがAIで、
        どちらがあなたか」をゲーム内の見た目そのままで示す。
        """
        players = []
        try:
            for idx, agent in enumerate(self.sim_agents):
                color = getattr(agent, 'color', None)
                players.append({
                    'name': 'あなた' if idx == self.idx_human else 'AI',
                    # ゲーム画面と同じキャラ画像を使う (misc/game/graphics/agent-<色>.png)
                    'sprite': f'agent-{color}' if color else None,
                })
        except Exception:
            pass
        return {'players': players}

    def _show_instruction_panel(self, candidates):
        """スペース押下時の指示カード画面。選択中だけ窓を横に広げる。"""
        # 先に描画スレッドを止めてから、自分で完全な1フレームを描いてコピーする。
        # on_render は screen.fill してから全オブジェクトを描き直すため、その途中で
        # copy() すると提供口やプレイヤーが欠けたスナップショットになってしまう。
        self._instruction_panel_active = True
        try:
            time.sleep(1.0 / max(self.fps, 1))  # 進行中の描画が終わるのを待つ
            self.on_render(paused=1)
            snapshot = self.screen.copy()
        except Exception as e:
            self._instruction_panel_active = False
            print(f"[GamePlay] 指示パネルの描画に失敗したためテキスト入力に切り替えます: {e}")
            return popup_task_choice("AIへの指示タスクを選択してください", candidates)

        old_size = (snapshot.get_width(), snapshot.get_height())
        # カードが3列でも窮屈にならない幅を確保する(選択中だけ広げるので実害はない)
        panel_width = max(360, old_size[0])
        try:
            widened = pygame.display.set_mode((old_size[0] + panel_width, old_size[1]))
            panel = InstructionPanel(candidates, env_summary=self._build_env_summary())
            return panel.run(widened, snapshot)
        finally:
            self._instruction_panel_active = False
            # 閉じたら元の窓サイズへ戻し、ゲーム画面を描き直す
            self.screen = pygame.display.set_mode(old_size)
            self.screen.blit(snapshot, (0, 0))
            pygame.display.flip()

    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._q_control.put(('Quit', {}))

        elif event.type == pygame.KEYDOWN:
            if event.key in KeyToTuple.keys():
                # Control
                action_dict = {agent.name: (0, 0) for agent in self.sim_agents}
                action = KeyToTuple[event.key]
                if self.idx_human is not None and self.idx_human < len(self.sim_agents):
                    action_dict[self.sim_agents[self.idx_human].name] = action
                self._q_env.put(
                    ('Action', {"agent": "human", "action": action}))
                self._q_ai.put(
                    ('Action', {"agent": "human", "action": action}))

            if pygame.key.name(event.key) == "space":
                if self.instruction_request_timing != INSTRUCTION_TIMING_FREE:
                    # free 以外はタイミングを実験条件として固定しているので、
                    # 人間の任意呼び出しは受け付けない。
                    return
                self._request_instruction(trigger='space')

    def _start_time_loss_estimation(self, pending_entry, display_text):
        """指示による時間損失量 L(d) = f'(d) - f を別スレッドで計算する。

        f      : その指示の制約なしで解いた最適 makespan
        f'(d)  : 「指示タスクより前に同エージェントが実行してよい他タスクは d 個まで」
                 という制約ありで解いた最適 makespan

        L が大きいほど、最適な段取りから外れた効率の悪い指示だったことになる。
        1回あたり CP-SAT を2回解くため 0.3〜4秒かかる。ゲームの進行や AI の
        再スケジューリングとは無関係に走らせたいので、必ず別スレッドで行う。
        結果は pending_entry['time_loss'] に入れ、リプレイにも残す。
        """
        ai = getattr(self, 'ai', None)
        if ai is None or not hasattr(ai, 'estimate_instruction_time_loss'):
            return
        env_state = getattr(self, '_latest_env_state', None)
        if env_state is None:
            return

        pending_entry['time_loss'] = {'status': 'calculating'}

        def work():
            started = time.time()
            try:
                result = ai.estimate_instruction_time_loss(env_state, pending_entry)
            except Exception as e:
                result = {'status': f'error: {e}', 'loss_seconds': None}
            result['elapsed_s'] = round(time.time() - started, 3)
            pending_entry['time_loss'] = result

            if result.get('loss_seconds') is not None:
                print(f"[TimeLoss] id={pending_entry['id']:.6f} task={display_text} "
                      f"d={result.get('skip_budget')} "
                      f"L={result['loss_seconds']:.1f}s "
                      f"(f={result['baseline_seconds']:.1f}s -> "
                      f"f'={result['constrained_seconds']:.1f}s) "
                      f"計算 {result['elapsed_s']:.2f}s")
            else:
                print(f"[TimeLoss] id={pending_entry['id']:.6f} task={display_text} "
                      f"計算できず: {result.get('status')}")

            try:
                self.replay.log('instruction_time_loss',
                                {'id': pending_entry['id'], 'task': display_text, **result})
            except Exception:
                pass

        threading.Thread(target=work, daemon=True).start()

    def _request_instruction(self, trigger='space', allow_text_fallback=True):
        """指示画面を出し、選ばれた指示を env / AI / リプレイへ登録する。

        Space 押下と、enable_cook の自動呼び出しの共通処理。
        """
        if self._instruction_panel_active:
            return

        self._q_env.put(('Pause', {}))
        try:
            candidates = self._get_unexecuted_task_candidates()
            if candidates:
                s = self._show_instruction_panel(candidates)
            elif allow_text_fallback:
                s = popup_text("Say to AI:")
            else:
                s = None

            if s is not None:
                # Record instruction and forward to env/AI
                inst_id = time.time()
                # get environment time (seconds) snapshot for correlation with CSP frames
                accepted_env_time = self._latest_env_state.time if getattr(self, '_latest_env_state', None) is not None else 0.0
                # target agent index for AI (default to configured ai_agent_idx or 0)
                target_idx = self.ai_agent_idx if self.ai_agent_idx is not None else 0
                # support structured return from popup: (display, payload)
                display_text = s[0] if isinstance(s, (list, tuple)) and len(s) >= 1 else str(s)
                pending_payload = s
                if isinstance(s, (list, tuple)) and len(s) >= 2:
                    pending_payload = s[1]
                # attach pending instruction to env for later correlation with events
                skip_budget_val = getattr(self.ai, 'skip_budget', None) if hasattr(self, 'ai') and self.ai is not None else None
                pending_entry = {'id': inst_id, 'task': pending_payload, 'target_idx': target_idx, 'accepted_env_time': accepted_env_time, 'execution_logged': False, 'deadline_constraint_applied': False, 'status': 'pending', 'skip_budget': skip_budget_val, 'remaining_skip_budget': skip_budget_val, 'tasks_before_target_log': [], 'trigger': trigger}
                try:
                    if not hasattr(self.env, '_pending_instructions'):
                        self.env._pending_instructions = []
                    self.env._pending_instructions.append(pending_entry)
                except Exception as e:
                    print(f"[GamePlay] Failed to attach pending instruction to env: {e}")

                try:
                    if hasattr(self, 'ai') and self.ai is not None:
                        if not hasattr(self.ai, '_pending_instructions'):
                            self.ai._pending_instructions = []
                        self.ai._pending_instructions.append(pending_entry)
                except Exception as e:
                    print(f"[GamePlay] Failed to attach pending instruction to agent: {e}")

                # replay log for instruction accepted
                try:
                    # log display text for human-readable logs, keep payload in pending entry
                    self.replay.log('instruction_accepted', {'id': inst_id, 'task': display_text, 'accepted_time_wall': inst_id, 'accepted_time_env': accepted_env_time, 'target_idx': target_idx, 'trigger': trigger})
                except Exception:
                    pass

                # Print to debug log (will be captured in debug_*.log)
                print(f"[Instruction] accepted id={inst_id:.6f} task={display_text} agent_idx={target_idx} env_time={accepted_env_time:.6f} wall_time={inst_id:.6f} trigger={trigger}")

                # 時間損失量 L(d) を計算する(別スレッド。進行は止めない)
                self._start_time_loss_estimation(pending_entry, display_text)

                # Signal AI to reschedule due to new instruction
                try:
                    if hasattr(self, 'ai') and self.ai is not None:
                        self.ai._mark_reschedule_needed('instruction_accepted')
                except Exception as e:
                    print(f"[GamePlay] Failed to notify AI of instruction: {e}")

                # send human-readable display to chat queues
                self._q_env.put(('ChatIn', {"chat": display_text, "mode": "text"}))
                self._q_ai.put(('Chat', dict(chat=display_text)))
        finally:
            self._q_env.put(('Continue', {}))

    def _poll_cook_instruction_trigger(self):
        """enable_cook: 調理タスクに今すぐ着手できる状態になった瞬間、指示画面を出す。

        AI は判断サイクルごとに「いま着手できる cook タスク」を
        ready_cook_actions へ書き出しているので、ここではそれを読むだけにする
        (毎フレーム世界を評価し直すと重く、置き場の割り当てにも副作用が出る)。

        発火は集合の差分、つまり「前回は着手できなかったタスクが着手可能になった」
        立ち上がりのときだけ。着手可能なまま留まっている間に何度も画面を出さない。
        """
        if self.instruction_request_timing != INSTRUCTION_TIMING_ENABLE_COOK:
            return
        if self._instruction_panel_active or self.ai is None:
            return

        ready = set(getattr(self.ai, 'ready_cook_actions', ()) or ())
        newly_ready = ready - self._seen_ready_cook_actions
        # 着手できなくなったタスクは記憶から外す。再び条件が揃えば改めて発火させる。
        self._seen_ready_cook_actions = ready
        if not newly_ready:
            return

        print(f"[Instruction] enable_cook trigger: {sorted(newly_ready)}")
        # 候補が無ければ何も出さない(テキスト入力へは落とさない)。
        # 条件が成立した瞬間に選ばせるモードなので、選択肢のない入力欄は意味がない。
        self._request_instruction(trigger='enable_cook', allow_text_fallback=False)

    def _run_env(self):
        seconds_per_step = 1 / self.fps
        idx_human = self.idx_human
        paused = 0
        chat_in, chat_out = "", ""
        last_t = time.time()
        action_dict = {agent.name: None for agent in self.sim_agents}

        self.on_render(paused=paused)
        info = self.env.get_ai_info()
        e = EnvState(world=info['world'],
                     agents=info['sim_agents'],
                     agent_idx=self.ai_agent_idx if self.ai_agent_idx is not None else 0,
                     order=info['order_scheduler'],
                     event_history=info['event_history'],
                     time=info['current_time'],
                     chg_grid=info['chg_grid'])
        self._latest_env_state = dcopy(e)
        self._q_ai.put_nowait(('Env', {"EnvState": e}))

        while True:
            loop_top = time.time()
            while not self._q_env.empty():
                event = self._q_env.get_nowait()
                event_type, args = event
                if event_type == 'Action':
                    if args['agent'] == "human" and idx_human is not None:
                        action_dict[self.sim_agents[idx_human].name] = args['action']
                    elif args['agent'] == "ai" and self.ai_agent_idx is not None:
                        action_dict[self.sim_agents[self.ai_agent_idx].name] = args['action']
                    elif args['agent'] == "ai_0":
                        action_dict[self.sim_agents[0].name] = args['action']
                        if self.debug_mode:
                            print(f"[ENVTRACE] recv_action wall={time.time():.4f} agent=ai_0 action={args['action']}")
                    elif args['agent'] == "ai_1":
                        if len(self.sim_agents) > 1:
                            action_dict[self.sim_agents[1].name] = args['action']
                elif event_type == 'Pause':
                    paused += 1
                elif event_type == 'Continue':
                    paused -= 1
                elif event_type == 'ChatIn':
                    # chat_in = f"User Input: [{args['mode']}]\n\n" + args['chat']
                    chat_in = ""  # User Inputの画面表示を無効化(指示後にゲーム画面へ重なるため)
                    chat_out = ""
                elif event_type == 'ChatOut':
                    # chat_out = "AI Output:\n\n" + args['chat']
                    chat_out = ""  # AI Outputの画面表示を無効化

            if not paused:
                ad = {k: v if v is not None else (
                    0, 0) for k, v in action_dict.items()}
                if self.debug_mode:
                    a0 = self.sim_agents[0]
                    print(f"[ENVTRACE] step_begin wall={time.time():.4f} pos_before={a0.location} "
                          f"applied_action={ad.get(a0.name)} hold_before={a0.get_holding()}")
                self.replay.log(
                    'env.step', {'action_dict': ad, 'passed_time': seconds_per_step})
                _, _, done, _ = self.env.step(ad, passed_time=seconds_per_step)
                if self.debug_mode:
                    a0 = self.sim_agents[0]
                    print(f"[ENVTRACE] step_end   wall={time.time():.4f} pos_after={a0.location} "
                          f"hold_after={a0.get_holding()}")
                if done:
                    self._success = True
                    self._q_control.put(('Quit', {}))
                    return

                info = self.env.get_ai_info()
                e = EnvState(world=info['world'],
                             agents=info['sim_agents'],
                             agent_idx=self.ai_agent_idx if self.ai_agent_idx is not None else 0,
                             order=info['order_scheduler'],
                             event_history=info['event_history'],
                             time=info['current_time'],
                             chg_grid=info['chg_grid'])
                self._latest_env_state = dcopy(e)

                # 毎ステップAIへ最新状態を送る。
                # 人間の操作だけで状態が変わった場合でも、CSPの再計画を即時に起こすため。
                # applied_actions: このステップで各エージェントに実際に適用したaction。
                # _run_ai 側が「自分が送ったコマンドが本当に反映されたか」を確認するために使う
                # (非同期キューのため、反映前に同じ状況を見て同じコマンドを二重に送ってしまい、
                #  移動が1マス行き過ぎたり、拾う/置くを繰り返してしまう不具合があったため)。
                self._q_ai.put(('Env', {"EnvState": dcopy(e), "applied_actions": dict(ad)}))
                action_dict = {agent.name: None for agent in self.sim_agents}

            # 描画は step の直後、sleep より前に行うこと。
            # sleep の後ろに置くと、step N の結果が画面に出るのは次の周回、
            # つまり約 1/fps 秒(既定で100ms)遅れになる。入力が反映されて
            # 見えるまでに、キュー待ちに加えてこの1周期が丸ごと乗るため、
            # 操作がはっきり重く感じられる。
            chat = chat_in + '\n\n' + chat_out

            debug_info = {}
            if self.debug_mode:
                if hasattr(self.ai, 'get_assigned_counters'):
                    debug_info['counters'] = self.ai.get_assigned_counters()
                if hasattr(self.ai, 'get_order_display_labels'):
                    debug_info['order_labels'] = self.ai.get_order_display_labels()
                if self.sc_2agent and hasattr(self.ai, 'task_agents'):
                    debug_info['tasks'] = {
                        "AI0": self.ai.task_agents[0].task_name if hasattr(self.ai.task_agents[0], 'task_name') and self.ai.task_agents[0].task_name else "Idle",
                        "AI1": self.ai.task_agents[1].task_name if hasattr(self.ai.task_agents[1], 'task_name') and self.ai.task_agents[1].task_name else "Idle"
                    }
                elif hasattr(self.ai, 'task_agent'):
                    debug_info['tasks'] = {
                        "AI": self.ai.task_agent.task_name if hasattr(self.ai.task_agent, 'task_name') and self.ai.task_agent.task_name else "Idle"
                    }

            if not paused:
                self.replay.log('on_render', {'paused': paused, 'chat': chat})
            # 指示パネル表示中はこのスレッドから描画しない。
            # on_render は screen.fill -> display.flip まで行うため、パネル側の
            # 描画と交互に画面全体を上書きし合って激しく点滅してしまう。
            if not self._instruction_panel_active:
                self.on_render(paused=paused, chat=chat, debug_info=debug_info)

            # 1周の実測。work は実際に処理へ使った時間、period は1周の間隔。
            # period が 1/fps より大きいのに work が小さいなら、CPU ではなく
            # 他スレッドに邪魔されて回れていないということ。
            now = time.time()
            self.loop_stats['work_s'] = now - loop_top
            if self.loop_stats['last_top'] is not None:
                self.loop_stats['period_s'] = loop_top - self.loop_stats['last_top']
            self.loop_stats['last_top'] = loop_top

            # AI の判断が 1 ステップ分の時間より長くかかっているなら、環境の 1 ステップを
            # その分だけ引き伸ばす。--debug では詳細トレースの出力だけで判断1回が
            # 数百msかかるため、これをしないと AI が数フレームに1回しか動けなくなる。
            # 引き伸ばすのは実時間だけで、env.step へ渡すゲーム内時間(passed_time)は
            # seconds_per_step のままなので、調理時間などのゲームロジックは変わらない。
            # 1ステップに必要なのは「判断時間」だけではない。AIは自分が送った行動が
            # 適用されたことを次の状態pushで確認してから次を送る(awaiting_confirm)ため、
            # 1行動あたり判断+往復ぶんの時間がかかる。判断時間と同じ周期にすると
            # AIは1ステップおきにしか動けない(実測 47.6%)。往復ぶんを見込んで倍にする。
            step_period = seconds_per_step
            if self.pace_env_to_ai:
                needed = self._ai_decide_seconds * AI_PACE_ROUNDTRIP_FACTOR
                if needed > step_period:
                    step_period = min(needed, seconds_per_step * self.max_pace_stretch)

            # last_t は sleep の「後」に更新すること。
            # 前に更新すると、次の周回で測る経過時間に自分の sleep 時間が含まれてしまい、
            # sleep_time が 0 になる周回と満額になる周回が交互に現れて、環境が設定 fps の
            # 約2倍(fps=10 に対し実測 18.7 step/秒)で回ってしまう。
            # AI は fps_ai(=10)/秒でしか行動を決められないため、環境だけが倍速で進むと
            # AI は全ステップの半分しか動けず、毎ステップ入力が反映される人間に対して
            # 半分の速度に見える(「AIの移動が遅い」の原因)。
            sleep_time = max(step_period - (time.time() - last_t), 0)
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_t = time.time()

    def _refresh_instruction_states(self, env):
        """実行開始/完了に応じて pending instruction の状態を更新する。"""
        try:
            pending_instr = getattr(env, '_pending_instructions', [])
            if not pending_instr:
                return
            for pending in pending_instr:
                if pending.get('status') in {'done', 'expired', 'canceled'}:
                    continue
                task_payload = pending.get('task')
                if isinstance(task_payload, (list, tuple)) and len(task_payload) >= 2:
                    payload = task_payload[1]
                else:
                    payload = task_payload
                fixed_task_id = None
                if isinstance(payload, dict):
                    fixed_task_id = payload.get('fixed_task_id')
                elif isinstance(payload, (list, tuple)) and len(payload) >= 1:
                    fixed_task_id = payload[0]
                else:
                    fixed_task_id = payload
                if fixed_task_id is None:
                    continue

                if hasattr(self.ai, 'get_active_task_ids'):
                    active_ids = self.ai.get_active_task_ids()
                elif hasattr(self.ai, '_get_active_task_ids'):
                    active_ids = self.ai._get_active_task_ids()
                else:
                    active_ids = set()

                if fixed_task_id in active_ids:
                    pending['execution_logged'] = True
                    pending['deadline_constraint_applied'] = True
        except Exception as e:
            print(f"[GamePlay] Failed to refresh instruction state: {e}")

    def _target_idx_for_agent_id(self, agent_id):
        if agent_id == "ai_0":
            return 0
        if agent_id == "ai_1":
            return 1
        return self.ai_agent_idx if self.ai_agent_idx is not None else 0

    def _dispatch_agent_action(self, agent_id, action, env, awaiting_confirm):
        """agent_id 宛ての action を _q_env に送る。ただし、直前にこのエージェントへ送った
        コマンドがまだ実際に環境へ適用されたと確認できていない場合は送らずスキップする。

        _run_ai と _run_env は非同期スレッド+キューで繋がっており、_run_ai は自分が送った
        コマンドの効果(座標の変化・所持品の変化など)が反映されるより前に、まだ古い状態を
        見て次の判断をしてしまうことがある。これにより:
        - 移動コマンドが2回連続で送られ、目的地を1マス通り過ぎてしまう
        - 「拾う」action が既に成功しているのに、もう一度同じ方向へのコマンドが送られて
          今度は逆に「置く」として作用し、拾っては置くを繰り返してしまう
        といった不具合が起きていた。(0,0) は状態を変えないため対象外。
        """
        target_idx = self._target_idx_for_agent_id(agent_id)
        if self.human_agent_idx is not None and target_idx == self.human_agent_idx:
            return
        if action != (0, 0) and awaiting_confirm.get(agent_id) is not None:
            # 直前に送ったコマンドの結果がまだ確認できていない -> 二重送信を防ぐため今回はスキップ
            return
        if action != (0, 0):
            awaiting_confirm[agent_id] = action
        if self.debug_mode and agent_id == "ai_0":
            print(f"[AITRACE] push_action wall={time.time():.4f} pos_seen={env.self_pos} action={action}")
        self._q_env.put(('Action', {"agent": agent_id, "action": action}))

    def _run_ai(self):
        time_per_step = 1 / self.fps_ai
        # 環境からの状態push(_q_ai.get)がブロックするため、AIの実行レートは
        # 何もしなくても環境のstep rateに一致する。これに加えて独自のsleepで
        # 待つと、AI側とenv側の周期がわずかにずれて環境のstepを取りこぼし、
        # AIが動けないフレームが生まれる(fps_ai=fps=10 の設定で、実測では
        # 全stepの86%までしか動けなかった)。
        # fps_ai が env の fps より明示的に低く設定されている場合、つまり
        # 「AIをわざと遅くしたい」ときだけ間引く。
        throttle_ai = self.fps_ai < self.fps
        time_last = time.time()
        human_act = True
        env = None
        env_update = False
        chat = ''
        # agent_id -> 直前に送信し、まだ実際に環境へ適用されたと確認できていない action。
        # None(未送信)になって初めて次のコマンドを送る。
        awaiting_confirm = {}
        while True:
            # AIをわざと遅くしたい設定のときだけ間引く。
            # この sleep は「判断 → 送信」の間ではなくループ先頭に置くこと。
            # 間に置くと、決めた行動が最大 1/fps_ai 秒ぶん遅れて環境に届き、
            # その間に環境が数ステップ進んでしまう。さらに awaiting_confirm は
            # 「送った行動が適用されたと確認できるまで次を送らない」ため、
            # 遅延ぶんがそのまま次の行動までの待ち時間に加算され、
            # AI が数フレームに1回しか動けなくなる。
            # 先頭で待ってから最新状態を取り込み、判断した行動は即座に送る。
            if throttle_ai:
                sleep_time = max(time_per_step - (time.time() - time_last), 0)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                time_last = time.time()

            event = self._q_ai.get()
            while True:
                event_type, args = event
                if event_type == 'Env':
                    env = args['EnvState']
                    env_update = True
                    applied_actions = args.get('applied_actions') or {}
                    for agent_id in list(awaiting_confirm.keys()):
                        target_idx = self._target_idx_for_agent_id(agent_id)
                        agents = getattr(env, 'agents', None)
                        if not agents or target_idx >= len(agents):
                            continue
                        agent_name = agents[target_idx].name
                        if applied_actions.get(agent_name) == awaiting_confirm[agent_id]:
                            # このステップで実際に適用されたことを確認できた -> 次のコマンドを送ってよい
                            awaiting_confirm.pop(agent_id, None)
                elif event_type == 'Chat':
                    chat = args['chat']
                elif event_type == "Action":
                    human_act = True
                elif event_type == "Quit":
                    return
                if not self._q_ai.empty():
                    event = self._q_ai.get()
                else:
                    break

            if chat != '':
                self.ai.high_level_infer(env, chat)
                chat = ''

            if env_update:
                self._refresh_instruction_states(env)
                if self.debug_mode:
                    hold_seen = getattr(env, 'hold', None)
                    hold_seen_name = getattr(hold_seen, 'full_name', None) if hold_seen is not None else None
                    print(f"[AITRACE] decide_begin wall={time.time():.4f} pos_seen={env.self_pos} hold_seen={hold_seen_name}")
                # 判断時間の計測は --debug のときだけ行う(pace_env_to_ai=debug_mode)。
                # 通常プレイでは計測も進行の引き伸ばしも一切行わない。
                decide_started = time.time() if self.pace_env_to_ai else None
                move, chat_ret = self.ai(env)
                if decide_started is not None:
                    # 環境側が「AIより速く進まない」ようにするための実測値。
                    # 再スケジュール時だけ跳ねる(CSP探索)ので、指数移動平均で均す。
                    decide_elapsed = time.time() - decide_started
                    if self._ai_decide_seconds <= 0:
                        self._ai_decide_seconds = decide_elapsed
                    else:
                        self._ai_decide_seconds = self._ai_decide_seconds * 0.8 + decide_elapsed * 0.2
                if self.debug_mode:
                    print(f"[AITRACE] decide_end   wall={time.time():.4f} pos_seen={env.self_pos} move={move}")

                if chat_ret:
                    self._q_env.put(('ChatOut', {"chat": chat_ret}))

                if isinstance(move, dict):
                    for agent_id, m in move.items():
                        self._dispatch_agent_action(agent_id, m, env, awaiting_confirm)
                else:
                    target_idx = self.ai_agent_idx if self.ai_agent_idx is not None else 0
                    self._dispatch_agent_action(f"ai_{target_idx}", move, env, awaiting_confirm)
                human_act = False
                env_update = False

    # def _run_listen(self):
    #     ena_listen = False
    #     self._q_env.put(('Setting', {"ena_listen": ena_listen}))

    #     dir_path = os.path.dirname(os.path.realpath(__file__))
    #     model_path = dir_path + r"/vosk"
    #     model = Model(model_path)
    #     recognizer = KaldiRecognizer(model, 16000)

    #     mic = pyaudio.PyAudio()
    #     stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
    #     stream.start_stream()

    #     while True:
    #         while not self._q_listen.empty():
    #             event = self._q_listen.get_nowait()
    #             event_type, args = event
    #             if event_type == 'Quit':
    #                 return
    #             elif event_type == 'ListenSwitch':
    #                 ena_listen = not ena_listen
    #                 self._q_env.put(('Setting', {"ena_listen": ena_listen}))

    #         if not ena_listen:
    #             time.sleep(0.1)
    #             continue

    #         data = stream.read(10000)

    #         if recognizer.AcceptWaveform(data):
    #             text = recognizer.Result()
    #             s = text[14:-3]
    #             if s is not None and len(s) >= 4:
    #                 self._q_env.put(('ChatIn', {"chat": s, "mode": "speech"}))
    #                 self._q_ai.put(('Chat', dict(chat=s)))

    def _run_human(self):
        while True:
            for event in pygame.event.get():
                self.on_event(event)
            # 指示パネルの描画は pygame の main スレッドから行う必要があるため、
            # 自動呼び出しの監視もこのループに置く。
            self._poll_cook_instruction_trigger()
            if not self._q_control.empty():
                event, args = self._q_control.get_nowait()
                if event == 'Quit':
                    self._q_ai.put(('Quit', {}))
                    return
            # 少しだけ待つ。待たずに回すとこのスレッドが GIL を握りっぱなしになり、
            # 環境スレッド(10Hz)・AIスレッド・(Web版では)配信スレッドがその分だけ
            # 割り込めなくなる。コア数が少ない環境ほど影響が大きく、環境の周期が
            # 伸びて画面が重くなる。ゲームは 10Hz なので 500Hz も見れば十分で、
            # 入力の取りこぼしや遅れは生じない。
            time.sleep(HUMAN_POLL_INTERVAL)

    def on_execute(self):
        if self.on_init() == False:
            exit()

        thread_env = threading.Thread(target=self._run_env, daemon=True)
        thread_ai = threading.Thread(target=self._run_ai, daemon=True)
        # thread_listen = threading.Thread(target=self._run_listen, daemon=True)
        thread_env.start()
        thread_ai.start()
        # thread_listen.start()

        try:
            self._run_human()
        finally:
            self._finalize_execution()

        return self._success

    def _finalize_execution(self):
        if self._finalized:
            return
        self._finalized = True

        # clean up
        self.on_cleanup()

        # save history
        if hasattr(self.ai, "_lock"):
            self.ai._lock.acquire()
        try:
            if hasattr(self.ai, "_int_hist"):
                self.replay['int_hist'] = self.ai._int_hist
            if hasattr(self.ai, "_llm_hist"):
                self.replay['llm_hist'] = self.ai._llm_hist
            if hasattr(self.ai, "_mov_hist"):
                self.replay['mov_hist'] = self.ai._mov_hist

            # log recipy infos
            self.replay['order_result'] = dict(
                success=self.env.order_scheduler.successful_orders,
                fail=self.env.order_scheduler.failed_orders,
                reward=self.env.order_scheduler.reward
            )
        finally:
            if hasattr(self.ai, "_lock"):
                self.ai._lock.release()

            # ログファイルを閉じる
            if self._log_file:
                sys.stdout = self._original_stdout
                self._log_file.close()
