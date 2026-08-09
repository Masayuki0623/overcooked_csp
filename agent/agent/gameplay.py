# modules for game
from gym_cooking.misc.game.game import Game
from gym_cooking.misc.game.utils import *
from gym_cooking.utils.gui import popup_text, popup_task_choice
from gym_cooking.utils.replay import Replay
from agent.executor.low import EnvState
from agent.mind.agent import get_agent, AgentSetting

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


class GamePlay(Game):
    def __init__(self, env, replay: Replay, agent_set: AgentSetting, debug_mode: bool = False, human_agent_idx: int | None = 1, ai_agent_idx: int | None = 0, sc_2agent: bool = False):
        Game.__init__(self, env, play=True)
        self.replay = replay
        self.agent_set = agent_set
        self.debug_mode = debug_mode
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
                self._q_env.put(('Pause', {}))

                candidates = self._get_unexecuted_task_candidates()
                if candidates:
                    s = popup_task_choice("AIへの指示タスクを選択してください", candidates)
                else:
                    s = popup_text("Say to AI:")

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
                    pending_entry = {'id': inst_id, 'task': pending_payload, 'target_idx': target_idx, 'accepted_env_time': accepted_env_time, 'execution_logged': False, 'deadline_constraint_applied': False, 'status': 'pending', 'skip_budget': skip_budget_val, 'remaining_skip_budget': skip_budget_val, 'tasks_before_target_log': []}
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
                        self.replay.log('instruction_accepted', {'id': inst_id, 'task': display_text, 'accepted_time_wall': inst_id, 'accepted_time_env': accepted_env_time, 'target_idx': target_idx})
                    except Exception:
                        pass

                    # Print to debug log (will be captured in debug_*.log)
                    print(f"[Instruction] accepted id={inst_id:.6f} task={display_text} agent_idx={target_idx} env_time={accepted_env_time:.6f} wall_time={inst_id:.6f}")

                    # Signal AI to reschedule due to new instruction
                    try:
                        if hasattr(self, 'ai') and self.ai is not None:
                            self.ai._mark_reschedule_needed('instruction_accepted')
                    except Exception as e:
                        print(f"[GamePlay] Failed to notify AI of instruction: {e}")

                    # send human-readable display to chat queues
                    self._q_env.put(('ChatIn', {"chat": display_text, "mode": "text"}))
                    self._q_ai.put(('Chat', dict(chat=display_text)))

                self._q_env.put(('Continue', {}))

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
                    elif args['agent'] == "ai_1":
                        if len(self.sim_agents) > 1:
                            action_dict[self.sim_agents[1].name] = args['action']
                elif event_type == 'Pause':
                    paused += 1
                elif event_type == 'Continue':
                    paused -= 1
                elif event_type == 'ChatIn':
                    chat_in = f"User Input: [{args['mode']}]\n\n" + \
                        args['chat']
                    chat_out = ""
                elif event_type == 'ChatOut':
                    # chat_out = "AI Output:\n\n" + args['chat']
                    chat_out = ""  # AI Outputの画面表示を無効化

            if not paused:
                ad = {k: v if v is not None else (
                    0, 0) for k, v in action_dict.items()}
                self.replay.log(
                    'env.step', {'action_dict': ad, 'passed_time': seconds_per_step})
                _, _, done, _ = self.env.step(ad, passed_time=seconds_per_step)
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

                if self.ai_agent_idx is None:
                    if any(action_dict[agent.name] is not None for agent in self.sim_agents):
                        self._q_ai.put(('Env', {"EnvState": dcopy(e)}))
                elif action_dict[self.sim_agents[self.ai_agent_idx].name] is not None:
                    self._q_ai.put(('Env', {"EnvState": dcopy(e)}))
                action_dict = {agent.name: None for agent in self.sim_agents}

            sleep_time = max(seconds_per_step - (time.time() - last_t), 0)
            last_t = time.time()
            time.sleep(sleep_time)

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
            self.on_render(paused=paused, chat=chat, debug_info=debug_info)

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

    def _run_ai(self):
        time_per_step = 1 / self.fps_ai
        time_last = time.time()
        human_act = True
        env = None
        env_update = False
        chat = ''
        while True:
            event = self._q_ai.get()
            while True:
                event_type, args = event
                if event_type == 'Env':
                    env = args['EnvState']
                    env_update = True
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
                move, chat_ret = self.ai(env)

                # sleep
                sleep_time = max(time_per_step - (time.time() - time_last), 0)
                time.sleep(sleep_time)
                time_last = time.time()

                if chat_ret:
                    self._q_env.put(('ChatOut', {"chat": chat_ret}))
                    
                if isinstance(move, dict):
                    for agent_id, m in move.items():
                        if agent_id == "ai_0":
                            target_idx = 0
                        elif agent_id == "ai_1":
                            target_idx = 1
                        else:
                            target_idx = self.ai_agent_idx if self.ai_agent_idx is not None else 0
                        if self.human_agent_idx is not None and target_idx == self.human_agent_idx:
                            continue
                        self._q_env.put(('Action', {"agent": agent_id, "action": m}))
                else:
                    target_idx = self.ai_agent_idx if self.ai_agent_idx is not None else 0
                    if self.human_agent_idx is None or target_idx != self.human_agent_idx:
                        self._q_env.put(('Action', {"agent": f"ai_{target_idx}", "action": move}))
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
            if not self._q_control.empty():
                event, args = self._q_control.get_nowait()
                if event == 'Quit':
                    self._q_ai.put(('Quit', {}))
                    return

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
