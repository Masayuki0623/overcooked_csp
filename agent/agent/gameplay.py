# modules for game
from gym_cooking.misc.game.game import Game
from gym_cooking.misc.game.utils import *
from gym_cooking.utils.gui import popup_text
from gym_cooking.utils.replay import Replay
from agent.executor.low import EnvState
from agent.mind.agent import get_agent, AgentSetting
from agent.myagent.HumanPredictor import HumanPredictor

# helpers
import pygame
import threading
import queue
import time

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


class GamePlay(Game):
    def __init__(self, env, replay: Replay, agent_set: AgentSetting, debug_mode: bool = False, num_ai=1, num_human=1):
        Game.__init__(self, env, play=True)
        self.replay = replay
        self.agent_set = agent_set
        self.debug_mode = debug_mode
        self.num_ai = num_ai
        self.num_human = num_human
        self.is_ai_only = (num_human == 0)

        # fps of human and ai
        self.fps = 10
        self.fps_ai = agent_set.speed

        self.ai_indices = list(range(num_ai))
        self.human_indices = list(range(num_ai, num_ai + num_human))

        self.ai = get_agent(self.agent_set, self.replay)
        self.predictor = None
        if num_human > 0:
            self.predictor = HumanPredictor(env)

        # concurrent control variables
        self._q_control = queue.Queue()  # receive
        self._q_env = queue.Queue()
        self._q_ai = queue.Queue()
        self._success = False

    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._q_control.put(('Quit', {}))

        elif event.type == pygame.KEYDOWN:
            # Human 1 (Arrows)
            if self.num_human >= 1 and event.key in KeyToTuple.keys():
                action = KeyToTuple[event.key]
                self._q_env.put(
                    ('Action', {"agent": "human", "idx": 0, "action": action}))
            
            # Human 2 (WASD)
            if self.num_human >= 2 and event.key in KeyToTuple2.keys():
                action = KeyToTuple2[event.key]
                self._q_env.put(
                    ('Action', {"agent": "human", "idx": 1, "action": action}))

            if pygame.key.name(event.key) == "space":
                self._q_env.put(('Pause', {}))

                s = popup_text("Say to AI:")

                if s is not None:
                    self._q_env.put(('ChatIn', {"chat": s, "mode": "text"}))
                    self._q_ai.put(('Chat', dict(chat=s)))

                self._q_env.put(('Continue', {}))

    def _run_env(self):
        seconds_per_step = 1 / self.fps
        paused = 0
        chat_in, chat_out = "", ""
        last_t = time.time()
        action_dict = {agent.name: None for agent in self.sim_agents}

        self.on_render(paused=paused)
        info = self.env.get_ai_info()
        
        e = EnvState(world=info['world'],
                     agents=info['sim_agents'],
                     agent_idx=self.ai_indices, # Pass list of AI indices
                     order=info['order_scheduler'],
                     event_history=info['event_history'],
                     time=info['current_time'],
                     chg_grid=info['chg_grid'])
        self._q_ai.put_nowait(('Env', {"EnvState": e}))

        while True:
            while not self._q_env.empty():
                event = self._q_env.get_nowait()
                event_type, args = event
                if event_type == 'Action':
                    if args['agent'] == "human":
                        # Human Action
                        h_idx = args.get('idx', 0)
                        if h_idx < len(self.human_indices):
                            real_idx = self.human_indices[h_idx]
                            if real_idx < len(self.sim_agents):
                                action_dict[self.sim_agents[real_idx].name] = args['action']
                    elif args['agent'] == "ai":
                        # AI Action (Expects dict or list)
                        ai_actions = args['action'] # {agent_name: action} OR list or single action
                        if isinstance(ai_actions, dict):
                            for name, act in ai_actions.items():
                                action_dict[name] = act
                        elif isinstance(ai_actions, (list, tuple)) and len(ai_actions) == self.num_ai:
                            for i, act in enumerate(ai_actions):
                                real_idx = self.ai_indices[i]
                                action_dict[self.sim_agents[real_idx].name] = act
                        else:
                            # Fallback for single AI (legacy)
                            if self.num_ai == 1:
                                action_dict[self.sim_agents[self.ai_indices[0]].name] = ai_actions
                elif event_type == 'Pause':
                    paused += 1
                elif event_type == 'Continue':
                    paused -= 1
                elif event_type == 'ChatIn':
                    chat_in = f"User Input: [{args['mode']}]\n\n" + \
                        args['chat']
                    chat_out = ""
                elif event_type == 'ChatOut':
                    chat_out = "AI Output:\n\n" + args['chat']

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
                             agent_idx=self.ai_indices,
                             order=info['order_scheduler'],
                             event_history=info['event_history'],
                             time=info['current_time'],
                             chg_grid=info['chg_grid'])
                
                # Human Prediction (Only for 1st human for now?)
                if self.human_indices and self.predictor:
                     # Predict based on first human?
                    task_name, cost, all_costs = self.predictor.predict(e, self.human_indices[0])
                    if all_costs:
                        all_costs.sort(key=lambda x: x[1])

                # Check if ANY AI action was processed? 
                # Actually, in _run_ai, the AI sleeps and thinks. 
                # We should send EnvState if we want continuous updates or just when ready.
                # Use simplified logic: Always send EnvState? 
                # The original code sent it "if action_dict[self.sim_agents[0].name] is not None"
                # implying lock-step or something.
                # Let's just send it.
                self._q_ai.put(('Env', {"EnvState": dcopy(e)}))

                action_dict = {agent.name: None for agent in self.sim_agents}

            sleep_time = max(seconds_per_step - (time.time() - last_t), 0)
            last_t = time.time()
            time.sleep(sleep_time)

            chat = chat_in + '\n\n' + chat_out

            debug_info = None
            if self.debug_mode and hasattr(self.ai, 'get_assigned_counters'):
                debug_info = self.ai.get_assigned_counters()

            if not paused:
                self.replay.log('on_render', {'paused': paused, 'chat': chat})
            self.on_render(paused=paused, chat=chat, debug_info=debug_info)

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

            if chat != '' and not self.is_ai_only:
                self.ai.high_level_infer(env, chat)
                chat = ''

            if env_update:
                move, chat_ret = self.ai(env)

                sleep_time = max(time_per_step - (time.time() - time_last), 0)
                time.sleep(sleep_time)
                time_last = time.time()

                if chat_ret:
                    self._q_env.put(('ChatOut', {"chat": chat_ret}))
                self._q_env.put(('Action', {"agent": "ai", "action": move}))
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
        thread_env.start()
        thread_ai.start()

        if not self.is_ai_only:
            self._run_human()
        else:
            # In AI-only mode, the main thread waits for the environment thread to finish.
            thread_env.join()
            # Also wait for AI thread to prevent premature exit
            self._q_ai.put(('Quit', {}))
            thread_ai.join()

        # clean up
        self.on_cleanup()

        # save history
        if hasattr(self.ai, "_lock"):
            self.ai._lock.acquire()
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
        if hasattr(self.ai, "_lock"):
            self.ai._lock.release()

        return self._success
