"""「使う」を離したときに、1回ぶん余計に手が出ないことの検証。

想定シナリオ(報告された現象):
    まな板の前で「使う」を押しっぱなしにすると、置く → 切る → 取る まで
    進む。ところが切り終えた直後に指を離すと、取ったばかりの材料をもう
    一度台に置いてしまう。

    押した分は待ち行列に溜まり、1手ずつ消化される。離したことだけ別に
    扱うと、まだ処理していない「使う」が残っているのに押しっぱなしの印が
    先に消え、残りが新しい押し始めとして通ってしまう。

    ここでは、実際の入力の流れをそのまま組み立てて確かめる。
      1. 入力を受ける側 (GamePlay.on_event) へ押した/離したを流す
      2. 環境の輪が待ち行列へ移す
      3. 1手ずつ取り出して env.step する
    2 は _run_env の輪の頭と同じことをしている(その部分だけは写し)。

実行方法:
    python tests/repro_interact_release_timing.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

import collections
import queue
import threading

import pygame

from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from gym_cooking.play_test import MAP_SETTINGS
from gym_cooking.utils import config as game_config
from gym_cooking.utils.core import Object, Onion
from gym_cooking.utils.interact import INTERACT
from agent.agent.gameplay import GamePlay, HUMAN_INPUT_BACKLOG

results = []
DOWN = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE)
UP = pygame.event.Event(pygame.KEYUP, key=pygame.K_SPACE)


def check(label, ok, detail=""):
    results.append(ok)
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}{(' -> ' + detail) if detail else ''}")


class Harness:
    """入力を受ける側と環境の輪を、本物のまま順に回す。"""

    def __init__(self):
        self.env = OvercookedEnvironment(MapSetting(**MAP_SETTINGS['tutorial_salad']))
        self.env.reset()
        self.me = self.env.sim_agents[0]
        self.board = tuple(self.env.world.objects['Cutboard'][0].location)
        # まな板の隣に立って、まな板を向く
        self.me.location = (self.board[0] - 1, self.board[1])
        self.me.facing = (1, 0)
        onion = Object(location=self.me.location, contents=[Onion()])
        self.env.world.insert(onion)
        self.me.acquire(onion)

        g = GamePlay.__new__(GamePlay)
        g.env = self.env
        g.sim_agents = self.env.sim_agents
        g.idx_human = 0
        g.interact_held = False
        g.interact_used = False
        g.human_inputs_done = 0
        g._human_backlog = collections.deque(maxlen=HUMAN_INPUT_BACKLOG)
        g._backlog_lock = threading.Lock()
        g._q_env = queue.Queue()
        g._q_ai = queue.Queue()
        self.g = g

    def press(self, n=1):
        for _ in range(n):
            self.g.on_event(DOWN)

    def release(self):
        self.g.on_event(UP)

    def _drain(self):
        """_run_env の輪の頭と同じ。届いた入力を待ち行列へ移す。"""
        while not self.g._q_env.empty():
            kind, args = self.g._q_env.get_nowait()
            if kind == 'Action' and args.get('agent') == 'human':
                with self.g._backlog_lock:
                    self.g._human_backlog.append(args['action'])

    def tick(self):
        self._drain()
        acts = {a.name: (0, 0) for a in self.env.sim_agents}
        nxt = self.g._take_human_action()
        if nxt is not None:
            acts[self.me.name] = nxt
        facing_before = tuple(self.me.facing)
        applied, held_before = self.g._gate_interact(self.me, acts)
        self.env.step({k: v or (0, 0) for k, v in acts.items()}, passed_time=0.2)
        if facing_before != tuple(self.me.facing):
            self.g.interact_used = False
        if applied:
            after = getattr(self.me.holding, 'full_name', None)
            if not self.g._hold_still_usable(self.me, held_before, after):
                self.g.interact_used = True

    def hold(self, ticks, burst=1):
        """押しっぱなし。1手ごとに burst 回ぶん届く(回線のゆらぎを模す)。"""
        for _ in range(ticks):
            self.press(burst)
            self.tick()

    def state(self):
        on_board = self.env.world.get_object_at(self.board, None, find_held_objects=False) \
            if self.env.world.is_occupied(self.board) else None
        return (getattr(self.me.holding, 'full_name', None),
                getattr(on_board, 'full_name', None))


def run(label, release_after, burst, extra_ticks=6):
    """release_after 手ぶん押しっぱなしにしてから離す。

    見たいのは「一度手に取った材料を、離したあとにもう一度置いていないか」。
    離した時点でまだ届いていない入力の数は回線のゆらぎで変わるので、
    burst で1手あたりに届く回数を変えて試す。
    """
    h = Harness()
    seen = []
    for _ in range(release_after):
        h.press(burst)
        h.tick()
        seen.append(h.state())
    before = h.state()
    h.release()
    for _ in range(extra_ticks):
        h.tick()
        seen.append(h.state())
    after = h.state()
    picked = any(hand == 'ChoppedOnion' for hand, _ in seen)
    print(f'  {label}')
    print(f'     離す直前 手={before[0]} 台={before[1]}')
    print(f'     離した後 手={after[0]} 台={after[1]} '
          f'(途中で手に取った={"はい" if picked else "いいえ"})')
    return picked, after


def main():
    steps = game_config.chopping_steps()
    print(f'[SETUP] まな板で刻むのに要る「使う」の回数 = {steps}')
    print('[SETUP] 生の玉ねぎを持って、まな板の前に立っている')
    print()

    # 置く(1) + 刻む(steps) で、切り終わったところ。まだ取っていない。
    n = 1 + steps
    for burst in (1, 2, 3):
        picked, after = run(f'切り終えた直後に離す(1手あたり{burst}回届く)', n, burst)
        check(f'取った材料を置き直さない(届き方{burst})',
              not (picked and after[1] is not None),
              f'手={after[0]} 台={after[1]}')
        print()

    for burst in (1, 2, 3):
        picked, after = run(f'取ったあとに離す(1手あたり{burst}回届く)', n + 1, burst)
        check(f'取ったあとに離しても置き直さない(届き方{burst})',
              picked and after[0] == 'ChoppedOnion' and after[1] is None,
              f'手={after[0]} 台={after[1]}')
        print()

    # 軽く1回だけ叩く(押してすぐ離す)
    h = Harness()
    h.press(1)
    h.release()
    h.tick()
    hand, board = h.state()
    check('軽く1回叩くと、ちゃんと台に置ける',
          hand is None and board is not None, f'手={hand} 台={board}')

    print()
    print(f"[{'SUCCESS' if all(results) else 'FAIL'}] "
          f"{results.count(True)}/{len(results)} 件が期待どおり")
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
