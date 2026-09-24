"""「使う」を押しっぱなしにしたときの続け方の検証。

    * まな板 : 置く → 切る → 取る まで一息でできる。取ったら終わり
               (そのまま続くと、取った材料をその場に置いてしまう)
    * ミキサー: 入れる → 混ぜる まで一息でできる(まな板と同じ扱い)
    * その他 : 置いた/取った時点で、その長押しは終わり
    * 離したとき : 待ち行列に残っている「使う」を捨てる
                   (入力は1手ずつしか消化されないので、離した知らせだけが
                    先に届き、残りが新しい押し始めとして処理されていた)

実行方法:
    python tests/repro_interact_hold_rules.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

import collections
import threading

import pygame

from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from gym_cooking.play_test import MAP_SETTINGS
from gym_cooking.utils.interact import INTERACT
from agent.agent.gameplay import GamePlay, HUMAN_INPUT_BACKLOG

results = []


def check(label, ok, detail=""):
    results.append(ok)
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}{(' -> ' + detail) if detail else ''}")


class FakeAgent:
    def __init__(self, location, facing):
        self.location = location
        self.facing = facing


def facing_from(env, target):
    """その台の隣に立って、台の方を向いた形を作る。"""
    floors = {tuple(f.location) for f in env.world.objects.get('Floor', [])}
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        stand = (target[0] - dx, target[1] - dy)
        if stand in floors:
            return FakeAgent(stand, (dx, dy))
    raise AssertionError(f'{target} の隣に立てません')


def main():
    env = OvercookedEnvironment(MapSetting(**MAP_SETTINGS['tutorial_juice']))
    env.reset()
    game = GamePlay.__new__(GamePlay)      # 判定だけ使う(盤面は作らない)
    game.env = env

    def pos_of(name):
        return tuple(env.world.objects[name][0].location)

    board = pos_of('Cutboard')
    blender = pos_of('Blender')
    # 隣に立てる、ただの台を1つ選ぶ(角の台は誰も触れない)
    floors = {tuple(f.location) for f in env.world.objects.get('Floor', [])}
    counter = next(tuple(c.location) for c in env.world.objects['Counter']
                   if any((c.location[0] + dx, c.location[1] + dy) in floors
                          for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))))
    print(f'[SETUP] まな板 {board} / ミキサー {blender} / ただの台 {counter}')

    at_board = facing_from(env, board)
    at_blender = facing_from(env, blender)
    at_counter = facing_from(env, counter)

    check('まな板に置いたら、そのまま切り続けられる',
          game._hold_still_usable(at_board, 'FreshApple', None))
    check('切っている間は続く(持ち物が変わらない)',
          game._hold_still_usable(at_board, None, None))
    check('切り終えて取ったら、そこで終わり',
          not game._hold_still_usable(at_board, None, 'ChoppedApple'),
          '続くと、取った材料をその場に置いてしまう')

    check('ミキサーに入れたら、そのまま混ぜ続けられる',
          game._hold_still_usable(at_blender, 'ChoppedApple-ChoppedOrange', None))
    check('コップを持ったまま入れた場合も続けられる',
          game._hold_still_usable(at_blender, 'ChoppedApple-ChoppedOrange-Cup', 'Cup'))
    check('混ぜている間は続く',
          game._hold_still_usable(at_blender, None, None))

    check('ただの台に置いたら、そこで終わり',
          not game._hold_still_usable(at_counter, 'ChoppedApple', None))
    check('ただの台から取ったら、そこで終わり',
          not game._hold_still_usable(at_counter, None, 'ChoppedApple'))

    # ---- 離したときに、余っている「使う」を捨てるか ----
    def release(used, queued):
        g = GamePlay.__new__(GamePlay)
        g._human_backlog = collections.deque(queued, maxlen=HUMAN_INPUT_BACKLOG)
        g._backlog_lock = threading.Lock()
        g.interact_used = used
        g.interact_held = True
        g.human_inputs_done = 0
        g._q_control = None
        g.on_event(pygame.event.Event(pygame.KEYUP, key=pygame.K_SPACE))
        return list(g._human_backlog), g.interact_held, g.human_inputs_done

    left, held, done = release(True, [INTERACT, INTERACT])
    check('もう手を出したあとに離したら、残りの「使う」は捨てる',
          left == [] and not held, f'残り={left} 押しっぱなし={held} 数えた={done}')

    left, _, _ = release(False, [INTERACT])
    check('まだ手を出していないなら残す(軽くたたいた分)',
          left == [INTERACT], f'残り={left}')

    left, _, _ = release(True, [(0, -1), INTERACT])
    check('移動の入力は捨てない', left == [(0, -1)], f'残り={left}')

    print()
    print(f"[{'SUCCESS' if all(results) else 'FAIL'}] "
          f"{results.count(True)}/{len(results)} 件が期待どおり")
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
