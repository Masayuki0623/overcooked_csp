"""指示を選んでいる間に動かしても、選び終えたあとに動かないことの検証。

報告された現象:
    「指示を受けている間になぜか少しだけ操作ができて、指示を入力を終えると
     指示入力中に移動した分だけ戻る」

仕組み:
    指示画面を出すとき、ゲームは Pause で止めてある。ところが人の操作は
    そのまま待ち行列へ溜まっていた。溜まったぶんは、選び終えてゲームが
    動き出した瞬間にまとめて効く。さらに Web 版は押した瞬間に端末側で
    先読みして描くため、止まっている間もキャラだけが進んで見え、実際の
    位置が届いた瞬間に引き戻される。

ここで確かめること:
    1. 止まっている間の操作は待ち行列に入らない
    2. 止まっている間の操作は「処理した数」に数える
       (端末側はこの数で先読みを合わせ直しているので、数え落とすと
        ズレたままになる)
    3. 「離した」の合図は止まっている間でも効く
       (落とすと押しっぱなしの印が残り、指示のあと手が出せなくなる)
    4. 動いている間はこれまでどおり溜まる

実行方法:
    python tests/repro_move_during_instruction.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

import collections
import threading

from agent.agent.gameplay import (GamePlay, HUMAN_INPUT_BACKLOG,
                                  HUMAN_INTERACT_RELEASE)

results = []


def check(label, ok, detail=""):
    results.append(ok)
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}{(' -> ' + detail) if detail else ''}")


def make():
    g = GamePlay.__new__(GamePlay)
    g._human_backlog = collections.deque(maxlen=HUMAN_INPUT_BACKLOG)
    g._backlog_lock = threading.Lock()
    g.human_inputs_done = 0
    g.interact_held = False
    g.interact_used = False
    return g


RIGHT = (1, 0)

# 1 + 2: 指示中(paused)の移動は溜まらないが、数だけは進む
g = make()
for _ in range(5):
    g._queue_human_input(RIGHT, paused=1)
check('指示中に5回押しても、待ち行列は空のまま',
      len(g._human_backlog) == 0, f'溜まった数={len(g._human_backlog)}')
check('指示中のぶんも「処理した数」に数える',
      g.human_inputs_done == 5, f'done={g.human_inputs_done}')

# 3: 「離した」の合図は指示中でも効く
g = make()
g.interact_held = True
g.interact_used = True
g._queue_human_input(HUMAN_INTERACT_RELEASE, paused=1)
check('指示中でも「離した」は効く',
      g.interact_held is False and g.interact_used is False,
      f'held={g.interact_held} used={g.interact_used}')
check('「離した」は操作として数えない',
      g.human_inputs_done == 0, f'done={g.human_inputs_done}')

# 4: 動いている間はこれまでどおり
g = make()
for _ in range(2):
    g._queue_human_input(RIGHT, paused=0)
check('動いている間は待ち行列に溜まる',
      len(g._human_backlog) == 2, f'溜まった数={len(g._human_backlog)}')
check('溜まっただけでは、まだ処理した数は増えない',
      g.human_inputs_done == 0, f'done={g.human_inputs_done}')

# 4': 溜まりすぎたぶんは、これまでどおり捨てて数える
g = make()
for _ in range(HUMAN_INPUT_BACKLOG + 2):
    g._queue_human_input(RIGHT, paused=0)
check('溜まりすぎたぶんは捨てて数える',
      len(g._human_backlog) == HUMAN_INPUT_BACKLOG and g.human_inputs_done == 2,
      f'溜まった数={len(g._human_backlog)} done={g.human_inputs_done}')

ok = sum(1 for r in results if r)
print()
if ok == len(results):
    print(f'[SUCCESS] {ok}/{len(results)} 件が期待どおり')
else:
    print(f'[FAILURE] {ok}/{len(results)} 件のみ期待どおり')
    sys.exit(1)
