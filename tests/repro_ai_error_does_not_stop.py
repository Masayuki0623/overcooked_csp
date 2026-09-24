"""AI の判断で例外が出ても、手を出し続けることの検証。

想定シナリオ(バグ報告 20260924_174355「トマトを切っている途中で動かなく
なった」の原因):
    AI を動かしている輪の中で self.ai(env) を裸で呼んでいたため、判断中に
    例外が出るとその輪ごと終わっていた。輪が終わると AI は二度と手を出さず、
    ゲームは動いたまま相方だけが固まる。実測のリプレイでは 8.0 秒以降、
    試合の最後まで 17.6 秒間、AI は 1 度も手を出していなかった。

    判断1回ぶんを捨てて次のフレームでやり直せば、止まらない。

実行方法:
    python tests/repro_ai_error_does_not_stop.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from gym_cooking.utils.replay import Replay
from agent.agent.gameplay import GamePlay

results = []


def check(label, ok, detail=""):
    results.append(ok)
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}{(' -> ' + detail) if detail else ''}")


class FakeEnv:
    time = 12.3


class Exploding:
    """何度呼ばれても失敗する AI。"""

    def __init__(self):
        self.calls = 0

    def __call__(self, env):
        self.calls += 1
        raise RuntimeError('わざと失敗させています')


class Working:
    def __call__(self, env):
        return {'ai_0': (0, -1)}, ''


def main():
    game = GamePlay.__new__(GamePlay)     # 盤面を作らずに判断部分だけ試す
    game.replay = Replay()
    game.ai = Exploding()

    moves = [GamePlay._ai_decide(game, FakeEnv()) for _ in range(5)]
    check('例外が出ても呼び出しは戻る', all(m == (None, '') for m in moves),
          f'{moves[0]}')
    check('毎フレーム呼び直す(1回で諦めない)', game.ai.calls == 5,
          f'{game.ai.calls} 回呼ばれた')

    logged = [h for h in list(game.replay) if h['name'] == 'ai_error']
    check('失敗をリプレイに残す', len(logged) == 5,
          f'{len(logged)} 件 / 中身={logged[0]["args"]["error"] if logged else "-"}')

    # 立ち直ったら、また普通に手を出せる
    game.ai = Working()
    move, _ = GamePlay._ai_decide(game, FakeEnv())
    check('直った後はそのまま動ける', move == {'ai_0': (0, -1)}, f'{move}')

    print()
    print(f"[{'SUCCESS' if all(results) else 'FAIL'}] "
          f"{results.count(True)}/{len(results)} 件が期待どおり")
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
