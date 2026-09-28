"""同じ台で拾う/置くを毎フレーム繰り返すのを、実行の出口で止めることの検証。

想定シナリオ(実測: web2-exp_ring 20260928_230525、16.6〜20.2 秒):
    切ったトマトを持っては置き、置いては拾い、を 0.2 秒ごとに繰り返した。
    参加者には「作業を中断している」ように見える。原因は工程によって
    違うので、出口で「持ち物が短時間に何度も変わり、次の一手がまた台への
    働きかけ(壁側へ進む)なら 1 秒手を止める」として押さえる。

実行方法:
    python tests/repro_pick_put_flicker_stop.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from agent.agent.myagent.CSPAgent import CSPAgent

results = []


def check(label, got, want):
    ok = got == want
    results.append(ok)
    print(f"  [{'OK  ' if ok else 'NG  '}] {label} -> {got!r}")
    if not ok:
        print(f"         期待: {want!r}")


class Item:
    def __init__(self, name):
        self.full_name = name


class E:
    """AI から見た盤面の最小限。(2,2) に立ち、右 (3,2) は台、上 (2,1) は床。"""

    def __init__(self, t, hold):
        self.time = t
        self.hold = Item(hold) if hold else None
        self.self_pos = (2, 2)
        g = [[1] * 5 for _ in range(5)]
        g[3][2] = 0     # 右隣は台(歩けない) = そちらへ進むとインタラクト
        self.to_grid = g


def main():
    ai = CSPAgent.__new__(CSPAgent)
    ai._hold_hist = {}
    ai._flicker_hold_until = {}
    ai._emit_counter_debug = lambda *a, **k: None
    RIGHT, UP = (1, 0), (0, -1)

    print('[1] 拾う/置くが続いたら、台への働きかけを止める')
    seq = [(0.0, 'ChoppedTomato'), (0.2, None), (0.4, 'ChoppedTomato'), (0.6, None)]
    out = None
    for t, h in seq:
        out = ai._stop_pick_put_flicker(E(t, h), 0, RIGHT, '食材の取得')
    check('4回目の判断で止まる', out, ((0, 0), '拾い置きが続くので少し待つ'))
    check('止めている間は何を返されても止める',
          ai._stop_pick_put_flicker(E(0.8, 'ChoppedTomato'), 0, RIGHT, '置く')[0], (0, 0))
    check('1秒たてば元に戻る',
          ai._stop_pick_put_flicker(E(1.7, 'ChoppedTomato'), 0, UP, '移動')[0], UP)

    print('[2] 往復していても、ただ歩くだけなら止めない')
    ai._hold_hist = {}
    ai._flicker_hold_until = {}
    for t, h in seq[:-1]:
        ai._stop_pick_put_flicker(E(t, h), 0, RIGHT, 'x')
    check('床へ進む一手はそのまま', ai._stop_pick_put_flicker(E(0.6, None), 0, UP, '移動')[0], UP)

    print('[3] 持ち物が変わっていなければ何もしない')
    ai._hold_hist = {}
    ai._flicker_hold_until = {}
    for t in (0.0, 0.2, 0.4, 0.6):
        out = ai._stop_pick_put_flicker(E(t, 'Plate'), 0, RIGHT, '置く')
    check('同じ物を持ち続けていれば止めない', out[0], RIGHT)

    print('[4] 変わり方が遅ければ(1.2秒に2回以下)止めない')
    ai._hold_hist = {}
    ai._flicker_hold_until = {}
    for t, h in [(0.0, 'A'), (1.0, None), (2.0, 'A'), (3.0, None)]:
        out = ai._stop_pick_put_flicker(E(t, h), 0, RIGHT, '置く')
    check('普通の拾い置きは止めない', out[0], RIGHT)

    ng = results.count(False)
    if ng:
        print(f'[FAILED] {ng}/{len(results)} 件が期待どおりではありません')
        return 1
    print(f'[SUCCESS] {len(results)}/{len(results)} 件すべて期待どおり')
    return 0


if __name__ == '__main__':
    sys.exit(main())
