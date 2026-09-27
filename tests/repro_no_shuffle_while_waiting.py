"""相手の材料を待っている間、取って置いてを繰り返さないことの検証。

報告された現象:
    「XYスープの調理が AI に割り当てられているとき、X がテーブルの上に
      あって Y がまだできていなかったり、プレイヤーが運んできている途中
      だと、AI が X を置いたり取ったりを繰り返して急かしているように
      見える」

仕組み:
    手が空くと、AI は足りない材料を探しに行く。台に乗っている X は
    「取りに行くべき材料」に見えるので拾う。ところが1人では重ねる相手が
    いないので、結局どこかの台へ置き戻すだけ。置いた次の瞬間また拾う。
    歯止めはあったが「指定の合流台に乗っているとき」に限られていて、
    合流台の割り当てが変われば素通りしていた
    (実測: 置いた 0.2 秒後に拾い直していた)。

いまの決まり:
    途中までの山は、重ねられる相手が他にあるときだけ取りに行く。
    同じ材料どうしは重ねられないので、材料が重ならない山があるかで見る。
    相手が1つも無ければ取らない(置き戻すだけで前に進まないため)。

ここで確かめること:
    1. 山が1つだけなら取りに行かない(相手が持ってくるのを待つ)
    2. 重ねられる山が別にあるなら、これまでどおり取りに行く
    3. 同じ材料の山が2つでも取りに行かない(重ねられないので無意味)
    4. 材料が全部そろった山は、途中の山ではないので取りに行く
    5. 最後の手段としては残す(呼び出し側が使えるように候補では返す)

実行方法:
    python tests/repro_no_shuffle_while_waiting.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from gym_cooking.utils.core import Object, Onion, Tomato
from agent.agent.myagent.TaskAgent import TaskAgent

results = []


def check(label, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}{(' -> ' + detail) if detail else ''}")


def chopped(kind, loc):
    ing = kind()
    for _ in range(20):
        if 'Chopped' in ing.full_name:
            break
        ing.update_state(0.0)
    return Object(location=loc, contents=[ing])


def merged(kinds, loc):
    obj = chopped(kinds[0], loc)
    for k in kinds[1:]:
        obj.contents.append(chopped(k, loc).contents[0])
    return obj


class FakeCounter:
    pass


FakeCounter.__name__ = 'Counter'


class FakeEnv:
    def __init__(self, contents, self_pos=(5, 3)):
        self.pos_gs = {pos: FakeCounter() for pos in contents}
        self.pos_obj = dict(contents)
        self.self_pos = self_pos
        self.order = None
        self.time = 20.0

    def get_pos_by_obj_gs(self, gs=None, obj=None):
        return list(self.pos_gs) if gs == 'Counter' else []


TaskAgent.reachable_positions = classmethod(lambda cls, env, ps: list(ps))
TaskAgent.can_use_position = classmethod(lambda cls, env, pos: True)


def agent():
    a = TaskAgent.__new__(TaskAgent)
    a.order_ingredients = ['onion', 'tomato']
    a.strict_counter_management = False
    a.protected_counters = set()
    return a


INGS = ['onion', 'tomato']
MISSING = ['ChoppedOnion', 'ChoppedTomato']


def find(env, assigned=None):
    return agent()._find_chopped_pickup_target(
        env, INGS, MISSING, assigned, False)


# 1. 山が1つだけ -> 取りに行かない
env = FakeEnv({(6, 3): None, (6, 4): chopped(Onion, (6, 4))})
target, _s, cand, _cs = find(env)
check('山が1つだけなら取りに行かない', target is None, f'取りに行く先={target}')
# 5
check('最後の手段としては候補に残す', cand == (6, 4), f'候補={cand}')

# 2. 重ねられる山が別にある -> 取りに行く
env = FakeEnv({(6, 4): chopped(Onion, (6, 4)), (6, 5): chopped(Tomato, (6, 5))})
target, _s, _c, _cs = find(env)
check('重ねられる山が別にあれば取りに行く', target in ((6, 4), (6, 5)),
      f'取りに行く先={target}')

# 3. 同じ材料の山が2つ -> 重ねられないので取りに行かない
env = FakeEnv({(6, 4): chopped(Onion, (6, 4)), (6, 5): chopped(Onion, (6, 5))})
target, _s, _c, _cs = find(env)
check('同じ材料の山が2つでも取りに行かない', target is None,
      f'取りに行く先={target}')

# 4. 全部そろった山 -> 取りに行く
env = FakeEnv({(6, 4): merged([Onion, Tomato], (6, 4))})
target, _s, _c, _cs = find(env)
check('材料が全部そろった山は取りに行く', target == (6, 4), f'取りに行く先={target}')

# 指定の合流台にある途中の山も、これまでどおり取りに行かない
env = FakeEnv({(6, 3): None, (6, 4): chopped(Onion, (6, 4))})
target, _s, cand, _cs = find(env, assigned=(6, 4))
check('指定の合流台にある途中の山も取りに行かない', target is None,
      f'取りに行く先={target}')

print()
ok = sum(1 for r in results if r)
if ok == len(results):
    print(f'[SUCCESS] {ok}/{len(results)} 件すべて期待どおり')
else:
    print(f'[FAILURE] {ok}/{len(results)} 件のみ期待どおり')
    sys.exit(1)
