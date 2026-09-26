"""同じ材料を取って置いてを繰り返さないことの検証。

報告された現象:
    「玉ねぎを取って置いてを繰り返していた。
      そして三回くらいそれを繰り返すと指示リクエストが起きた」
    (20260926_153223 の回。17.2〜20.4 秒のあいだ 0.2 秒ごとに
     ChoppedOnion を取る/置くを 16 回繰り返していた)

仕組み:
    指定の合流台に同じ材料が既に乗っていると重ねられない。そのとき
    「空いている別の台」へ逃がしていた。ところが置いた次の瞬間、材料を
    探す側から見ると、そこは「取りに行くべき材料がある台」なので拾い直す。
    指定の台と往復を止める番人はあったが、逃がし先の台は見ていなかった。

ここで確かめること:
    1. 重ねられる山があるなら、空き台ではなくそちらへ置きに行く
       (置いた時点で合流が済むので往復にならない)
    2. どの注文にもならない組み合わせは作らない
    3. 重ねられる山が無いときは、これまでどおり空き台へ逃がす

実行方法:
    python tests/repro_pickup_putdown_loop.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from gym_cooking.utils.core import Object, Onion, Tomato, Lettuce
from agent.agent.myagent.TaskAgent import TaskAgent

results = []


def check(label, ok, detail=""):
    results.append(ok)
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}{(' -> ' + detail) if detail else ''}")


def chopped(kind, loc):
    """刻んだ材料を1つ作る。"""
    ing = kind()
    for _ in range(20):         # Fresh -> Chopped まで刻む
        if 'Chopped' in ing.full_name:
            break
        ing.update_state(0.0)
    return Object(location=loc, contents=[ing])


class FakeCounter:
    pass


FakeCounter.__name__ = 'Counter'


class FakeEnv:
    """台と、その上の物だけを持つ最小の盤面。全部の台に手が届く。"""

    def __init__(self, contents):
        self.pos_gs = {pos: FakeCounter() for pos in contents}
        self.pos_obj = dict(contents)
        self.self_pos = (5, 3)
        self.order = None
        self.time = 20.0

    def get_pos_by_obj_gs(self, gs=None, obj=None):
        return list(self.pos_gs) if gs == 'Counter' else []


def make_agent(order_ings):
    a = TaskAgent.__new__(TaskAgent)
    a.order_ingredients = order_ings
    a.strict_counter_management = False
    return a


# 盤面: 指定台(6,4) に玉ねぎ / (6,5) にトマト / (6,3) は空
def board():
    return FakeEnv({
        (6, 3): None,
        (6, 4): chopped(Onion, (6, 4)),
        (6, 5): chopped(Tomato, (6, 5)),
    })


TaskAgent.reachable_positions = classmethod(lambda cls, env, ps: list(ps))
TaskAgent.can_use_position = classmethod(lambda cls, env, pos: True)

hold = chopped(Onion, (5, 3))
a = make_agent(['onion', 'tomato'])
env = board()

# 1. 重ねられる山(トマト)を選ぶ。空き台(6,3)ではない。
dest, _ = a._resolve_assigned_counter_target(env, hold, (6, 4), 'blocked')
check('重ねられる山へ置きに行く(空き台ではない)', dest == (6, 5),
      f'置き先={dest}')

# 2. どの注文にもならない組み合わせは作らない
#    レタスしか要らない注文なら、トマトの山に玉ねぎを足してはいけない
a2 = make_agent(['onion', 'lettuce'])
dest2, _ = a2._resolve_assigned_counter_target(env, hold, (6, 4), 'blocked')
check('注文にならない山には置かない', dest2 == (6, 3), f'置き先={dest2}')

# 3. 重ねられる山が無ければ、これまでどおり空き台へ
env3 = FakeEnv({(6, 3): None, (6, 4): chopped(Onion, (6, 4))})
dest3, _ = a._resolve_assigned_counter_target(env3, hold, (6, 4), 'blocked')
check('重ねられる山が無ければ空き台へ', dest3 == (6, 3), f'置き先={dest3}')

ok = sum(1 for r in results if r)
print()
if ok == len(results):
    print(f'[SUCCESS] {ok}/{len(results)} 件が期待どおり')
else:
    print(f'[FAILURE] {ok}/{len(results)} 件のみ期待どおり')
    sys.exit(1)
