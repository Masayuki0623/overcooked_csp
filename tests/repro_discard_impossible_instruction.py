"""実行できなくなった指示を、その時点で棄却することの検証。

背景:
    指示の選択肢は「いまこの瞬間に手を付けられる工程」だけに絞って出して
    いる。ところが選んだあとに盤面が変わって実行できなくなることがある
    (人が材料を先に使った、料理を先に出された、注文が入れ替わった等)。
    そのまま抱えていると、AI は着手できない作業を計画の先頭に置いたまま
    動けなくなる。

ここで確かめること:
    1. 材料が無くなったら、その時点で棄却する
    2. まだできるうちは棄却しない
    3. 取りかかったあとは棄却しない
       (材料を持った・鍋へ入れた時点で盤面の見え方が変わるため)
    4. 鍋が塞がっているだけでは棄却しない
       (材料を1つ入れた時点で塞がる。進めている最中の作業を捨ててしまう)
    5. いまその作業を進めている最中なら棄却しない

実行方法:
    python tests/repro_discard_impossible_instruction.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)

from agent.agent.myagent.CSPAgent import CSPAgent

results = []


def check(label, ok, detail=""):
    results.append(ok)
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}{(' -> ' + detail) if detail else ''}")


class FakeObj:
    def __init__(self, full_name):
        self.full_name = full_name


class FakePot:
    pass


class FakeEnv:
    """盤面にある物の名前と、鍋の空き具合だけを持つ最小の盤面。"""

    def __init__(self, names, pot_busy=False):
        self.all_obj_a = [FakeObj(n) for n in names]
        pot_loc = (9, 9)
        self.pos_gs = {pot_loc: FakePot()}
        self.pos_obj = {pot_loc: FakeObj('ChoppedLettuce')} if pot_busy else {}
        self.time = 30.0


FakePot.__name__ = 'Pot'


def make_agent(pending_task, schedule=None):
    a = CSPAgent.__new__(CSPAgent)
    a._pending_instructions = [{
        'id': 1.0, 'status': 'pending', 'target_idx': 0,
        'task': {'verb': pending_task[0], 'obj': pending_task[1]},
    }]
    a.schedule_per_agent = schedule or {}
    a.current_task_idx = {0: 0, 1: 0}
    return a


def status_after(agent, env):
    agent._drop_unstartable_instructions(env)
    return agent._pending_instructions[0]['status']


# 1. サラダの材料が片方無くなった -> 棄却
a = make_agent(('serve_salad', 'lettuce-tomato salad'))
check('材料が無くなったら棄却する',
      status_after(a, FakeEnv(['ChoppedLettuce'])) == 'canceled')

# 2. 材料がそろっている -> 棄却しない
a = make_agent(('serve_salad', 'lettuce-tomato salad'))
check('まだできるうちは棄却しない',
      status_after(a, FakeEnv(['ChoppedLettuce', 'ChoppedTomato'])) == 'pending')

# 3. 取りかかったあとは見ない
a = make_agent(('serve_salad', 'lettuce-tomato salad'))
a._pending_instructions[0]['execution_logged'] = True
check('取りかかったあとは棄却しない',
      status_after(a, FakeEnv([])) == 'pending')

# 4. 鍋が塞がっているだけでは棄却しない
a = make_agent(('cook', 'lettuce-tomato soup'))
check('鍋が塞がっているだけでは棄却しない',
      status_after(a, FakeEnv(['ChoppedLettuce', 'ChoppedTomato'],
                              pot_busy=True)) == 'pending')

# 4'. 材料そのものが無ければ、鍋の状態によらず棄却
a = make_agent(('cook', 'lettuce-tomato soup'))
check('材料が無ければ調理の指示も棄却する',
      status_after(a, FakeEnv(['ChoppedLettuce'], pot_busy=True)) == 'canceled')

# 5. いまその作業を進めている最中なら棄却しない
a = make_agent(('serve', 'lettuce-tomato soup'),
               schedule={0: [{'id': ('serve', 'lettuce-tomato soup', 1)}]})
check('進めている最中なら棄却しない',
      status_after(a, FakeEnv([])) == 'pending')

# 5'. 別の作業をしているなら棄却する
a = make_agent(('serve', 'lettuce-tomato soup'),
               schedule={0: [{'id': ('chop', 'onion', 1)}]})
check('別の作業をしていて、もうできないなら棄却する',
      status_after(a, FakeEnv([])) == 'canceled')

# 6. 刻む指示は、盤面を見ないので棄却の対象外
a = make_agent(('chop', 'onion'))
check('刻む指示は棄却の対象にしない',
      status_after(a, FakeEnv([])) == 'pending')

ok = sum(1 for r in results if r)
print()
if ok == len(results):
    print(f'[SUCCESS] {ok}/{len(results)} 件が期待どおり')
else:
    print(f'[FAILURE] {ok}/{len(results)} 件のみ期待どおり')
    sys.exit(1)
