"""相手が手に持っている材料を「使える」と数えないことの検証。

報告された現象(20260927_210030):
    「玉ねぎスープを調理して、というのは実際には今すぐできなかった。
      人間が切られたレタスを手に持っていて、それをテーブルに置く前に
      指示がリクエストされた。本当は、テーブルに置いてから初めて
      リクエストされるべき」

原因:
    盤面にある物を数えるとき、持っている物も入れていた。自分が持って
    いる物はそのまま使えるので正しいが、相手の手の中の物は、台に置いて
    もらうまで AI は触れない。それを数に入れていたので、まだ作れない
    料理が指示の候補に出ていた。

ここで確かめること:
    1. 相手が持っている物は、候補を出すときの数に入らない
    2. 自分が持っている物は入る(そのまま使える)
    3. 台に置かれたら入る
    4. 受けた指示がまだできるかを見るときは、相手の持ち物も入れる
       (持っているだけで棄却すると、置いた瞬間にできる指示が消える)

実行方法:
    python tests/repro_partner_held_not_available.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from agent.agent.myagent.CSPAgent import CSPAgent

results = []


def check(label, ok, detail=""):
    results.append(ok)
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}{(' -> ' + detail) if detail else ''}")


class Obj:
    def __init__(self, name):
        self.full_name = name


class Agent:
    def __init__(self, holding=None):
        self.holding = holding


class FakePot:
    pass


FakePot.__name__ = 'Pot'


class FakeEnv:
    """台の上の物と、二人の持ち物だけを持つ最小の盤面。"""

    def __init__(self, on_table, ai_hand=None, human_hand=None, pot_busy=False):
        self.all_obj = [Obj(n) for n in on_table]
        ai = Agent(Obj(ai_hand) if ai_hand else None)
        hu = Agent(Obj(human_hand) if human_hand else None)
        self.agents = [ai, hu]
        self.all_obj_a = self.all_obj + [a.holding for a in self.agents
                                         if a.holding is not None]
        self.pos_gs = {(0, 7): FakePot()}
        self.pos_obj = {(0, 7): object()} if pot_busy else {}


def agent():
    a = CSPAgent.__new__(CSPAgent)
    a.sc_2agent = True
    a.own_agent_idx = 0
    return a


SOUP = {'id': ('cook', 'lettuce-onion soup', 0)}

# 1. 報告の場面。人がレタスを持っていて、台には玉ねぎだけ。
env = FakeEnv(on_table=['ChoppedOnion'], human_hand='ChoppedLettuce')
a = agent()
check('相手が持っている材料は、候補を出すときに数えない',
      a._task_startable_now(env, SOUP, ignore_partner_held=True) is False)

# 2. 自分が持っているぶんは数える
env2 = FakeEnv(on_table=['ChoppedOnion'], ai_hand='ChoppedLettuce')
check('自分が持っている材料は数える',
      a._task_startable_now(env2, SOUP, ignore_partner_held=True) is True)

# 3. 台に置かれたら数える(報告の「置いてから初めて」)
env3 = FakeEnv(on_table=['ChoppedOnion', 'ChoppedLettuce'])
check('台に置かれたら数える',
      a._task_startable_now(env3, SOUP, ignore_partner_held=True) is True)

# 4. 受けた指示がまだできるかを見るときは、相手の持ち物も入れる
check('受けた指示の見直しでは、相手の持ち物も数える',
      a._task_startable_now(env, SOUP, require_station_free=False,
                            ignore_partner_held=False) is True)

# 5. 名前を集めるところの動き
names_all = a._world_object_names(env)
names_mine = a._world_object_names(env, ignore_partner_held=True)
check('相手の持ち物だけが外れる',
      sorted(names_all) == ['ChoppedLettuce', 'ChoppedOnion']
      and names_mine == ['ChoppedOnion'],
      f'{sorted(names_all)} -> {names_mine}')

# 6. 1人だけの地図(相手が居ない)でも落ちない
solo = FakeEnv(on_table=['ChoppedOnion', 'ChoppedLettuce'])
solo.agents = [Agent(None)]
solo.all_obj_a = list(solo.all_obj)
check('相手が居なくても数えられる',
      a._task_startable_now(solo, SOUP, ignore_partner_held=True) is True)

ok = sum(1 for r in results if r)
print()
if ok == len(results):
    print(f'[SUCCESS] {ok}/{len(results)} 件が期待どおり')
else:
    print(f'[FAILURE] {ok}/{len(results)} 件のみ期待どおり')
    sys.exit(1)
