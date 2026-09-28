"""皿に載せたスープを持ったまま、相手の作業を横取りしないことの検証。

想定シナリオ(実際の報告: 20260928_223402 / p06「なぜスープを提供しないのか」):
    スープが煮上がり、AI は皿に載せて持っている。提供口へ行くだけ。
    ところが「提供が進められるか」を刻んだ材料の有無で見ていたため、
    煮えた時点で材料は消えていて「進められない」と判定された。
    -> 相手の担当(レタスを刻む)を引き取り、皿ごと台に置いて、
       そのまま出さずに試合が終わった。

ここで見るのは2点:
    1. 出来上がった料理が世界(鍋・台・手)のどこかにあれば、提供系の
       工程は「進められる」と見なすこと。
    2. 空の皿・コップ以外を持っている間は、横取りしないこと。

実行方法:
    python tests/repro_serve_not_abandoned_for_takeover.py
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


class Thing:
    def __init__(self, name, contents=()):
        self.full_name = name
        self.name = name
        self.contents = list(contents)


class Agent:
    def __init__(self, holding=None):
        self.holding = holding


class FakeEnv:
    def __init__(self, pos_obj=None, agents=()):
        self.pos_obj = dict(pos_obj or {})
        self.pos_gs = {}
        self.world = None
        self.agents = list(agents)


def main():
    ai = CSPAgent.__new__(CSPAgent)

    print('[1] 出来上がった料理を、置き場所によらず見つける')
    soup = 'lettuce-onion soup'
    check('鍋の中(煮えた)',
          ai._finished_dish_exists(FakeEnv({(0, 3): Thing('CookedLettuce-CookedOnion')}), soup), True)
    check('皿に載せて手に持っている',
          ai._finished_dish_exists(FakeEnv(agents=[Agent(Thing('CookedLettuce-CookedOnion-Plate'))]), soup), True)
    check('皿ごと台に置いてある',
          ai._finished_dish_exists(FakeEnv({(2, 2): Thing('CookedLettuce-CookedOnion-Plate')}), soup), True)
    check('まだ煮えている途中は数えない',
          ai._finished_dish_exists(FakeEnv({(0, 3): Thing('CookingLettuce-CookingOnion')}), soup), False)
    check('材料が片方しかない鍋は数えない',
          ai._finished_dish_exists(FakeEnv({(0, 3): Thing('CookedOnion')}), soup), False)
    check('ジュースは混ざり終わり(Mixed)で見る',
          ai._finished_dish_exists(FakeEnv({(0, 4): Thing('MixedApple-MixedBanana')}), 'apple-banana juice'), True)
    check('混ぜかけ(Mixing)は数えない',
          ai._finished_dish_exists(FakeEnv({(0, 4): Thing('MixingApple-MixingBanana')}), 'apple-banana juice'), False)
    check('サラダは刻んだ材料が皿に載っていればよい',
          ai._finished_dish_exists(FakeEnv({(2, 2): Thing('ChoppedLettuce-ChoppedTomato-Plate')}), 'lettuce-tomato salad'), True)

    print('[2] 刻んだ材料で見ると、煮えた時点で「無い」になる(これが誤判定の元)')
    env = FakeEnv(agents=[Agent(Thing('CookedLettuce-CookedOnion-Plate'))])
    check('材料で見る(旧)', ai._cook_dependency_ready_from_world(env, soup), False)
    check('料理で見る(新)', ai._finished_dish_exists(env, soup), True)

    ng = results.count(False)
    if ng:
        print(f'[FAILED] {ng}/{len(results)} 件が期待どおりではありません')
        return 1
    print(f'[SUCCESS] {len(results)}/{len(results)} 件すべて期待どおり')
    return 0


if __name__ == '__main__':
    sys.exit(main())
