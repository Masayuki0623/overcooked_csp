"""別のジュースが入ったミキサーで止まらないことの検証。

想定シナリオ(実際の報告: 20260928_221537 / p05):
    注文が2つのジュースを同時に抱えている。AI はミキサーへ片方の材料を
    入れて回し、混ぜ終わった。ところが計画の先頭はもう片方のジュースの
    mix になっていて、ミキサーには別の中身が入ったまま。
    ミキサーは鍋と同じで、中身があると足せない。取り出すには注がなければ
    ならないが、その注ぎは計画の後ろにある。
    -> 「ミキサーの中身が注文と違う。取り出せないため進められない」を
       返し続けて、18秒その場から動かないまま試合が終わった。

ここで見るのは2点:
    1. ミキサーの中身の名前(Mixing.../Mixed...)を、鍋(Cooking/Cooked)と
       同じように素材名へ戻せること。戻せないと、中身がどの注文の物か
       分からず、注ぎにも行けない。
    2. 塞がっているとき、それを空けられる作業(混ぜかけなら回し切る mix、
       混ぜ終わりなら serve_juice)を計画から見つけられること。

実行方法:
    python tests/repro_blender_blocked_by_other_juice.py
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


class FakeFood:
    def __init__(self, name):
        self.full_name = name
        self.name = name


class FakeBlender:
    """中身と、混ざりきったかどうかだけを持つミキサー。"""

    def __init__(self, names, mixed):
        self.contents = [FakeFood(n) for n in names]
        self.full_name = '-'.join(names)
        self._mixed = mixed

    def is_mixed(self):
        return self._mixed


class FakeEnv:
    def __init__(self, pos_obj):
        self.pos_obj = pos_obj


def make_agent(blenders):
    ai = CSPAgent.__new__(CSPAgent)
    ai._get_resources = lambda env: {'blenders': blenders}
    return ai


BL = (0, 4)


def main():
    ai = make_agent([BL])

    print('[1] ミキサーの中身の名前を素材名へ戻せる')
    check('混ぜかけ (Mixing)', ai._normalize_ingredient_name('MixingBanana'), 'banana')
    check('混ぜ終わり (Mixed)', ai._normalize_ingredient_name('MixedOrange'), 'orange')
    check('鍋側はこれまでどおり', ai._normalize_ingredient_name('CookedOnion'), 'onion')

    print('[2] 中身が分かるので、どの注文の物かを突き合わせられる')
    env = FakeEnv({BL: FakeBlender(['MixingBanana', 'MixingOrange'], mixed=False)})
    check('中身', ai._get_counter_food_names(env, BL), {'banana', 'orange'})
    check('同じジュースなら使える',
          ai._has_usable_blender_for_mix(env, 'orange-banana juice'), True)
    check('別のジュースには使えない',
          ai._has_usable_blender_for_mix(env, 'apple-banana juice'), False)

    print('[3] 塞がっているとき、空けられる作業を計画から見つける')
    plan = [
        {'id': ('mix', 'orange-banana juice', 2)},
        {'id': ('serve_juice', 'orange-banana juice', 2)},
        {'id': ('mix', 'apple-banana juice', 1)},
        {'id': ('serve_juice', 'apple-banana juice', 1)},
    ]
    ai.schedule_per_agent = {0: plan, 1: [{'id': ('chop', 'banana', 0)}]}
    check('混ぜかけなら、回し切る mix',
          (ai._find_blender_unblock_task(env, 0) or {}).get('id'),
          ('mix', 'orange-banana juice', 2))

    env_done = FakeEnv({BL: FakeBlender(['MixedBanana', 'MixedOrange'], mixed=True)})
    check('混ぜ終わりなら、注ぐ serve_juice',
          (ai._find_blender_unblock_task(env_done, 0) or {}).get('id'),
          ('serve_juice', 'orange-banana juice', 2))

    print('[4] 相手の計画にしか無くても見つける(塞がったミキサーは相手も使えない)')
    ai.schedule_per_agent = {0: [{'id': ('mix', 'apple-banana juice', 1)}],
                             1: [{'id': ('serve_juice', 'orange-banana juice', 2)}]}
    check('相手の分から拾う',
          (ai._find_blender_unblock_task(env_done, 0) or {}).get('id'),
          ('serve_juice', 'orange-banana juice', 2))

    print('[5] 空けられる作業が無ければ None(前の挙動のまま)')
    ai.schedule_per_agent = {0: [{'id': ('mix', 'apple-banana juice', 1)}]}
    check('見つからない', ai._find_blender_unblock_task(env_done, 0), None)
    check('空のミキサーは誰でも使える',
          ai._has_usable_blender_for_mix(FakeEnv({}), 'apple-banana juice'), True)

    ng = results.count(False)
    if ng:
        print(f'[FAILED] {ng}/{len(results)} 件が期待どおりではありません')
        return 1
    print(f'[SUCCESS] {len(results)}/{len(results)} 件すべて期待どおり')
    return 0


if __name__ == '__main__':
    sys.exit(main())
