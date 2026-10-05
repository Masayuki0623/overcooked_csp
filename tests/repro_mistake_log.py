"""人のミス(種類・回数・時点)が記録されることの検証。

ミスは3種類(testbed-cooking/gym_cooking/utils/mistakes.py)。
    misserve    注文外の提供
    unusable    残っている注文のどれにも合わない組み合わせを作った
    extra_chop  残っている注文に要る数より多く切った

注文は「トマト・レタスのサラダ / 玉ねぎ・レタスのスープ / 玉ねぎ・レタスの
サラダ」。玉ねぎとトマトを一緒に使う料理は無く、トマトは1つしか要らない。

実行方法:
    python tests/repro_mistake_log.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from gym_cooking.utils.core import (Onion, Tomato, Lettuce, Plate, Object,
                                    FoodState, Floor)
from gym_cooking.utils.interact import INTERACT
from gym_cooking.utils import mistakes

ORDERS = ('TomatoLettuceSalad', 'OnionLettuceSoup', 'OnionLettuceSalad')
STEP = 0.2
results = []


def check(label, ok, detail=""):
    results.append(ok)
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}{(' -> ' + detail) if detail else ''}")


def chopped(cls):
    f = cls()
    f.set_state(FoodState.CHOPPED)
    return f


def free_spot(env, name, used):
    """その種類の空いた台と、その前の床。(台の位置, 床の位置, 向き)"""
    w = env.world
    taken = {tuple(a.location) for a in env.sim_agents}
    for gs in w.get_object_list():
        if getattr(gs, 'name', None) != name or isinstance(gs, Object):
            continue
        loc = tuple(gs.location)
        if loc in used or w.is_occupied(loc):
            continue
        for d in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            floor = (loc[0] - d[0], loc[1] - d[1])
            if not (0 <= floor[0] < w.width and 0 <= floor[1] < w.height):
                continue
            if isinstance(w.get_gridsquare_at(floor), Floor) and floor not in taken:
                used.add(loc)
                return loc, floor, d
    raise RuntimeError(f'{name} の空きが見つかりません')


def stand(env, agent, floor, facing):
    agent.move_to(floor)
    agent.facing = tuple(facing)


def give(env, agent, contents):
    if agent.holding is not None:
        env.world.remove(agent.holding)
        agent.release()
    if not contents:
        return
    held = Object(location=tuple(agent.location), contents=list(contents))
    env.world.insert(held)
    agent.acquire(held)


def put(env, loc, contents):
    obj = Object(location=loc, contents=list(contents))
    env.world.insert(obj)
    env.world.get_gridsquare_at(loc).acquire(obj)


def press(env, agent):
    acts = {a.name: (0, 0) for a in env.sim_agents}
    acts[agent.name] = INTERACT
    env.step(acts, passed_time=STEP)


def kinds(env, by=None):
    return [m['type'] for m in env.mistake_log if by is None or m['by'] == by]


def main():
    env = OvercookedEnvironment(MapSetting(level='exp_ring_2pot_veg',
                                           order_recipes=ORDERS))
    env.reset()
    ai, human = env.sim_agents[0], env.sim_agents[1]
    used = set()
    print(f'[SETUP] 注文={ORDERS}')

    # --- 使える組み合わせは数えない: レタスの上にトマト ---
    c, floor, d = free_spot(env, 'Counter', used)
    put(env, c, [chopped(Lettuce)])
    stand(env, human, floor, d)
    give(env, human, [chopped(Tomato)])
    press(env, human)
    check('注文にある組み合わせ(トマト+レタス)は数えない', kinds(env) == [], str(env.mistake_log))

    # --- 使えない組み合わせ: トマトの上に玉ねぎ ---
    c, floor, d = free_spot(env, 'Counter', used)
    put(env, c, [chopped(Tomato)])
    stand(env, human, floor, d)
    give(env, human, [chopped(Onion)])
    press(env, human)
    check('注文に無い組み合わせ(玉ねぎ+トマト)を数える', kinds(env, human.name) == ['unusable'],
          str(env.mistake_log))
    t_unusable = env.mistake_log[-1]['time'] if env.mistake_log else None
    check('時点が残る', t_unusable == round(env.current_time, 1), str(t_unusable))

    # 同じ山に皿を足しても、もう1回とは数えない
    give(env, human, [Plate()])
    press(env, human)
    check('同じ山に皿を足しても増えない', kinds(env) == ['unusable'], str(env.mistake_log))
    give(env, human, [])            # 手を空ける(山は皿ごと手に移っている)

    # --- 鍋: ぴったり合う注文が無ければ使えない ---
    p, floor, d = free_spot(env, 'Pot', used)
    stand(env, human, floor, d)
    give(env, human, [chopped(Lettuce), chopped(Onion)])
    press(env, human)
    check('注文どおりのスープ(玉ねぎ+レタス)は数えない', kinds(env) == ['unusable'],
          str(env.mistake_log))
    p, floor, d = free_spot(env, 'Pot', used)
    stand(env, human, floor, d)
    give(env, human, [chopped(Lettuce), chopped(Tomato)])
    press(env, human)
    check('注文に無いスープ(トマト+レタス)を鍋に入れたら数える',
          kinds(env) == ['unusable', 'unusable'], str(env.mistake_log))

    # --- 余分に切った: トマトは1つしか要らない ---
    # ここまでで、使えるトマトは最初の「トマト+レタス」の1つ(残りの2つは
    # 使えない山と使えない鍋の中)。もう1つ切れば余分。
    n0 = len(env.mistake_log)
    b, floor, d = free_spot(env, 'Cutboard', used)
    stand(env, human, floor, d)
    give(env, human, [Tomato()])
    press(env, human)                # 置く(切り始め)
    check('要る数を超えて切り始めたら数える', kinds(env)[n0:] == ['extra_chop'],
          str(env.mistake_log[n0:]))
    for _ in range(8):               # 切り終えるまで押す
        obj = env.world.get_object_at(b, None, find_held_objects=False)
        if not obj.needs_chopped():
            break
        press(env, human)
    check('切り終えても2回目とは数えない', kinds(env)[n0:] == ['extra_chop'],
          str(env.mistake_log[n0:]))

    # 要る数の内なら数えない: 玉ねぎは2つ要り、いま使えるのは鍋の1つだけ
    n1 = len(env.mistake_log)
    b, floor, d = free_spot(env, 'Cutboard', used)
    stand(env, human, floor, d)
    give(env, human, [Onion()])
    press(env, human)
    check('要る数の内(玉ねぎ2つ目)は数えない', kinds(env)[n1:] == [], str(env.mistake_log[n1:]))

    # --- 注文外の提供 ---
    n2 = len(env.mistake_log)
    g, floor, d = free_spot(env, 'Delivery', used)
    stand(env, human, floor, d)
    give(env, human, [chopped(Onion), Plate()])
    press(env, human)
    check('注文に無い物を出したら数える', kinds(env)[n2:] == ['misserve'], str(env.mistake_log[n2:]))

    # --- AI がやったぶんは、人のぶんに混ざらない ---
    # (前の「玉ねぎ+トマト」の山は盤面から消えているので、同じ組み合わせを
    #  もう一度作れば、もう1回と数える)
    c, floor, d = free_spot(env, 'Counter', used)
    put(env, c, [chopped(Tomato)])
    stand(env, ai, floor, d)
    give(env, ai, [chopped(Onion)])
    press(env, ai)
    s_h = mistakes.summarize(env.mistake_log, by=human.name)
    s_a = mistakes.summarize(env.mistake_log, by=ai.name)
    check('人のぶんの集計', s_h['counts'] == {'misserve': 1, 'unusable': 2, 'extra_chop': 1},
          str(s_h['counts']))
    check('AI のぶんは別に数える', s_a['total'] == 1, str(s_a['counts']))
    print('  内訳(人):', s_h['detail'])

    print()
    print(f"[{'SUCCESS' if all(results) else 'FAIL'}] "
          f"{results.count(True)}/{len(results)} 件が期待どおり")
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
