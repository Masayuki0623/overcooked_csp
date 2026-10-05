"""手を出している人のマスへ相手が入ろうとしても、手を出すほうは取り消されないことの検証。

想定シナリオ(模擬対戦で 116 秒止まった場面):
    AI は皿を持って鍋の前に立ち、煮上がったスープを取ろうとしている。
    相方も同じスープを取りに来て、AI のいるマスへ進もうとし続ける。

    衝突の判定は「手を出す」を「その場に留まる」に読み替えていたので、
    相手が同じマスへ入ろうとすると、入ろうとした側だけでなく、手を出して
    いる側の行動まで取り消していた。相手が押し続ける限り、AI は鍋の前で
    何もできない。人が AI に押されている間、切る・取るが効かないのも同じ。

実行方法:
    python tests/repro_interact_not_cancelled_by_push.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from gym_cooking.utils.core import Lettuce, Tomato, Plate, Object, FoodState, Floor
from gym_cooking.utils.interact import INTERACT

ORDERS = ('TomatoLettuceSoup', 'FullSoup', 'OnionLettuceSalad')
results = []


def check(label, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}{(' -> ' + detail) if detail else ''}")


def cooked(cls):
    f = cls()
    f.set_state(FoodState.COOKED)
    # 煮上がった時刻を持たせる(鍋に残ったままだと、環境が毎手これを見る)。
    f.state = FoodState.COOKED(obj=f.name, start_time=0.1)
    return f


def main():
    env = OvercookedEnvironment(MapSetting(level='exp_ring_2pot_veg', order_recipes=ORDERS))
    env.reset()
    w = env.world
    ai, partner = env.sim_agents[0], env.sim_agents[1]

    # 鍋と、その前の床。床のもう1つ隣(鍋と反対でない側)に相方を立たせる。
    pot = next(tuple(g.location) for g in w.get_object_list()
               if getattr(g, 'name', None) == 'Pot' and not isinstance(g, Object))
    front = side = None
    for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        f = (pot[0] + d[0], pot[1] + d[1])
        if not (0 <= f[0] < w.width and 0 <= f[1] < w.height):
            continue
        if not isinstance(w.get_gridsquare_at(f), Floor):
            continue
        for e in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            s = (f[0] + e[0], f[1] + e[1])
            if s != pot and 0 <= s[0] < w.width and 0 <= s[1] < w.height \
                    and isinstance(w.get_gridsquare_at(s), Floor):
                front, side, facing, push = f, s, (-d[0], -d[1]), (-e[0], -e[1])
                break
        if front:
            break
    soup = Object(location=pot, contents=[cooked(Lettuce), cooked(Tomato)])
    w.insert(soup)
    w.get_gridsquare_at(pot).acquire(soup)

    ai.move_to(front)
    ai.facing = facing
    plate = Object(location=front, contents=[Plate()])
    w.insert(plate)
    ai.acquire(plate)
    partner.move_to(side)
    print(f'[SETUP] 鍋 {pot} に煮上がったスープ。AI は {front} で皿を持って鍋を向く。'
          f'相方は {side} から AI のマスへ進もうとする')

    acts = {ai.name: INTERACT, partner.name: push}
    env.step(acts, passed_time=0.2)
    got = ai.holding.full_name if ai.holding is not None else None
    check('相手に押されていても、鍋からスープを取れる', got and 'Cooked' in got, str(got))
    check('押した側は AI のマスへ入れない', tuple(partner.location) == side, str(partner.location))

    print()
    print(f"[{'SUCCESS' if all(results) else 'FAIL'}] "
          f"{results.count(True)}/{len(results)} 件が期待どおり")
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
