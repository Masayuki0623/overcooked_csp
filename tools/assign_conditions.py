"""参加者ごとの条件の並びを出す(順序統制)。

    python tools/assign_conditions.py --participant p01
    python tools/assign_conditions.py --participant yamada --group G3
    python tools/assign_conditions.py --all 16        # 16人ぶんの表
    python tools/assign_conditions.py --check 16      # 釣り合いを確かめる

割り当ての決まりは experiment_design.py にある。
"""
import argparse
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in ('.', 'testbed-cooking'):
    sys.path.insert(0, str(ROOT / _p))
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

import experiment_design as design  # noqa: E402
from gym_cooking.utils.order_preset import (  # noqa: E402
    enumerate_order_recipes, experiment_case_indices)

PRESET_OF_MAP = {'exp_ring': 'experiment1', 'exp_partition': 'experiment2'}


def case_of(map_name, position):
    cases = experiment_case_indices(PRESET_OF_MAP[map_name]) or [0]
    return design.case_for(cases, position)


def show(participant, group=None):
    g = design.group_of(participant, group)
    first = 'ring先' if g['first_map'] == 'exp_ring' else 'partition先'
    print(f"{participant} ({g['name']}: {first}, 行{g['row']})")
    pr = design.practice_condition(participant, group)
    preset = PRESET_OF_MAP[pr['map']]
    pcase = design.practice_case(len(enumerate_order_recipes(preset)),
                                 experiment_case_indices(preset) or [])
    print(f"  練習      : map={pr['map']:<15s} skip_budget={pr['skip_budget']:<4} "
          f"case={pcase}")
    for c in design.plan_for(participant, group):
        print(f"  session {c['session']}: map={c['map']:<15s} "
              f"skip_budget={str(c['skip_budget']):<4} "
              f"case={case_of(c['map'], c['position'])}")


def table(n):
    print('参加者 | グループ | ' + ' | '.join(f'S{i}' for i in range(1, design.TOTAL + 1)))
    for i in range(1, n + 1):
        pid = f'p{i:02d}'
        g = design.group_of(pid)
        cells = [f"{'R' if c['map'] == 'exp_ring' else 'P'}{c['skip_budget']}"
                 for c in design.plan_for(pid)]
        print(f"{pid:>6} | {g['name']:>8} | " + ' | '.join(f'{c:>5}' for c in cells))


def check(n):
    """釣り合っているかを数える。"""
    by_pos = {i: Counter() for i in range(1, design.TOTAL + 1)}
    first_map = Counter()
    pair = Counter()
    for i in range(1, n + 1):
        pid = f'p{i:02d}'
        plan = design.plan_for(pid)
        first_map[plan[0]['map']] += 1
        for c in plan:
            by_pos[c['session']][(c['map'], c['skip_budget'])] += 1
            pair[(c['skip_budget'], case_of(c['map'], c['position']))] += 1
    print(f'--- {n} 人で確かめる ---')
    print('最初の地図:', dict(first_map))
    ok_pos = all(len(set(cnt.values())) == 1 for cnt in by_pos.values())
    print('各セッション位置で条件が均等か:', 'はい' if ok_pos else 'いいえ')
    for s, cnt in by_pos.items():
        print(f'  session {s}: ' + ', '.join(
            f'{m.replace("exp_", "")}/{b}={v}' for (m, b), v in sorted(
                cnt.items(), key=lambda x: str(x[0]))))
    vals = set(pair.values())
    print('(skip_budget x 注文構成) が均等か:',
          'はい' if len(vals) == 1 else f'いいえ {sorted(vals)}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--participant')
    ap.add_argument('--group', help='G1〜G6。省略すると参加者IDから決める')
    ap.add_argument('--all', type=int, metavar='N', help='N 人ぶんの表を出す')
    ap.add_argument('--check', type=int, metavar='N', help='N 人で釣り合いを確かめる')
    a = ap.parse_args()
    if a.check:
        check(a.check)
    elif a.all:
        table(a.all)
    elif a.participant:
        show(a.participant, a.group)
    else:
        ap.error('--participant か --all か --check を指定してください')


if __name__ == '__main__':
    main()
