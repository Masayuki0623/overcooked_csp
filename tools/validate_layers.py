"""層ごとに、壊れていないかを段階的に検査する。

エピソードを最後まで走らせて「完走しなかった」と分かっても、どの層が
壊れているのかは分からない。下の層から順に検査して、最初に壊れている
場所を特定できるようにする。

  第1層 計画: 料理に必要な工程が全部あるか。所要時間が測れるか
  第2層 割当: 各タスクを実行できる担当者がいるか
  第3層 実行: その担当者が、実際にそのタスクを進められるか
  第4層 統合: 通しで完走するか(これは既存の diagnose_stalls が担当)

    python tools/validate_layers.py            # 全18通り
    python tools/validate_layers.py --case 0   # 1通りだけ詳しく
"""
import argparse
import os
import sys
from copy import deepcopy as dcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'agent'))
sys.path.insert(0, str(ROOT / 'testbed-cooking'))
sys.path.insert(0, str(ROOT / 'tools'))

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

from gym_cooking.utils.order_preset import enumerate_order_recipes  # noqa: E402
import run_human_model_experiment as H  # noqa: E402

# 料理の系統ごとに、最低限そろっていなければならない工程。
# 提供は、仕切りの向こうに提供口がある場合 handover + serve_from_counter に
# 分かれるので、どちらかの形であればよい。
REQUIRED = {
    'salad': {'chop', ('serve_salad', 'handover')},
    'soup': {'chop', 'cook', ('serve', 'handover')},
    'juice': {'chop', 'mix', ('serve_juice', 'handover')},
}


def kind_of(name):
    for k in ('salad', 'soup', 'juice'):
        if name.endswith(k):
            return k
    return None


def check_case(case, recipes, verbose=False):
    """1つの注文構成について、第1〜3層を検査する。"""
    problems = []
    env = H.make_env('experiment', case, 'experiment2', recipes)
    ai = H.make_ai(0, partner_is_external=True)
    state = H.state_for(env, 0)
    orders = ai._build_order_tasks(dcopy(state))

    for o in orders:
        kind = kind_of(o['name'])
        verbs = [t['id'][0] for t in o['tasks']]
        if verbose:
            print(f"    注文{o['order']} {o['name']}: {verbs}")

        # --- 第1層: 工程がそろっているか ---
        for need in REQUIRED.get(kind, set()):
            names = need if isinstance(need, tuple) else (need,)
            if not any(v in names for v in verbs):
                problems.append(f'[計画] 注文{o["order"]} {o["name"]}: '
                                f'{"/".join(names)} が無い (実際: {verbs})')

        for t in o['tasks']:
            tid = t['id']
            # --- 第1層: 所要時間が測れるか ---
            if t.get('dur') is None:
                problems.append(f'[計画] {tid}: 所要時間が測れていない')

            # --- 第2層: 実行できる担当者がいるか ---
            allowed = ai._task_allowed_agents(state, t)
            if not allowed:
                assignable = ai._assignable_agents(state, t)
                if not assignable:
                    problems.append(f'[割当] {tid}: 実行できる担当者がいない')

            # --- 第2層: 合流地点が両側から使えるか ---
            counter = t.get('assigned_counter')
            if counter is not None and ai._map_is_partitioned(state):
                comps = ai._components_touching(state, tuple(counter))
                if len(comps) < len(ai._walkable_components(state)):
                    problems.append(f'[割当] {tid}: 合流地点 {counter} が片側からしか使えない')

    # --- 第3層: 担当者が実際にその位置へ行けるか ---
    for o in orders:
        for t in o['tasks']:
            for key in ('start_pos', 'end_pos'):
                pos = t.get(key)
                if pos is None:
                    continue
                allowed = ai._task_allowed_agents(state, t)
                for a in allowed:
                    st_a = H.state_for(env, a)
                    if ai.astar_distance(st_a, st_a.self_pos, tuple(pos)) is None:
                        problems.append(f'[実行] {t["id"]}: A{a} が {key}={pos} へ行けない')
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', type=int, default=None)
    args = ap.parse_args()

    sets = enumerate_order_recipes('experiment2')
    cases = [args.case] if args.case is not None else range(len(sets))

    total = 0
    for case in cases:
        verbose = args.case is not None
        if verbose:
            print(f'case {case}: ' + ' | '.join(sets[case]))
        problems = check_case(case, sets[case], verbose)
        total += len(problems)
        if problems:
            print(f'case {case:2d}: 問題 {len(problems)} 件')
            for p in problems:
                print(f'    {p}')
        elif verbose:
            print(f'case {case:2d}: 問題なし')

    print(f'\n検査した注文構成 {len(list(cases))} 通り / 問題 {total} 件')


if __name__ == '__main__':
    main()
