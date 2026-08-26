"""いまの盤面で、どのタスクを誰が実行できるかを一覧で出す。

「タスクと資材の紐づけを担当者ごとに解く」改修が効いているかを、
目で確かめるための観測用スクリプト。ゲーム側は何も変更しない。

    python tools/show_task_eligibility.py             # 全列挙の 0 番目
    python tools/show_task_eligibility.py --case 7
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', type=int, default=0, help='全列挙した注文構成の番号')
    ap.add_argument('--map', default='experiment')
    args = ap.parse_args()

    sets = enumerate_order_recipes('experiment2')
    recipes = sets[args.case % len(sets)]
    env = H.make_env(args.map, args.case, 'experiment2', recipes)
    ai = H.make_ai(0, partner_is_external=True)
    state = H.state_for(env, 0)

    print(f'地図: {args.map} / 注文: ' + ' | '.join(recipes))
    print(f'歩ける領域の数: {len(ai._walkable_components(state))}'
          f'  (2以上なら仕切りあり)')
    print(f'AI(0番)の位置: {env.sim_agents[0].location}'
          f'  人(1番)の位置: {env.sim_agents[1].location}')
    print()

    for idx in (0, 1):
        res = ai._resources_for_agent(state, idx)
        who = 'AI(0番)' if idx == 0 else '人(1番)'
        print(f'{who} が自分の側で使える資材:')
        for key, label in (('cutboards', 'まな板'), ('pots', '鍋'),
                           ('blenders', 'ミキサー'), ('plates', '皿置き場'),
                           ('cups', 'コップ置き場'), ('deliveries', '提供口')):
            print(f'    {label:8s} {res.get(key)}')
    print()

    orders = ai._build_order_tasks(dcopy(state))
    print('タスクごとの実行可能な担当者:')
    print(f"  {'タスク':38s} {'AI':>4s} {'人':>4s}")
    for o in orders:
        for t in o['tasks']:
            allowed = ai._task_allowed_agents(state, t)
            verb, obj, uid = t['id']
            name = f'{verb}:{obj} (注文{uid})'
            print(f"  {name:38s} {'○' if 0 in allowed else '×':>4s}"
                  f" {'○' if 1 in allowed else '×':>4s}")


if __name__ == '__main__':
    main()
