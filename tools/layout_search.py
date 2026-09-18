"""仕切りのある地図で、器具と材料の置き場所を総当たりし、仕事量の偏りを比べる。

地図の形(13x11、真ん中の列が仕切り、仕切りの台は両側から使える)は
今の実験の地図と同じにして、「どの器具・材料をどちら側に置くか」だけを
変える。左が AI(0番)、右が人間(1番)。

  まな板       : 左 / 右 / 両方
  鍋・ミキサー : 左 / 右
  提供口       : 左 / 右
  材料 6 種    : それぞれ 左 / 右
  皿・コップ   : 両側に置く

それぞれを 12 の注文構成で CSP に計画させ(二人とも計画どおりに動く前提)、
計画上の仕事量(担当した工程の所要時間の合計)の割合と、所要時間を出す。
計画が全部の注文を提供まで運べない配置は除く。

    python tools/layout_search.py --shard 0/16 --out results/layouts_0.csv
    python tools/layout_search.py --show L-R-R-L-...   # 1つの配置を地図で表示
"""
import argparse
import csv
import itertools
import os
import sys
from copy import deepcopy as dcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in ('.', 'agent', 'testbed-cooking', 'tools'):
    sys.path.insert(0, str(ROOT / _p))

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

from gym_cooking.utils.order_preset import (  # noqa: E402
    enumerate_order_recipes, experiment_case_indices)
import run_human_model_experiment as H  # noqa: E402
from workload import make_env  # noqa: E402

LEVEL_DIR = ROOT / 'testbed-cooking' / 'gym_cooking' / 'utils' / 'levels'
SEARCH_SUBDIR = '_search'

W, HGT = 13, 11
DIVIDER_X = 6
INGREDIENTS = [('lettuce', 'L'), ('onion', 'O'), ('tomato', 'T'),
               ('apple', 'A'), ('orange', 'R'), ('banana', 'N')]
TERMINAL_VERBS = ('serve', 'serve_salad', 'serve_juice', 'serve_from_counter')


def layout_key(cfg):
    parts = [cfg['cutboard'], cfg['pot'], cfg['blender'], cfg['delivery']]
    parts += [cfg[name] for name, _ in INGREDIENTS]
    return '-'.join(parts)


def parse_key(key):
    p = key.split('-')
    cfg = {'cutboard': p[0], 'pot': p[1], 'blender': p[2], 'delivery': p[3]}
    for (name, _), side in zip(INGREDIENTS, p[4:]):
        cfg[name] = side
    return cfg


def all_layouts():
    for cut, pot, blend, deliv in itertools.product('LRB', 'LR', 'LR', 'LR'):
        for sides in itertools.product('LR', repeat=len(INGREDIENTS)):
            cfg = {'cutboard': cut, 'pot': pot, 'blender': blend, 'delivery': deliv}
            for (name, _), s in zip(INGREDIENTS, sides):
                cfg[name] = s
            yield cfg


def render(cfg, gap=None):
    """配置から地図の文字列を作る。gap=行番号 で仕切りにすき間を開ける。"""
    grid = [[' '] * W for _ in range(HGT)]
    for x in range(W):
        grid[0][x] = '-'
        grid[HGT - 1][x] = '-'
    for y in range(HGT):
        grid[y][0] = '-'
        grid[y][W - 1] = '-'
        grid[y][DIVIDER_X] = '-'
    if gap is not None:
        grid[gap][DIVIDER_X] = ' '

    def side_has(side, item):
        v = cfg[item]
        return v == side or v == 'B'

    for side, wall_x, top_cols, deliv_cols in (
            ('L', 0, range(1, DIVIDER_X), (1, 2, 3)),
            ('R', W - 1, range(DIVIDER_X + 1, W - 1), (9, 10, 11))):
        stations = [ch for name, ch in INGREDIENTS if cfg[name] == side]
        if side_has(side, 'cutboard'):
            stations.append('/')
        if cfg['blender'] == side:
            stations.append('M')
        stations += ['C', 'P']            # コップと皿は両側に置く
        if cfg['pot'] == side:
            stations.append('U')
        slots = [(wall_x, y) for y in range(1, HGT - 1)]
        slots += [(x, 0) for x in top_cols]
        if len(stations) > len(slots):
            return None
        for ch, (x, y) in zip(stations, slots):
            grid[y][x] = ch
        if cfg['delivery'] == side:
            for x in deliv_cols:
                grid[HGT - 1][x] = '*'
    lines = [''.join(r) for r in grid]
    # 料理の一覧(env は注文を別に受け取るので飾り)と、二人の出発点
    return '\n'.join(lines) + '\n\nTomatoLettuceSoup\n\n1 1\n7 1\n'


def write_level(name, text):
    d = LEVEL_DIR / SEARCH_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    (d / f'{name}.txt').write_text(text, encoding='utf-8')
    return f'{SEARCH_SUBDIR}/{name}'


def plan_metrics(level, recipes):
    """その地図・その注文で、二人とも計画どおり動くとしたときの計画。"""
    env = make_env(level, recipes)
    ai = H.make_ai(0, partner_is_external=False)
    state = H.state_for(env, 0)
    ai(dcopy(state))
    sched = ai.schedule_per_agent or {}
    metrics = getattr(ai, '_last_solve_metrics', {}) or {}
    tasks = {0: sched.get(0, []), 1: sched.get(1, [])}
    work = [sum(t['end'] - t['start'] for t in tasks[a]) for a in (0, 1)]
    # 全部の注文が提供まで計画されているか
    terminal_orders = {t['id'][2] for a in (0, 1) for t in tasks[a]
                       if t['id'][0] in TERMINAL_VERBS}
    n_orders = len(env.order_scheduler.current_orders)
    cross = sum(1 for a in (0, 1) for t in tasks[a] if t['id'][0] in ('carry', 'handover'))
    # 二人ともが手を入れる注文の数。仕事量が半々でも、各自が別々の料理を
    # 一人で作っているだけなら協力にはならない。
    by_order = {}
    for a in (0, 1):
        for t in tasks[a]:
            by_order.setdefault(t['id'][2], set()).add(a)
    shared = sum(1 for agents in by_order.values() if agents == {0, 1})
    return {
        'shared_orders': shared,
        'ok': metrics.get('status') == 'OPTIMAL' and len(terminal_orders) >= n_orders,
        'makespan': (metrics.get('makespan_frames') or 0) / 10.0,
        'ai_work': work[0] / 10.0,
        'human_work': work[1] / 10.0,
        'ai_tasks': len(tasks[0]),
        'human_tasks': len(tasks[1]),
        'cross': cross,
    }


FIELDS = ['layout', 'feasible_cases', 'makespan', 'ai_work', 'human_work',
          'ai_share', 'ai_tasks', 'human_tasks', 'cross', 'shared_orders']


def evaluate(cfg, cases, sets, gap=None):
    key = layout_key(cfg) + (f'-gap{gap}' if gap is not None else '')
    text = render(cfg, gap=gap)
    if text is None:
        return None
    level = write_level(key, text)
    rows = []
    for c in cases:
        try:
            rows.append(plan_metrics(level, sets[c]))
        except Exception:
            rows.append({'ok': False})
    good = [r for r in rows if r.get('ok')]
    out = {'layout': key, 'feasible_cases': len(good)}
    if len(good) == len(rows):
        n = len(good)
        ai = sum(r['ai_work'] for r in good)
        hu = sum(r['human_work'] for r in good)
        out.update(
            makespan=round(sum(r['makespan'] for r in good) / n, 2),
            ai_work=round(ai / n, 2), human_work=round(hu / n, 2),
            ai_share=round(ai / max(ai + hu, 1e-9), 3),
            ai_tasks=round(sum(r['ai_tasks'] for r in good) / n, 2),
            human_tasks=round(sum(r['human_tasks'] for r in good) / n, 2),
            cross=round(sum(r['cross'] for r in good) / n, 2),
            shared_orders=round(sum(r['shared_orders'] for r in good) / n, 2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', default=None, help='"i/n"')
    ap.add_argument('--out', default=None)
    ap.add_argument('--show', default=None, help='配置の名前を渡すと地図を表示する')
    ap.add_argument('--only', default=None, help='配置の名前を , 区切りで')
    ap.add_argument('--gap', type=int, default=None,
                    help='仕切りのこの行を1マス開ける(行き来できる地図と比べる)')
    ap.add_argument('--cases', default=None,
                    help='注文構成の番号を , 区切りで(絞り込みを速くするとき)')
    args = ap.parse_args()

    if args.show:
        print(render(parse_key(args.show), gap=args.gap))
        return

    H.MAX_SECONDS_OVERRIDE = 100.0
    sets = enumerate_order_recipes('experiment2')
    cases = experiment_case_indices('experiment2')
    if args.cases:
        cases = [int(x) for x in args.cases.split(',')]
    layouts = ([parse_key(k) for k in args.only.split(',')] if args.only
               else list(all_layouts()))
    if args.shard:
        i, n = (int(x) for x in args.shard.split('/'))
        layouts = [c for k, c in enumerate(layouts) if k % n == i]

    rows = []
    for k, cfg in enumerate(layouts, 1):
        r = evaluate(cfg, cases, sets, gap=args.gap)
        if r is not None:
            rows.append(r)
            if args.only:
                print(r, flush=True)
        if k % 10 == 0:
            print('  %d/%d' % (k, len(layouts)), flush=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)


if __name__ == '__main__':
    main()
