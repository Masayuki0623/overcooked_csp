"""run_instruction_experiment.py の結果を図にする。

    python tools/plot_instruction_experiment.py --csv results/instruction_experiment.csv

マップを複数含む CSV なら、すべての図でマップを横に並べて比較できる形にする
(段階Aは ring のみなので1列だけになる)。
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# 日本語ラベルが豆腐にならないよう、入っていれば日本語フォントを使う。
for _f in ('Yu Gothic', 'Meiryo', 'MS Gothic', 'Noto Sans CJK JP'):
    if any(_f == f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = _f
        break
plt.rcParams['axes.unicode_minus'] = False

QUALITIES = ('good', 'bad', 'random')
QUALITY_JA = {'good': '良い指示', 'bad': '悪い指示', 'random': 'ランダム'}
COLORS = {'good': '#2f7fd4', 'bad': '#d4552f', 'random': '#8a8f98'}
SKIP_BUDGETS = (0, 1, 2)


def load(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            for k in ('seed', 'skip_budget', 'natural_rank', 'excess',
                      'inserted_count', 'constrained_rank'):
                r[k] = int(r[k]) if r.get(k) not in (None, '') else None
            for k in ('loss_s', 'makespan_baseline_s', 'makespan_constrained_s',
                      'inserted_total_s', 'baseline_solve_s', 'constrained_solve_s'):
                r[k] = float(r[k]) if r.get(k) not in (None, '') else None
            rows.append(r)
    return rows


def maps_of(rows):
    seen = []
    for r in rows:
        if r['map'] not in seen:
            seen.append(r['map'])
    return seen


def grid(maps, title, height=4.2):
    fig, axes = plt.subplots(1, len(maps), figsize=(6.2 * len(maps), height),
                             squeeze=False)
    fig.suptitle(title, fontsize=13, fontweight='bold')
    return fig, axes[0]


def save(fig, out_dir, name):
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = Path(out_dir) / name
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f'  {path}')
    return path


def fig1_natural_rank(rows, maps, out):
    """図1: 指示の質ごとの自然順位の分布。"""
    fig, axes = grid(maps, '図1: 対象タスクの自然順位の分布（指示の質別）')
    ok = [r for r in rows if r['status'] == 'ok' and r['natural_rank'] is not None]
    if ok:
        bins = np.arange(-0.5, max(r['natural_rank'] for r in ok) + 1.5)
    else:
        bins = np.arange(-0.5, 5.5)
    for ax, m in zip(axes, maps):
        for q in QUALITIES:
            vals = [r['natural_rank'] for r in ok if r['map'] == m and r['quality'] == q]
            ax.hist(vals, bins=bins, alpha=0.55, label=f'{QUALITY_JA[q]} (n={len(vals)})',
                    color=COLORS[q])
        ax.set_title(f'マップ: {m}')
        ax.set_xlabel('自然順位（指示なしの最適計画での実行順、0始まり）')
        ax.set_ylabel('試行数')
        ax.legend()
        ax.grid(alpha=0.25)
    return save(fig, out, 'fig1_natural_rank.png')


def fig2_loss_vs_excess(rows, maps, out):
    """図2: 効率損失 vs 超過分。H2 の核心。"""
    fig, axes = grid(maps, '図2: 効率損失 L と超過分（自然順位 − skip_budget）の関係')
    ok = [r for r in rows if r['status'] == 'ok' and r['excess'] is not None]
    for ax, m in zip(axes, maps):
        sub = [r for r in ok if r['map'] == m]
        for q in QUALITIES:
            pts = [(r['excess'], r['loss_s']) for r in sub if r['quality'] == q]
            if pts:
                x, y = zip(*pts)
                # 同じ座標が重なるので、わずかに散らして密度を見せる
                jitter = (np.random.default_rng(0).random(len(x)) - 0.5) * 0.22
                ax.scatter(np.array(x) + jitter, y, s=26, alpha=0.6,
                           color=COLORS[q], label=QUALITY_JA[q])
        # 超過分ごとの平均を折れ線で重ねる
        by_excess = defaultdict(list)
        for r in sub:
            by_excess[r['excess']].append(r['loss_s'])
        if by_excess:
            xs = sorted(by_excess)
            ax.plot(xs, [np.mean(by_excess[x]) for x in xs], color='#222',
                    lw=1.8, marker='o', ms=4, label='超過分ごとの平均')
        ax.axhline(0, color='#999', lw=1, ls='--')
        ax.set_title(f'マップ: {m}')
        ax.set_xlabel('超過分 = max(0, 自然順位 − skip_budget)')
        ax.set_ylabel('効率損失 L（秒）')
        ax.legend()
        ax.grid(alpha=0.25)
    return save(fig, out, 'fig2_loss_vs_excess.png')


def fig3_loss_by_budget(rows, maps, out):
    """図3: skip_budget 別・指示の質別の平均損失。"""
    fig, axes = grid(maps, '図3: skip_budget と効率損失（指示の質別・平均±標準誤差）')
    ok = [r for r in rows if r['status'] == 'ok']
    for ax, m in zip(axes, maps):
        width = 0.26
        for i, q in enumerate(QUALITIES):
            means, errs = [], []
            for d in SKIP_BUDGETS:
                vals = [r['loss_s'] for r in ok
                        if r['map'] == m and r['quality'] == q and r['skip_budget'] == d]
                means.append(np.mean(vals) if vals else 0)
                errs.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
            xs = np.arange(len(SKIP_BUDGETS)) + (i - 1) * width
            ax.bar(xs, means, width, yerr=errs, capsize=3,
                   color=COLORS[q], label=QUALITY_JA[q], alpha=0.9)
        ax.set_xticks(np.arange(len(SKIP_BUDGETS)))
        ax.set_xticklabels([f'd={d}' for d in SKIP_BUDGETS])
        ax.set_title(f'マップ: {m}')
        ax.set_xlabel('skip_budget')
        ax.set_ylabel('効率損失 L の平均（秒）')
        ax.legend()
        ax.grid(alpha=0.25, axis='y')
    return save(fig, out, 'fig3_loss_by_budget.png')


def fig4_inserted(rows, maps, out):
    """図4: 実際に前へ挟まった他タスクの個数と、その合計所要時間。"""
    fig, axes = plt.subplots(2, len(maps), figsize=(6.2 * len(maps), 7.4), squeeze=False)
    fig.suptitle('図4: 対象タスクの前に実際に挟まった他タスク（個数と合計時間）',
                 fontsize=13, fontweight='bold')
    ok = [r for r in rows if r['status'] == 'ok' and r['inserted_count'] is not None]
    for col, m in enumerate(maps):
        sub = [r for r in ok if r['map'] == m]
        ax = axes[0][col]
        data = [[r['inserted_count'] for r in sub if r['skip_budget'] == d]
                for d in SKIP_BUDGETS]
        ax.boxplot(data, tick_labels=[f'd={d}' for d in SKIP_BUDGETS])
        ax.set_title(f'マップ: {m} — 挟まった個数')
        ax.set_ylabel('個数')
        ax.grid(alpha=0.25, axis='y')

        ax = axes[1][col]
        data = [[r['inserted_total_s'] for r in sub
                 if r['skip_budget'] == d and r['inserted_total_s'] is not None]
                for d in SKIP_BUDGETS]
        ax.boxplot(data, tick_labels=[f'd={d}' for d in SKIP_BUDGETS])
        ax.set_title(f'マップ: {m} — 合計所要時間')
        ax.set_xlabel('skip_budget')
        ax.set_ylabel('秒')
        ax.grid(alpha=0.25, axis='y')
    return save(fig, out, 'fig4_inserted_tasks.png')


def fig5_infeasible(rows, maps, out):
    """図5: INFEASIBLE の発生率。"""
    fig, axes = grid(maps, '図5: solve が INFEASIBLE になった割合')
    for ax, m in zip(axes, maps):
        width = 0.26
        for i, q in enumerate(QUALITIES):
            rates = []
            for d in SKIP_BUDGETS:
                sub = [r for r in rows
                       if r['map'] == m and r['quality'] == q and r['skip_budget'] == d]
                bad = sum(1 for r in sub if r.get('constrained_status') == 'INFEASIBLE')
                rates.append(100 * bad / len(sub) if sub else 0)
            xs = np.arange(len(SKIP_BUDGETS)) + (i - 1) * width
            ax.bar(xs, rates, width, color=COLORS[q], label=QUALITY_JA[q], alpha=0.9)
        ax.set_xticks(np.arange(len(SKIP_BUDGETS)))
        ax.set_xticklabels([f'd={d}' for d in SKIP_BUDGETS])
        ax.set_ylim(0, 100)
        ax.set_title(f'マップ: {m}')
        ax.set_xlabel('skip_budget')
        ax.set_ylabel('INFEASIBLE の割合（%）')
        ax.legend()
        ax.grid(alpha=0.25, axis='y')
    return save(fig, out, 'fig5_infeasible_rate.png')


def summarize(rows, maps):
    ok = [r for r in rows if r['status'] == 'ok']
    print(f'\n--- 集計 ---')
    print(f'総試行 {len(rows)} / 正常 {len(ok)}')
    for m in maps:
        print(f'\n[{m}]')
        for q in QUALITIES:
            ranks = [r['natural_rank'] for r in ok if r['map'] == m and r['quality'] == q]
            if ranks:
                print(f'  {QUALITY_JA[q]:<6} 自然順位 中央値={np.median(ranks):.1f} '
                      f'平均={np.mean(ranks):.2f} 範囲={min(ranks)}-{max(ranks)}')
        print('  L の平均（秒）')
        for q in QUALITIES:
            line = []
            for d in SKIP_BUDGETS:
                vals = [r['loss_s'] for r in ok
                        if r['map'] == m and r['quality'] == q and r['skip_budget'] == d]
                line.append(f'd={d}: {np.mean(vals):5.2f}' if vals else f'd={d}:   n/a')
            print(f'    {QUALITY_JA[q]:<6} ' + '  '.join(line))
        zero = [r['loss_s'] for r in ok if r['map'] == m and r['excess'] == 0]
        pos = [r['loss_s'] for r in ok if r['map'] == m and r['excess'] and r['excess'] > 0]
        print(f'  超過分=0 のとき L: 平均={np.mean(zero):.2f} '
              f'最大={max(zero):.2f} (n={len(zero)})' if zero else '  超過分=0 のデータなし')
        print(f'  超過分>0 のとき L: 平均={np.mean(pos):.2f} '
              f'最大={max(pos):.2f} (n={len(pos)})' if pos else '  超過分>0 のデータなし')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='results/instruction_experiment.csv')
    ap.add_argument('--out', default='results/figures')
    args = ap.parse_args()

    rows = load(args.csv)
    maps = maps_of(rows)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    print(f'{len(rows)} 行 / マップ {maps}\n図を出力:')
    fig1_natural_rank(rows, maps, args.out)
    fig2_loss_vs_excess(rows, maps, args.out)
    fig3_loss_by_budget(rows, maps, args.out)
    fig4_inserted(rows, maps, args.out)
    fig5_infeasible(rows, maps, args.out)
    summarize(rows, maps)


if __name__ == '__main__':
    main()
