"""run_human_model_experiment.py の結果を図にする。

    python tools/plot_human_model_experiment.py --csv results/human_model_experiment.csv
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

for _f in ('Yu Gothic', 'Meiryo', 'MS Gothic', 'Noto Sans CJK JP'):
    if any(_f == f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = _f
        break
plt.rcParams['axes.unicode_minus'] = False

MODELS = ('follow_plan', 'greedy', 'random')
MODEL_JA = {'follow_plan': '計画どおり(上限)', 'greedy': '貪欲', 'random': 'ランダム'}
MODEL_COLOR = {'follow_plan': '#2f7fd4', 'greedy': '#d4552f', 'random': '#8a8f98'}
QUALITIES = ('good', 'bad', 'random')
QUALITY_JA = {'good': '良い指示', 'bad': '悪い指示', 'random': 'ランダム'}
QUALITY_COLOR = {'good': '#2f7fd4', 'bad': '#d4552f', 'random': '#8a8f98'}
BUDGETS = (0, 2, 4)

INT_COLS = ('seed', 'skip_budget', 'served', 'orders_left', 'completed',
            'timed_out', 'prediction_samples')
FLOAT_COLS = ('loss_s', 'makespan_actual_s', 'makespan_plan_baseline_s',
              'makespan_plan_constrained_s', 'prediction_match', 'wall_seconds')


def load(paths):
    rows = []
    for path in paths:
        with open(path, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                for k in INT_COLS:
                    r[k] = int(r[k]) if r.get(k) not in (None, '') else None
                for k in FLOAT_COLS:
                    r[k] = float(r[k]) if r.get(k) not in (None, '') else None
                rows.append(r)
    return rows


def save(fig, out_dir, name):
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = Path(out_dir) / name
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f'  {path}')


def mean_se(values):
    if not values:
        return 0.0, 0.0
    m = float(np.mean(values))
    se = float(np.std(values) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return m, se


def figA(rows, out):
    """図A: 人間役の行動モデル別・skip_budget別の効率。

    最も知りたいのは「非最適な人間のとき、skip_budget が大きいほど
    効率が悪化する(右肩上がりになる)か」なので、計画上の損失 L と
    実測 makespan を並べて出す。
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    fig.suptitle('図A: 人間役の行動モデル別・skip_budget 別の効率',
                 fontsize=13, fontweight='bold')

    for ax, key, label in (
            (axes[0], 'loss_s', '計画上の効率損失 L（秒）'),
            (axes[1], 'makespan_actual_s', '実測 makespan（秒）')):
        for m in MODELS:
            ys, es = [], []
            for d in BUDGETS:
                vals = [r[key] for r in rows
                        if r['human_model'] == m and r['skip_budget'] == d
                        and r[key] is not None]
                mu, se = mean_se(vals)
                ys.append(mu)
                es.append(se)
            ax.errorbar(BUDGETS, ys, yerr=es, marker='o', capsize=4, lw=2,
                        color=MODEL_COLOR[m], label=MODEL_JA[m])
        ax.set_xticks(BUDGETS)
        ax.set_xlabel('skip_budget')
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        ax.legend()
    save(fig, out, 'figA_efficiency_by_human_model.png')


def figB(rows, out):
    """図B: AIの予測がどれだけ外れたか。"""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    fig.suptitle('図B: AIの予測と、人間役の実際の行動のズレ',
                 fontsize=13, fontweight='bold')

    ax = axes[0]
    for m in MODELS:
        ys, es = [], []
        for d in BUDGETS:
            vals = [100 * r['prediction_match'] for r in rows
                    if r['human_model'] == m and r['skip_budget'] == d
                    and r['prediction_match'] is not None]
            mu, se = mean_se(vals)
            ys.append(mu)
            es.append(se)
        ax.errorbar(BUDGETS, ys, yerr=es, marker='o', capsize=4, lw=2,
                    color=MODEL_COLOR[m], label=MODEL_JA[m])
    ax.set_xticks(BUDGETS)
    ax.set_ylim(0, 100)
    ax.set_xlabel('skip_budget')
    ax.set_ylabel('予測の一致率（%）')
    ax.set_title('予測が当たった割合（高いほどズレが小さい）')
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1]
    data = [[100 * r['prediction_match'] for r in rows
             if r['human_model'] == m and r['prediction_match'] is not None]
            for m in MODELS]
    ax.boxplot(data, tick_labels=[MODEL_JA[m] for m in MODELS])
    ax.set_ylabel('予測の一致率（%）')
    ax.set_title('モデル別の分布')
    ax.grid(alpha=0.25, axis='y')
    save(fig, out, 'figB_prediction_mismatch.png')


def figC(rows, out):
    """図C: skip_budget × 指示の質 を、人間役のモデルごとに並べる。"""
    fig, axes = plt.subplots(1, len(MODELS), figsize=(5.6 * len(MODELS), 4.6),
                             squeeze=False, sharey=True)
    fig.suptitle('図C: skip_budget と指示の質（人間役の行動モデル別・実測 makespan）',
                 fontsize=13, fontweight='bold')
    width = 0.26
    for col, m in enumerate(MODELS):
        ax = axes[0][col]
        for i, q in enumerate(QUALITIES):
            ys, es = [], []
            for d in BUDGETS:
                vals = [r['makespan_actual_s'] for r in rows
                        if r['human_model'] == m and r['skip_budget'] == d
                        and r['quality'] == q and r['makespan_actual_s'] is not None]
                mu, se = mean_se(vals)
                ys.append(mu)
                es.append(se)
            xs = np.arange(len(BUDGETS)) + (i - 1) * width
            ax.bar(xs, ys, width, yerr=es, capsize=3,
                   color=QUALITY_COLOR[q], label=QUALITY_JA[q], alpha=0.9)
        ax.set_xticks(np.arange(len(BUDGETS)))
        ax.set_xticklabels([f'd={d}' for d in BUDGETS])
        ax.set_title(f'人間役: {MODEL_JA[m]}')
        ax.set_xlabel('skip_budget')
        if col == 0:
            ax.set_ylabel('実測 makespan（秒）')
            ax.legend()
        ax.grid(alpha=0.25, axis='y')
    save(fig, out, 'figC_quality_by_human_model.png')


def summarize(rows):
    print('\n--- 集計 ---')
    print(f'総試行 {len(rows)}')
    for m in MODELS:
        sub = [r for r in rows if r['human_model'] == m]
        if not sub:
            continue
        done = sum(1 for r in sub if r['completed'])
        served = [r['served'] for r in sub if r['served'] is not None]
        match = [100 * r['prediction_match'] for r in sub
                 if r['prediction_match'] is not None]
        print(f'\n[{MODEL_JA[m]}] n={len(sub)}')
        print(f'  全注文を提供できた: {done}/{len(sub)} ({100*done/len(sub):.0f}%)  '
              f'提供数の平均={np.mean(served):.2f}')
        if match:
            print(f'  予測の一致率: 平均={np.mean(match):.1f}% 中央値={np.median(match):.1f}%')
        for key, label in (('loss_s', '計画上のL'), ('makespan_actual_s', '実測makespan')):
            line = []
            for d in BUDGETS:
                vals = [r[key] for r in sub if r['skip_budget'] == d and r[key] is not None]
                line.append(f'd={d}: {np.mean(vals):6.2f}' if vals else f'd={d}:  n/a')
            print(f'  {label:<12} ' + '  '.join(line))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='results/human_model_experiment.csv',
                    help='カンマ区切りで複数指定可(分割実行の結果をまとめる)')
    ap.add_argument('--out', default='results/figures')
    args = ap.parse_args()

    paths = [p.strip() for p in args.csv.split(',')]
    rows = [r for r in load(paths) if r.get('status') == 'ok']
    Path(args.out).mkdir(parents=True, exist_ok=True)
    print(f'{len(rows)} 行\n図を出力:')
    figA(rows, args.out)
    figB(rows, args.out)
    figC(rows, args.out)
    summarize(rows)


if __name__ == '__main__':
    main()
