"""run_instruction_wait_experiment.py の結果を図にする。

    python tools/plot_instruction_wait.py --csv results/instruction_wait.csv
"""
import argparse
import csv
from collections import Counter, defaultdict
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

BUDGETS = (0, 2, 4)
QUALITIES = ('good', 'bad', 'random')
QUALITY_JA = {'good': '良い指示', 'bad': '悪い指示', 'random': 'ランダムな指示'}
QUALITY_COLOR = {'good': '#2f7fd4', 'bad': '#d4552f', 'random': '#8a8f98'}
VERB_JA = {'chop': '刻む', 'cook': '煮る', 'mix': '混ぜる', 'serve': '提供',
           'serve_salad': 'サラダ提供', 'serve_juice': 'ジュース提供',
           'handover': '受け渡し', 'serve_from_counter': '受け取り提供'}
VERB_COLOR = {'chop': '#4c9be8', 'cook': '#e8734c', 'mix': '#7bc47f',
              'serve': '#c9a227', 'serve_salad': '#c98fd6', 'serve_juice': '#9fd6c9',
              'handover': '#d66f8f', 'serve_from_counter': '#8a8f98'}


def load(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r.get('status') != 'ok':
                continue
            for k in ('skip_budget', 'tasks_before', 'exec_rank', 'natural_rank',
                      'rank_gain', 'wait_censored', 'done_by_other'):
                r[k] = int(r[k]) if r.get(k) not in (None, '') else None
            for k in ('wait_seconds', 'wait_any_seconds', 'ai_idle_pct_while_waiting',
                      'ai_idle_seconds_while_waiting'):
                r[k] = float(r[k]) if r.get(k) not in (None, '') else None
            rows.append(r)
    return rows


def save(fig, out_dir, name):
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = Path(out_dir) / name
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f'  {path}')


def waits(rows, quality, budget, only_started=True):
    """待ち時間の一覧。only_started=True なら AI が着手した試行だけ。"""
    out = []
    for r in rows:
        if r['quality'] != quality or r['skip_budget'] != budget:
            continue
        if only_started and r['wait_censored']:
            continue
        if r['wait_seconds'] is not None:
            out.append(r['wait_seconds'])
    return out


def fig1(rows, out):
    """図1: 待たされる時間の分布。今回の主図。"""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    fig.suptitle('図1: 指示を出してから AI がその作業に着手するまでの時間',
                 fontsize=13, fontweight='bold')

    ax = axes[0]
    width = 0.24
    for qi, q in enumerate(QUALITIES):
        data = [waits(rows, q, d) for d in BUDGETS]
        pos = [i + (qi - 1) * width for i in range(len(BUDGETS))]
        bp = ax.boxplot([d if d else [0] for d in data], positions=pos, widths=width * 0.85,
                        patch_artist=True, showmeans=True, manage_ticks=False)
        for box in bp['boxes']:
            box.set_facecolor(QUALITY_COLOR[q])
            box.set_alpha(0.55)
        for key in ('medians', 'whiskers', 'caps'):
            for line in bp[key]:
                line.set_color('#333')
        ax.plot([], [], color=QUALITY_COLOR[q], lw=8, alpha=0.55, label=QUALITY_JA[q])
    ax.set_xticks(range(len(BUDGETS)))
    ax.set_xticklabels([str(d) for d in BUDGETS])
    ax.set_xlabel('skip_budget')
    ax.set_ylabel('待ち時間（秒）')
    ax.set_title('着手できた試行の分布', fontsize=10)
    ax.grid(axis='y', alpha=0.25)
    ax.legend()

    # 右: 平均と、AI が最後まで着手しなかった割合
    ax = axes[1]
    for q in QUALITIES:
        ys, es, cens = [], [], []
        for d in BUDGETS:
            vals = waits(rows, q, d)
            ys.append(float(np.mean(vals)) if vals else 0.0)
            es.append(float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)
            grp = [r for r in rows if r['quality'] == q and r['skip_budget'] == d]
            cens.append(100 * sum(r['wait_censored'] for r in grp) / len(grp) if grp else 0)
        ax.errorbar(BUDGETS, ys, yerr=es, marker='o', capsize=4, lw=2,
                    color=QUALITY_COLOR[q], label=f'{QUALITY_JA[q]}（平均）')
        ax.plot(BUDGETS, cens, marker='s', ls='--', lw=1.4, alpha=0.7,
                color=QUALITY_COLOR[q], label=f'{QUALITY_JA[q]}（着手されず %）')
    ax.set_xticks(BUDGETS)
    ax.set_xlabel('skip_budget')
    ax.set_ylabel('待ち時間の平均（秒） / 着手されなかった割合（%）')
    ax.set_title('実線=平均待ち時間、破線=AI が最後まで着手しなかった割合', fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    save(fig, out, 'fig1_wait_time.png')


def fig2(rows, out):
    """図2: 待っている間に AI がこなした他タスクの個数と内訳。"""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    fig.suptitle('図2: 指示された作業に着手するまでに、AI がこなした他の作業',
                 fontsize=13, fontweight='bold')

    ax = axes[0]
    width = 0.24
    for qi, q in enumerate(QUALITIES):
        data = []
        for d in BUDGETS:
            vals = [r['tasks_before'] for r in rows
                    if r['quality'] == q and r['skip_budget'] == d
                    and not r['wait_censored']]
            data.append(vals if vals else [0])
        pos = [i + (qi - 1) * width for i in range(len(BUDGETS))]
        bp = ax.boxplot(data, positions=pos, widths=width * 0.85,
                        patch_artist=True, showmeans=True, manage_ticks=False)
        for box in bp['boxes']:
            box.set_facecolor(QUALITY_COLOR[q])
            box.set_alpha(0.55)
        for key in ('medians', 'whiskers', 'caps'):
            for line in bp[key]:
                line.set_color('#333')
        ax.plot([], [], color=QUALITY_COLOR[q], lw=8, alpha=0.55, label=QUALITY_JA[q])
    ax.set_xticks(range(len(BUDGETS)))
    ax.set_xticklabels([str(d) for d in BUDGETS])
    ax.set_xlabel('skip_budget')
    ax.set_ylabel('先にこなされた作業の個数')
    ax.set_title('個数の分布', fontsize=10)
    ax.grid(axis='y', alpha=0.25)
    ax.legend()

    # 右: 種別の内訳(積み上げ)
    ax = axes[1]
    labels, stacks = [], []
    for d in BUDGETS:
        for q in QUALITIES:
            c = Counter()
            for r in rows:
                if r['quality'] != q or r['skip_budget'] != d or r['wait_censored']:
                    continue
                for v in (r.get('verbs_before') or '').split('|'):
                    if v:
                        c[v] += 1
            labels.append(f'd={d}\n{QUALITY_JA[q]}')
            stacks.append(c)
    verbs = sorted({v for c in stacks for v in c}, key=lambda v: -sum(c[v] for c in stacks))
    bottom = np.zeros(len(stacks))
    x = np.arange(len(stacks))
    for v in verbs:
        vals = np.array([c.get(v, 0) for c in stacks], dtype=float)
        ax.bar(x, vals, bottom=bottom, color=VERB_COLOR.get(v, '#bbb'),
               label=VERB_JA.get(v, v), width=0.7)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('のべ件数')
    ax.set_title('先にこなされた作業の種別', fontsize=10)
    ax.grid(axis='y', alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    save(fig, out, 'fig2_tasks_before.png')


def fig3(rows, out):
    """図3: 指示された作業の実行順位と、指示が無かった場合の自然順位。"""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    fig.suptitle('図3: 指示された作業は、AI の作業列で何番目に実行されたか',
                 fontsize=13, fontweight='bold')

    ax = axes[0]
    x = np.arange(len(BUDGETS))
    width = 0.35
    nat, act, nat_e, act_e = [], [], [], []
    for d in BUDGETS:
        grp = [r for r in rows if r['skip_budget'] == d and not r['wait_censored']
               and r['natural_rank'] is not None and r['exec_rank'] is not None]
        n = [r['natural_rank'] for r in grp]
        a = [r['exec_rank'] for r in grp]
        nat.append(np.mean(n) if n else 0)
        act.append(np.mean(a) if a else 0)
        nat_e.append(np.std(n) / np.sqrt(len(n)) if len(n) > 1 else 0)
        act_e.append(np.std(a) / np.sqrt(len(a)) if len(a) > 1 else 0)
    ax.bar(x - width / 2, nat, width, yerr=nat_e, capsize=4,
           color='#8a8f98', label='指示が無かった場合（自然順位）')
    ax.bar(x + width / 2, act, width, yerr=act_e, capsize=4,
           color='#2f7fd4', label='指示を出した場合（実際の順位）')
    ax.set_xticks(x)
    ax.set_xticklabels([f'd={d}' for d in BUDGETS])
    ax.set_ylabel('作業列の中での順位（小さいほど早い）')
    ax.set_title('平均順位の比較', fontsize=10)
    ax.grid(axis='y', alpha=0.25)
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.axhline(0, color='#444', lw=1)
    for q in QUALITIES:
        ys, es = [], []
        for d in BUDGETS:
            vals = [r['rank_gain'] for r in rows
                    if r['quality'] == q and r['skip_budget'] == d
                    and r['rank_gain'] is not None]
            ys.append(np.mean(vals) if vals else 0)
            es.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
        ax.errorbar(BUDGETS, ys, yerr=es, marker='o', capsize=4, lw=2,
                    color=QUALITY_COLOR[q], label=QUALITY_JA[q])
    ax.set_xticks(BUDGETS)
    ax.set_xlabel('skip_budget')
    ax.set_ylabel('前倒しされた順位（自然順位 − 実際の順位）')
    ax.set_title('正なら指示で前倒しされた、負なら後回しにされた', fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend()
    save(fig, out, 'fig3_exec_rank.png')


def summarize(rows):
    def q(vals, p):
        return float(np.percentile(vals, p)) if vals else float('nan')

    print(f'\n有効試行 {len(rows)} 件\n')
    print('=== 3-1 待たされる時間（AI が着手した試行のみ、秒）===')
    print(f"{'指示の質':10s}{'d':>3s}{'n':>4s}{'平均':>7s}{'中央':>7s}"
          f"{'25%':>7s}{'75%':>7s}{'最大':>7s}{'着手されず':>10s}")
    for qual in QUALITIES:
        for d in BUDGETS:
            grp = [r for r in rows if r['quality'] == qual and r['skip_budget'] == d]
            vals = [r['wait_seconds'] for r in grp if not r['wait_censored']]
            cens = sum(r['wait_censored'] for r in grp)
            print(f'{QUALITY_JA[qual]:10s}{d:>3d}{len(vals):>4d}'
                  f'{np.mean(vals) if vals else float("nan"):>7.1f}'
                  f'{q(vals, 50):>7.1f}{q(vals, 25):>7.1f}{q(vals, 75):>7.1f}'
                  f'{max(vals) if vals else float("nan"):>7.1f}'
                  f'{f"{cens}/{len(grp)}":>10s}')

    print('\n=== 3-2 待っている間の AI ===')
    print(f"{'指示の質':10s}{'d':>3s}{'先行作業数':>10s}{'AI停止%':>9s}{'AI停止秒':>9s}")
    for qual in QUALITIES:
        for d in BUDGETS:
            grp = [r for r in rows if r['quality'] == qual and r['skip_budget'] == d
                   and not r['wait_censored']]
            if not grp:
                continue
            print(f'{QUALITY_JA[qual]:10s}{d:>3d}'
                  f'{np.mean([r["tasks_before"] for r in grp]):>10.1f}'
                  f'{np.mean([r["ai_idle_pct_while_waiting"] or 0 for r in grp]):>9.1f}'
                  f'{np.mean([r["ai_idle_seconds_while_waiting"] or 0 for r in grp]):>9.1f}')

    print('\n=== 3-3 実行順位 ===')
    print(f"{'指示の質':10s}{'d':>3s}{'自然順位':>9s}{'実際の順位':>11s}{'前倒し':>8s}")
    for qual in QUALITIES:
        for d in BUDGETS:
            grp = [r for r in rows if r['quality'] == qual and r['skip_budget'] == d
                   and r['rank_gain'] is not None]
            if not grp:
                continue
            print(f'{QUALITY_JA[qual]:10s}{d:>3d}'
                  f'{np.mean([r["natural_rank"] for r in grp]):>9.1f}'
                  f'{np.mean([r["exec_rank"] for r in grp]):>11.1f}'
                  f'{np.mean([r["rank_gain"] for r in grp]):>+8.1f}')

    print('\n=== 指示された作業を、計画は誰に割り当てたか ===')
    owners = Counter(r.get('plan_owner') or '(不明)' for r in rows)
    for k, v in owners.most_common():
        print(f'  {k:12s} {v:4d} 件 ({100 * v / len(rows):.0f}%)')
    other = sum(r.get('done_by_other') or 0 for r in rows)
    print(f'  AI は着手しなかったが相手がやった: {other} 件 ({100 * other / len(rows):.0f}%)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='results/instruction_wait.csv')
    ap.add_argument('--out', default='results/figures')
    args = ap.parse_args()

    rows = load(args.csv)
    if not rows:
        print('有効な行がありません')
        return
    summarize(rows)
    print('\n図:')
    fig1(rows, args.out)
    fig2(rows, args.out)
    fig3(rows, args.out)


if __name__ == '__main__':
    main()
