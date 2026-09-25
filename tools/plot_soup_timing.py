# -*- coding: utf-8 -*-
"""スープを後回しにすると合計時間がどうなるかを描く。

  results/soup_timeline.json (tools/soup_timing 系で作る) を読んで
  「鍋に火を入れた時刻」と「合計時間」の関係を散布図にする。
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

for _f in ('Yu Gothic', 'Meiryo', 'MS Gothic', 'Noto Sans CJK JP'):
    if any(_f == f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = _f
        break
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path(__file__).resolve().parents[1]


def load():
    d = json.load(open(ROOT / 'results/soup_timeline.json', encoding='utf-8'))
    rows = []
    for r in d:
        ser = [float(x) for x in (r['serve_times'] or '').split('|') if x]
        dishes = (r['serve_dishes'] or '').split('|')
        pos = None
        for i, (dn, _t) in enumerate(zip(dishes, ser)):
            if 'Cooked' in dn:
                pos = i + 1
        others = [t for dn, t in zip(dishes, ser) if 'Cooked' not in dn]
        rows.append(dict(day=r['pid'][:3], pos=pos, cook=r['t_cook'],
                         mk=r['makespan'], other=max(others) if others else 0.0))
    return rows


def main():
    rows = load()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    xs = [0, 42]
    ax.plot(xs, [x + 16 for x in xs], color='#888', ls='--', lw=1,
            label='煮込み開始 + 16秒')
    groups = [
        ('9/24 早めに鍋(1〜2品目で提供)', [r for r in rows if r['day'] == '924' and r['pos'] < 3], '#2a7', 'o'),
        ('9/24 後回し(最後に提供)', [r for r in rows if r['day'] == '924' and r['pos'] == 3], '#e84', 's'),
        ('9/25 わざと後回し', [r for r in rows if r['day'] == '925'], '#c33', '^'),
    ]
    for label, rs, c, m in groups:
        ax.scatter([r['cook'] for r in rs], [r['mk'] for r in rs],
                   c=c, marker=m, s=62, label=f'{label} (n={len(rs)})',
                   edgecolors='white', linewidths=0.8, zorder=3)
    ax.set_xlabel('鍋に火が入った時刻 [秒]')
    ax.set_ylabel('合計時間(3品出し終わるまで) [秒]')
    ax.set_title('後回しにした回は、合計時間が「煮込み開始+16秒」に貼りつく')
    ax.grid(alpha=.3)
    ax.legend(fontsize=8.5, loc='upper left')

    # 右: 最後のスープだけを待っていた時間
    names, vals, cols = [], [], []
    for label, rs, c, _m in groups:
        if not rs:
            continue
        names.append(label.replace(' ', '\n', 1))
        vals.append(sum(max(0.0, r['mk'] - r['other']) for r in rs) / len(rs))
        cols.append(c)
    bars = ax2.bar(names, vals, color=cols, width=.55)
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + .25, '%.1f秒' % v,
                 ha='center', fontsize=10, fontweight='bold')
    ax2.set_ylabel('スープだけを待っていた時間 [秒]')
    ax2.set_title('他の料理が終わったあと、スープ待ちで過ぎた時間')
    ax2.grid(axis='y', alpha=.3)
    ax2.set_ylim(0, max(vals) * 1.3)

    fig.tight_layout()
    out = ROOT / 'results/figures/soup_deferral.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(out)


main()
