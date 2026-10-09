"""指示のうまさ(即時実行の効率損失 L0)で参加者を 2 群に分け、d による体験の変わり方を比べる。

    python tools/analyze_pattern7_by_skill.py
    python tools/analyze_pattern7_by_skill.py --pids p25,p27,...

L0 = 「その指示を、いますぐやらせたら最適な段取りより何秒遅くなるか」(ゲームの始めの盤面で
計算)。小さいほど、段取りに合った指示。人ごとに 4 ゲームの平均を取り、中央で 2 群に分ける
(うまい群 / 下手な群)。

比べ方:
  - 群 × d の平均(図と表)
  - 群の中での d の傾き(Page。人ごとの Spearman の平均 ρ̄、並べ替え p)
  - 群の差(交互作用): 人ごとに「自由の効果」= d=1,2,∞ の平均 − d=0 を出し、
    2 群で平均の差を比べる(4 人 vs 4 人の全 70 通りの並べ替えで正確な p)
  - 補足(ゲーム単位): その回の指示の L0 が大きい(2 秒以上)か小さいかと、d=0 か否か

書くもの: results/experience1/analysis/pattern7/by_skill/
"""
import argparse
import html
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_pattern7 as A  # noqa: E402

OUT = A.OUT / 'by_skill'
L0 = '即時実行の効率損失量L0_秒'
GAME_HIGH_L0 = 2.0      # ゲーム単位の補足で「指示のロスが大きい」とする秒

MEASURES = [m for m in A.MEASURES if m[2] in ('perf', 'survey')
            and m[0] not in ('効率損失量L_秒', '人のミスの回数')]
GROUPS = ['うまい群', '下手な群']
COLORS = {'うまい群': '#1f6f43', '下手な群': '#b4461b'}


def exact_perm_diff(a, b):
    """2 群の平均の差。全部の分け方を数える正確な両側 p。"""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    pool = np.concatenate([a, b])
    obs = abs(a.mean() - b.mean())
    hits = total = 0
    for idx in combinations(range(len(pool)), len(a)):
        m = np.zeros(len(pool), bool)
        m[list(idx)] = True
        total += 1
        hits += abs(pool[m].mean() - pool[~m].mean()) >= obs - 1e-9
    return hits / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pids', default=','.join(A.VALID))
    args = ap.parse_args()
    pids = [p.strip() for p in args.pids.split(',') if p.strip()]
    rng = np.random.default_rng(A.SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    df, _ = A.load(pids)

    # 人ごとの L0 と群分け(中央で 2 つに。同じ値が中央をまたぐときは大きい側へ)
    per = df.groupby('参加者ID')[L0].agg(['mean', 'median', 'max']).sort_values('mean')
    half = len(per) // 2
    per['群'] = ['うまい群'] * half + ['下手な群'] * (len(per) - half)
    per['指示の動作'] = df.sort_values('セッション番号').groupby('参加者ID')['指示の動作'].agg(
        lambda s: ' / '.join(s))
    per.to_csv(OUT / 'groups.csv', encoding='utf-8-sig')
    df['群'] = df['参加者ID'].map(per['群'])

    # 群 × d の平均
    rows = []
    for col, name, _ in MEASURES:
        for g in GROUPS:
            r = {'指標': name, '群': g}
            for lv in A.D_ORDER:
                v = df.loc[(df['群'] == g) & (df['d'] == lv), col].astype(float)
                r[A.D_LABEL[lv]] = v.mean()
            rows.append(r)
    means = pd.DataFrame(rows)
    means.to_csv(OUT / 'means.csv', index=False, encoding='utf-8-sig')

    # 群の中の傾きと、群の差(自由の効果)
    tests = []
    for col, name, _ in MEASURES:
        w = df.pivot_table(index='参加者ID', columns='d', values=col, aggfunc='first').reindex(
            columns=A.D_ORDER)
        effect = w[['1', '2', 'inf']].mean(axis=1) - w['0']
        row = {'指標': name}
        for g in GROUPS:
            ids = per.index[per['群'] == g]
            sub = w.loc[ids].dropna()
            rho, p = A.page_trend(sub.to_numpy(float), rng) if len(sub) >= 3 else (np.nan, np.nan)
            row[f'{g}_傾きρ'] = rho
            row[f'{g}_傾きp'] = p
            row[f'{g}_自由の効果'] = effect.loc[ids].mean()
        a = effect.loc[per.index[per['群'] == '下手な群']]
        b = effect.loc[per.index[per['群'] == 'うまい群']]
        row['群の差(下手−うまい)'] = a.mean() - b.mean()
        row['群の差_p'] = exact_perm_diff(a, b)
        tests.append(row)
    tests = pd.DataFrame(tests)
    tests.to_csv(OUT / 'tests.csv', index=False, encoding='utf-8-sig')

    # 補足: ゲーム単位(その回の指示のロスの大小 × d=0 か)
    df['その回のL0'] = np.where(df[L0] >= GAME_HIGH_L0, f'大({GAME_HIGH_L0:g}秒以上)', f'小({GAME_HIGH_L0:g}秒未満)')
    df['d=0か'] = np.where(df['d'] == '0', 'd=0', 'd=1,2,∞')
    game_rows = []
    for col, name, _ in MEASURES:
        r = {'指標': name}
        for big in sorted(df['その回のL0'].unique()):
            for z in ['d=0', 'd=1,2,∞']:
                v = df.loc[(df['その回のL0'] == big) & (df['d=0か'] == z), col].astype(float)
                r[f'L0{big[0]}・{z}'] = v.mean()
                r[f'L0{big[0]}・{z}_n'] = len(v)
        game_rows.append(r)
    games = pd.DataFrame(game_rows)
    games.to_csv(OUT / 'by_game.csv', index=False, encoding='utf-8-sig')

    # 図: 群ごとの d の線
    plt = A.setup_mpl()
    show = [m for m in MEASURES]
    ncol = 5
    nrow = math.ceil(len(show) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(17, 3.3 * nrow))
    x = np.arange(len(A.D_ORDER))
    for ax, (col, name, grp) in zip(axes.flat, show):
        for g in GROUPS:
            w = df[df['群'] == g].pivot_table(index='参加者ID', columns='d', values=col,
                                              aggfunc='first').reindex(columns=A.D_ORDER)
            mu = w.mean().to_numpy(float)
            se = (w.std(ddof=1) / np.sqrt(w.count())).to_numpy(float)
            off = -0.06 if g == GROUPS[0] else 0.06
            ax.errorbar(x + off, mu, yerr=se, color=COLORS[g], lw=2, marker='o', ms=5,
                        capsize=3, label=g)
        ax.set_xticks(x)
        ax.set_xticklabels([A.D_LABEL[l] for l in A.D_ORDER])
        ax.set_title(name, fontsize=10)
        if grp == 'survey':
            ax.set_ylim(0.6, 5.4)
        ax.grid(axis='y', alpha=0.3)
    for ax in list(axes.flat)[len(show):]:
        ax.axis('off')
    axes.flat[0].legend(fontsize=9)
    fig.tight_layout()
    img = A.fig_to_png(fig, OUT / 'fig_by_skill.png')
    plt.close(fig)

    write_report(per, means, tests, games, img)
    print('群:', {g: list(per.index[per['群'] == g]) for g in GROUPS})
    print('出力:', OUT)


def write_report(per, means, tests, games, img):
    t = A.table
    findings = OUT / 'findings.html'
    fh = findings.read_text(encoding='utf-8') if findings.exists() else ''
    per_show = per.reset_index().rename(columns={'mean': 'L0の平均', 'median': 'L0の中央値', 'max': 'L0の最大'})
    gcols = ['指標'] + [c for c in games.columns if c != '指標' and not c.endswith('_n')]
    doc = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>指示のうまさ別</title>
<style>
:root {{ --bg:#fff; --fg:#1d2330; --muted:#6b7280; --line:#e3e6ec; --accent:#1f3b73; --sig:#b42318; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#14171d; --fg:#e7e9ee; --muted:#9aa1ad; --line:#2a2f39; --accent:#9db7f0; --sig:#ff8a7a; }} img {{ background:#fff; }} }}
body {{ background:var(--bg); color:var(--fg); font-family:'BIZ UDPGothic','Yu Gothic',sans-serif; margin:0 auto; max-width:1100px; padding:16px; line-height:1.6; }}
h1 {{ font-size:22px; }} h2 {{ font-size:18px; border-bottom:2px solid var(--accent); padding-bottom:4px; margin-top:32px; }}
table {{ border-collapse:collapse; font-size:13px; margin:8px 0; display:block; overflow-x:auto; }}
th, td {{ border:1px solid var(--line); padding:4px 8px; text-align:right; white-space:nowrap; }}
td:first-child, th:first-child {{ text-align:left; }}
.sig {{ color:var(--sig); }} .tr {{ text-decoration:underline dotted; }} img {{ max-width:100%; }}
.note {{ color:var(--muted); font-size:13px; }}
.findings {{ border-left:4px solid var(--accent); padding:4px 16px; background:color-mix(in srgb, var(--accent) 6%, transparent); }}
</style></head><body>
<h1>指示のうまさ別に見た、d による体験の変化</h1>
<p class="note">指示のうまさ = 即時実行の効率損失 L0(その指示をすぐやらせたら最適より何秒遅くなるか)の、人ごとの 4 ゲーム平均。小さい 4 人を「うまい群」、大きい 4 人を「下手な群」とした。作成: tools/analyze_pattern7_by_skill.py</p>
{('<div class="findings">' + fh + '</div>') if fh else ''}
<h2>群分け</h2>
{t(per_show, ['参加者ID', '群', 'L0の平均', 'L0の中央値', 'L0の最大', '指示の動作'])}
<h2>図: 群ごとの d の線(平均 ± 標準誤差)</h2>
<img src="data:image/png;base64,{img}" alt="群ごと">
<h2>群 × d の平均</h2>
{t(means, ['指標', '群'] + [A.D_LABEL[l] for l in A.D_ORDER])}
<h2>群の中の傾きと、群の差</h2>
<p class="note">自由の効果 = (d=1, 2, ∞ の平均) − d=0。正なら AI に自由があるほうが高い(時間なら遅い)。
群の差の p は 4 人 vs 4 人の全 70 通りの並べ替え(最小でも .029)。傾きの p は群ごと 4 人なので参考程度。</p>
{t(tests, ['指標', 'うまい群_自由の効果', '下手な群_自由の効果', '群の差(下手−うまい)', '群の差_p', 'うまい群_傾きρ', 'うまい群_傾きp', '下手な群_傾きρ', '下手な群_傾きp'], pcols=('群の差_p', 'うまい群_傾きp', '下手な群_傾きp'))}
<h2>補足: ゲーム単位(その回の指示のロスの大小 × d=0 か)</h2>
<p class="note">L0大 = その回の指示の L0 が {GAME_HIGH_L0:g} 秒以上。人をまたいだ比較なので、個人差が混ざる。</p>
{t(games, gcols)}
</body></html>"""
    (OUT / 'report.html').write_text(doc, encoding='utf-8')


if __name__ == '__main__':
    main()
