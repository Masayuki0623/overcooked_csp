"""パターン7(自由記述の指示)の本番データを、割り込み許容数 d ごとに比べる。

    python tools/analyze_pattern7.py                     # 有効な参加者(下の VALID)で
    python tools/analyze_pattern7.py --pids p25,p27      # 参加者を指定して

読むもの: results/experience1/all_in_one_pattern7.csv(読むだけ。書き換えない)
書くもの: results/experience1/analysis/pattern7/
    by_d.csv          d ごとの平均・標準偏差・中央値
    tests.csv         検定の結果(Friedman、傾きの検定、対ごとの Wilcoxon)
    order.csv         何ゲーム目か(慣れ)による違い
    instructions.csv  参加者が書いた指示と、その解釈
    fig_*.png         図
    report.html       上の全部を 1 枚にまとめたもの(findings.html があれば先頭に入れる)

検定は scipy を使わず、並べ替え(順列)で正確に出す。人数が少ないので漸近近似より確か。
  - Friedman 検定: 人ごとに 4 条件の順位を付け、条件の順位和の偏りを見る。p は人ごとに
    条件のラベルを入れ替えた 20 万回の並べ替えから。効果量は Kendall の W。
  - 傾きの検定(Page): d の大小(0 < 1 < 2 < inf)に沿って増える/減るか。
    ρ̄ は人ごとの Spearman 相関(d の順と値)の平均。p は両側。
  - 対ごとの比較: Wilcoxon の符号順位検定(全通りの符号を数える正確な p)、Holm で補正。
    効果量 r は対応のある順位双列相関((W+ − W−)/(W+ + W−))。
"""
import argparse
import base64
import html
import io
import math
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'results' / 'experience1' / 'all_in_one_pattern7.csv'
OUT = ROOT / 'results' / 'experience1' / 'analysis' / 'pattern7'

# 有効な参加者(2026-10-09 にユーザーが指定。p24 以前は動作確認、p26/p29/p30/p31 は途中まで・確認用)
VALID = ['p25', 'p27', 'p28', 'p32', 'p33', 'p34', 'p35', 'p36']

D_ORDER = ['0', '1', '2', 'inf']
D_LABEL = {'0': 'd=0', '1': 'd=1', '2': 'd=2', 'inf': 'd=∞'}
AGENT = {'0': 'A', '1': 'B', '2': 'C', 'inf': 'D'}
N_PERM = 200_000
SEED = 20261009

# (列, 表示名, 群)。群: perf=成績、behav=AI の動き、survey=アンケート
MEASURES = [
    ('スコア_かかった時間_秒', 'かかった時間(秒)', 'perf'),
    ('効率損失量L_秒', '効率損失 L(秒)', 'perf'),
    ('人のミスの回数', '人のミスの回数', 'perf'),
    ('終わるまでに挟まった作業数', '指示の完了までに挟まった作業数', 'behav'),
    ('指示後に着手するまで_秒', '指示から着手まで(秒)', 'behav'),
    ('ラポール', 'ラポール(CCR 短縮)', 'survey'),
    ('つながり平均', 'つながり', 'survey'),
    ('協調平均', '協調', 'survey'),
    ('オーダー尊重感', '9 指示を大切に扱った', 'survey'),
    ('意図理解感', '10 したいことを分かっていた', 'survey'),
    ('補完感', '11 言わなくても必要な作業をした', 'survey'),
    ('判断納得感', '13 納得できる判断で動いた', 'survey'),
    ('予測可能感', '14 予想したとおりに動いた', 'survey'),
    ('後回しにされた感', '15 指示を後回しにした', 'survey'),
    ('指示への自信', '指示への自信', 'survey'),
    ('楽しさ', '16 楽しくプレイできた', 'survey'),
    ('自律性', '17 自由にプレイできた', 'survey'),
    ('熟達', '18 うまくプレイできた', 'survey'),
]


# ---------------------------------------------------------------- 検定
def chi2_sf(x, df):
    """カイ二乗分布の上側確率(df=2, 3 だけ使う。scipy なしで閉じた式)。"""
    if df == 2:
        return math.exp(-x / 2)
    if df == 3:
        return math.erfc(math.sqrt(x / 2)) + math.sqrt(2 * x / math.pi) * math.exp(-x / 2)
    if df == 1:
        return math.erfc(math.sqrt(x / 2))
    raise ValueError(df)


def row_ranks(m):
    """各行の中での順位(同順位は平均)。"""
    return pd.DataFrame(m).rank(axis=1).to_numpy()


def friedman(m, rng):
    """m: 人 × 条件(欠けなし)。Q(同順位補正つき)、漸近 p、並べ替え p、Kendall の W。"""
    n, k = m.shape
    r = row_ranks(m)
    ties = 0.0
    for row in r:
        _, c = np.unique(row, return_counts=True)
        ties += float(((c ** 3) - c).sum())
    denom = 1 - ties / (n * (k ** 3 - k))

    def stat(rr):
        rs = rr.sum(axis=-2)
        return ((12 / (n * k * (k + 1))) * (rs ** 2).sum(axis=-1) - 3 * n * (k + 1))

    q0 = stat(r)
    q = q0 / denom if denom > 0 else float('nan')
    perm = rng.permuted(np.broadcast_to(r, (N_PERM, n, k)), axis=2)
    p_perm = float((stat(perm) >= q0 - 1e-9).mean())
    w = q / (n * (k - 1)) if denom > 0 else float('nan')
    return q, chi2_sf(q, k - 1) if denom > 0 else float('nan'), p_perm, w


def page_trend(m, rng):
    """d の順に沿う傾き。ρ̄(人ごとの Spearman の平均)と両側の並べ替え p。"""
    n, k = m.shape
    r = row_ranks(m)
    c = np.arange(1, k + 1)

    def lstat(rr):
        return (rr.sum(axis=-2) * c).sum(axis=-1)

    l0 = lstat(r)
    mean_l = n * (k + 1) / 2 * c.sum()
    perm = rng.permuted(np.broadcast_to(r, (N_PERM, n, k)), axis=2)
    p = float((np.abs(lstat(perm) - mean_l) >= abs(l0 - mean_l) - 1e-9).mean())
    rho = 12 * l0 / (n * k * (k ** 2 - 1)) - 3 * (k + 1) / (k - 1)
    return rho, p


def wilcoxon_exact(x, y):
    """対応のある 2 群。0 の差は除く。(W+, 両側の正確な p, 順位双列相関 r, 使った人数)"""
    dlt = np.asarray(y, float) - np.asarray(x, float)
    dlt = dlt[np.abs(dlt) > 1e-9]
    m = len(dlt)
    if m == 0:
        return 0.0, 1.0, 0.0, 0
    ranks = pd.Series(np.abs(dlt)).rank().to_numpy()
    wp = float(ranks[dlt > 0].sum())
    total = ranks.sum()
    center = total / 2
    obs = abs(wp - center)
    hits = 0
    for signs in product((0, 1), repeat=m):
        s = float((ranks * np.array(signs)).sum())
        hits += abs(s - center) >= obs - 1e-9
    p = hits / 2 ** m
    r = (wp - (total - wp)) / total
    return wp, p, r, m


def holm(ps):
    ps = list(ps)
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out = [None] * len(ps)
    run = 0.0
    for j, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - j) * ps[i]))
        out[i] = run
    return out


def spearman_within(df, a, b, rng, n_perm=20000):
    """人ごとに中心化した順位で Spearman。p は人の中で値を入れ替えて。"""
    sub = df[['参加者ID', a, b]].dropna()
    if sub[a].nunique() < 2 or sub[b].nunique() < 2:
        return float('nan'), float('nan'), len(sub)
    ra = sub.groupby('参加者ID')[a].rank()
    rb = sub.groupby('参加者ID')[b].rank()
    ra = ra - ra.groupby(sub['参加者ID']).transform('mean')
    rb = rb - rb.groupby(sub['参加者ID']).transform('mean')

    def corr(u, v):
        u = u - u.mean()
        v = v - v.mean()
        den = math.sqrt((u ** 2).sum() * (v ** 2).sum())
        return float((u * v).sum() / den) if den else float('nan')

    r0 = corr(ra.to_numpy(), rb.to_numpy())
    groups = [np.where(sub['参加者ID'].to_numpy() == g)[0] for g in sub['参加者ID'].unique()]
    rbv = rb.to_numpy()
    hits = 0
    for _ in range(n_perm):
        sh = rbv.copy()
        for idx in groups:
            sh[idx] = rbv[rng.permutation(idx)]
        hits += abs(corr(ra.to_numpy(), sh)) >= abs(r0) - 1e-9
    return r0, hits / n_perm, len(sub)


# ---------------------------------------------------------------- 読み込み
def load(pids):
    df = pd.read_csv(SRC, encoding='utf-8-sig', skiprows=[1], dtype={'割り込み許容数': str})
    df = df[df['参加者ID'].isin(pids)].copy()
    df['d'] = df['割り込み許容数'].astype(str).str.replace('.0', '', regex=False)
    missing = sorted(set(pids) - set(df['参加者ID']))
    return df, missing


def wide(df, col, levels):
    w = df.pivot_table(index='参加者ID', columns='d', values=col, aggfunc='first')
    w = w.reindex(columns=levels)
    return w.dropna()


# ---------------------------------------------------------------- 図
def setup_mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in ('C:/Windows/Fonts/BIZ-UDGothicR.ttc', 'C:/Windows/Fonts/BIZ-UDGothicB.ttc'):
        if Path(f).exists():
            font_manager.fontManager.addfont(f)
    plt.rcParams['font.family'] = ['BIZ UDGothic', 'Yu Gothic', 'MS Gothic']
    plt.rcParams['axes.unicode_minus'] = False
    return plt


def panel(ax, df, col, title, levels, ylim=None):
    w = df.pivot_table(index='参加者ID', columns='d', values=col, aggfunc='first').reindex(columns=levels)
    x = np.arange(len(levels))
    rng = np.random.default_rng(0)
    for _, row in w.iterrows():
        jit = rng.uniform(-0.06, 0.06)
        ax.plot(x + jit, row.to_numpy(float), color='#9aa4b2', lw=0.8, alpha=0.7, marker='o', ms=3)
    mean = w.mean()
    se = w.std(ddof=1) / np.sqrt(w.count())
    ax.errorbar(x, mean.to_numpy(float), yerr=se.to_numpy(float), color='#1f3b73', lw=2.2,
                marker='s', ms=6, capsize=4, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels([D_LABEL[l] for l in levels])
    ax.set_title(title, fontsize=10)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis='y', alpha=0.3)


def fig_to_png(fig, path):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    path.write_bytes(buf.getvalue())
    return base64.b64encode(buf.getvalue()).decode('ascii')


# ---------------------------------------------------------------- 本体
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pids', default=','.join(VALID))
    args = ap.parse_args()
    pids = [p.strip() for p in args.pids.split(',') if p.strip()]
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    df, missing = load(pids)
    n_games = df.groupby('参加者ID').size()

    # d ごとの記述統計
    rows = []
    for col, name, grp in MEASURES:
        for lv in D_ORDER:
            v = df.loc[df['d'] == lv, col].dropna().astype(float)
            rows.append({'指標': name, '列': col, '群': grp, 'd': D_LABEL[lv], 'n': len(v),
                         '平均': v.mean() if len(v) else np.nan,
                         '標準偏差': v.std(ddof=1) if len(v) > 1 else np.nan,
                         '中央値': v.median() if len(v) else np.nan})
    by_d = pd.DataFrame(rows)
    by_d.to_csv(OUT / 'by_d.csv', index=False, encoding='utf-8-sig')

    # 検定
    tests = []
    pairs_out = []
    for col, name, grp in MEASURES:
        levels = [l for l in D_ORDER if df.loc[df['d'] == l, col].notna().any()]
        w = wide(df, col, levels)
        if len(levels) < 3 or len(w) < 3 or w.to_numpy().std() == 0:
            continue
        m = w.to_numpy(float)
        q, p_asym, p_perm, kw = friedman(m, rng)
        rho, p_tr = page_trend(m, rng)
        tests.append({'指標': name, '列': col, '群': grp, '条件': '/'.join(D_LABEL[l] for l in levels),
                      'n': len(w), 'Friedman_Q': q, 'p_並べ替え': p_perm, 'p_漸近': p_asym,
                      'Kendall_W': kw, '傾き_rho': rho, '傾き_p': p_tr})
        prs = list(combinations(levels, 2))
        res = [wilcoxon_exact(w[a], w[b]) for a, b in prs]
        adj = holm([r[1] for r in res])
        for (a, b), (wp, p, r, nn), pa in zip(prs, res, adj):
            pairs_out.append({'指標': name, '比較': f'{D_LABEL[a]} → {D_LABEL[b]}',
                              '平均の差(後−前)': float((w[b] - w[a]).mean()),
                              'n(差が0でない)': nn, 'W+': wp, 'p': p, 'p_Holm': pa, 'r': r})
    tests = pd.DataFrame(tests)
    pairs = pd.DataFrame(pairs_out)
    tests.to_csv(OUT / 'tests.csv', index=False, encoding='utf-8-sig')
    pairs.to_csv(OUT / 'pairs.csv', index=False, encoding='utf-8-sig')

    # 何ゲーム目か(慣れ)
    order_rows = []
    for col, name in [('スコア_かかった時間_秒', 'かかった時間(秒)'), ('ラポール', 'ラポール'),
                      ('熟達', '18 うまくプレイできた')]:
        w = df.pivot_table(index='参加者ID', columns='セッション番号', values=col,
                           aggfunc='first').reindex(columns=[1, 2, 3, 4]).dropna()
        q, p_asym, p_perm, kw = friedman(w.to_numpy(float), rng)
        rho, p_tr = page_trend(w.to_numpy(float), rng)
        order_rows.append({'指標': name, **{f'{i}回目の平均': w[i].mean() for i in [1, 2, 3, 4]},
                           'Friedman_Q': q, 'p_並べ替え': p_perm, 'Kendall_W': kw,
                           '傾き_rho': rho, '傾き_p': p_tr})
    order = pd.DataFrame(order_rows)
    order.to_csv(OUT / 'order.csv', index=False, encoding='utf-8-sig')

    # 人の中での関係
    corr_specs = [
        ('終わるまでに挟まった作業数', '後回しにされた感'),
        ('終わるまでに挟まった作業数', 'オーダー尊重感'),
        ('スコア_かかった時間_秒', '熟達'),
        ('スコア_かかった時間_秒', 'ラポール'),
        ('スコア_かかった時間_秒', '楽しさ'),
        ('オーダー尊重感', 'ラポール'),
        ('予測可能感', 'ラポール'),
    ]
    corr_rows = []
    for a, b in corr_specs:
        r, p, nn = spearman_within(df, a, b, rng)
        corr_rows.append({'A': a, 'B': b, 'n': nn, '人の中の順位相関': r, 'p_並べ替え': p})
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(OUT / 'correlations.csv', index=False, encoding='utf-8-sig')

    # 指示の文
    ins = df.sort_values(['参加者ID', 'セッション番号'])[
        ['参加者ID', 'セッション番号', 'd', '指示の文', '解釈の回数', '解釈した作業', '指示の工程数_AI',
         '終わるまでに挟まった作業数', 'スコア_かかった時間_秒', '指示のやりとり']].copy()
    ins['d'] = ins['d'].map(D_LABEL)
    ins.to_csv(OUT / 'instructions.csv', index=False, encoding='utf-8-sig')

    # 参加者の属性
    people = df.groupby('参加者ID').agg(
        グループ=('グループ', 'first'), 年齢=('年齢', 'first'), 性別=('性別', 'first'),
        ゲーム経験=('ゲーム経験', 'first'), 端末=('端末', 'first'),
        AI予想_段取り=('AIの能力の予想_段取り', 'first'),
        AI予想_意図=('AIの能力の予想_意図の理解', 'first'),
        ゲーム数=('セッション番号', 'count'))
    people['d の順'] = df.sort_values('セッション番号').groupby('参加者ID')['d'].agg(
        lambda s: ' → '.join(D_LABEL[x] for x in s))
    people.to_csv(OUT / 'participants.csv', encoding='utf-8-sig')

    # 図
    plt = setup_mpl()
    imgs = {}
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    panel(axes[0], df, 'スコア_かかった時間_秒', 'かかった時間(秒)', D_ORDER)
    panel(axes[1], df, '効率損失量L_秒', '効率損失 L(秒) ※d=∞ は算出なし', D_ORDER[:3])
    panel(axes[2], df, '終わるまでに挟まった作業数', '指示の完了までに挟まった作業数', D_ORDER)
    panel(axes[3], df, '人のミスの回数', '人のミスの回数', D_ORDER)
    imgs['perf'] = fig_to_png(fig, OUT / 'fig_performance.png')
    plt.close(fig)

    surv = [m for m in MEASURES if m[2] == 'survey']
    ncol = 5
    nrow = math.ceil(len(surv) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(17, 3.3 * nrow))
    for ax, (col, name, _) in zip(axes.flat, surv):
        panel(ax, df, col, name, D_ORDER, ylim=(0.6, 5.4))
    for ax in list(axes.flat)[len(surv):]:
        ax.axis('off')
    fig.tight_layout()
    imgs['survey'] = fig_to_png(fig, OUT / 'fig_survey.png')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, (col, name) in zip(axes, [('スコア_かかった時間_秒', 'かかった時間(秒)'), ('ラポール', 'ラポール')]):
        w = df.pivot_table(index='参加者ID', columns='セッション番号', values=col, aggfunc='first')
        for _, row in w.iterrows():
            ax.plot(w.columns, row.to_numpy(float), color='#9aa4b2', lw=0.8, marker='o', ms=3)
        ax.errorbar(w.columns, w.mean().to_numpy(float),
                    yerr=(w.std(ddof=1) / np.sqrt(w.count())).to_numpy(float),
                    color='#8a3b12', lw=2.2, marker='s', capsize=4)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels([f'{i}回目' for i in [1, 2, 3, 4]])
        ax.set_title(name + '(何ゲーム目か)', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    imgs['order'] = fig_to_png(fig, OUT / 'fig_order.png')
    plt.close(fig)

    write_report(df, pids, missing, n_games, by_d, tests, pairs, order, corr, ins, people, imgs)
    print('参加者', len(n_games), '人 / ゲーム', len(df), '件 / 足りない参加者', missing or 'なし')
    print('出力:', OUT)


# ---------------------------------------------------------------- まとめの HTML
def fmt(v, nd=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return '—'
    if isinstance(v, (float, np.floating)):
        return f'{v:.{nd}f}'
    return html.escape(str(v))


def pfmt(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return '—'
    s = '<.001' if p < 0.001 else f'{p:.3f}'
    return f'<b class="sig">{s}</b>' if p < 0.05 else (f'<span class="tr">{s}</span>' if p < 0.10 else s)


def table(df, cols, pcols=(), nd=2, nds=None):
    nds = nds or {}
    h = ['<table><thead><tr>' + ''.join(f'<th>{html.escape(c)}</th>' for c in cols) + '</tr></thead><tbody>']
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f'<td>{pfmt(v) if c in pcols else fmt(v, nds.get(c, nd))}</td>')
        h.append('<tr>' + ''.join(cells) + '</tr>')
    h.append('</tbody></table>')
    return ''.join(h)


def write_report(df, pids, missing, n_games, by_d, tests, pairs, order, corr, ins, people, imgs):
    piv = by_d.pivot_table(index=['群', '指標'], columns='d', values=['平均', '標準偏差'], sort=False)
    mrows = []
    for (grp, name) in piv.index:
        r = {'指標': name}
        for lv in D_ORDER:
            lab = D_LABEL[lv]
            mu = piv.loc[(grp, name)].get(('平均', lab), np.nan)
            sd = piv.loc[(grp, name)].get(('標準偏差', lab), np.nan)
            r[lab] = '—' if pd.isna(mu) else f'{mu:.2f} <span class="sd">({sd:.2f})</span>'
        mrows.append(r)
    mean_tbl = ['<table><thead><tr><th>指標</th>' + ''.join(f'<th>{D_LABEL[l]}(エージェント {AGENT[l]})</th>' for l in D_ORDER) + '</tr></thead><tbody>']
    for r in mrows:
        mean_tbl.append('<tr><td>' + html.escape(r['指標']) + '</td>' + ''.join(f'<td>{r[D_LABEL[l]]}</td>' for l in D_ORDER) + '</tr>')
    mean_tbl.append('</tbody></table>')

    findings = OUT / 'findings.html'
    findings_html = findings.read_text(encoding='utf-8') if findings.exists() else ''
    ins_show = ins.copy()
    ins_show['指示のやりとり'] = ins_show['指示のやりとり'].fillna('')

    doc = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>パターン7 分析</title>
<style>
:root {{ --bg:#fff; --fg:#1d2330; --muted:#6b7280; --line:#e3e6ec; --accent:#1f3b73; --sig:#b42318; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#14171d; --fg:#e7e9ee; --muted:#9aa1ad; --line:#2a2f39; --accent:#9db7f0; --sig:#ff8a7a; }} img {{ background:#fff; }} }}
body {{ background:var(--bg); color:var(--fg); font-family:'BIZ UDPGothic','Yu Gothic',sans-serif; margin:0 auto; max-width:1100px; padding:16px; line-height:1.6; }}
h1 {{ font-size:22px; }} h2 {{ font-size:18px; border-bottom:2px solid var(--accent); padding-bottom:4px; margin-top:32px; }}
table {{ border-collapse:collapse; font-size:13px; margin:8px 0; display:block; overflow-x:auto; }}
th, td {{ border:1px solid var(--line); padding:4px 8px; text-align:right; white-space:nowrap; }}
th {{ background:color-mix(in srgb, var(--accent) 10%, transparent); }}
td:first-child, th:first-child {{ text-align:left; }}
.sd {{ color:var(--muted); font-size:11px; }} .sig {{ color:var(--sig); }} .tr {{ text-decoration:underline dotted; }}
img {{ max-width:100%; }} .note {{ color:var(--muted); font-size:13px; }}
.wrap td {{ white-space:normal; text-align:left; max-width:420px; }}
.findings {{ border-left:4px solid var(--accent); padding:4px 16px; background:color-mix(in srgb, var(--accent) 6%, transparent); }}
</style></head><body>
<h1>パターン7(自由記述の指示)の分析</h1>
<p class="note">参加者 {len(n_games)} 人({', '.join(pids)})/ ゲーム {len(df)} 件。
足りない参加者: {', '.join(missing) if missing else 'なし'}。
元データ: results/experience1/all_in_one_pattern7.csv。作成: tools/analyze_pattern7.py</p>
{('<div class="findings">' + findings_html + '</div>') if findings_html else ''}

<h2>参加者</h2>
{table(people.reset_index(), ['参加者ID', 'グループ', '年齢', '性別', 'ゲーム経験', '端末', 'AI予想_段取り', 'AI予想_意図', 'ゲーム数', 'd の順'], nd=0)}

<h2>d ごとの平均(標準偏差)</h2>
<p class="note">d は割り込み許容数(AI が指示より先に挟んでよい作業の数)。アンケートは 1〜5。かかった時間は小さいほど速い。</p>
{''.join(mean_tbl)}

<h2>図: 成績と AI の動き</h2>
<p class="note">灰色の線が一人ずつ、紺が平均 ± 標準誤差。</p>
<img src="data:image/png;base64,{imgs['perf']}" alt="成績">

<h2>図: アンケート</h2>
<img src="data:image/png;base64,{imgs['survey']}" alt="アンケート">

<h2>検定: d による違い</h2>
<p class="note">Friedman の p は並べ替え {N_PERM:,} 回。W は Kendall の一致係数(0.1 小 / 0.3 中 / 0.5 大 が目安)。
傾き ρ̄ は d が大きくなるほど値が上がるなら正。赤字 p&lt;.05、点線 p&lt;.10。人数が 8 人なので、検出力は低い。</p>
{table(tests, ['指標', '条件', 'n', 'Friedman_Q', 'p_並べ替え', 'p_漸近', 'Kendall_W', '傾き_rho', '傾き_p'], pcols=('p_並べ替え', 'p_漸近', '傾き_p'))}

<h2>対ごとの比較(Wilcoxon 符号順位検定、正確な p、Holm 補正)</h2>
<p class="note">r は対応のある順位双列相関(後の条件のほうが大きければ正)。差が 0 の人は除く。</p>
{table(pairs, ['指標', '比較', '平均の差(後−前)', 'n(差が0でない)', 'W+', 'p', 'p_Holm', 'r'], pcols=('p', 'p_Holm'), nds={'n(差が0でない)': 0, 'W+': 1})}

<h2>何ゲーム目か(慣れ)</h2>
<img src="data:image/png;base64,{imgs['order']}" alt="順序">
{table(order, ['指標', '1回目の平均', '2回目の平均', '3回目の平均', '4回目の平均', 'Friedman_Q', 'p_並べ替え', 'Kendall_W', '傾き_rho', '傾き_p'], pcols=('p_並べ替え', '傾き_p'))}

<h2>人の中での関係</h2>
<p class="note">人ごとに順位を中心化してから求めた順位相関(人による水準の違いを除く)。p は人の中での入れ替え。</p>
{table(corr, ['A', 'B', 'n', '人の中の順位相関', 'p_並べ替え'], pcols=('p_並べ替え',), nds={'n': 0})}

<h2>参加者が書いた指示</h2>
<div class="wrap">{table(ins_show, ['参加者ID', 'セッション番号', 'd', '指示の文', '解釈の回数', '解釈した作業', '指示の工程数_AI', '終わるまでに挟まった作業数', 'スコア_かかった時間_秒', '指示のやりとり'], nd=1, nds={'セッション番号': 0, '解釈の回数': 0, '指示の工程数_AI': 0, '終わるまでに挟まった作業数': 0})}</div>
</body></html>"""
    (OUT / 'report.html').write_text(doc, encoding='utf-8')


if __name__ == '__main__':
    main()
