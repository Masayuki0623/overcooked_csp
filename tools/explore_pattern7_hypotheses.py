"""パターン7のデータで、どんな作業仮説なら差が出るかを総当たりで探す(探索用)。

    python tools/explore_pattern7_hypotheses.py

注意: 同じデータで仮説を探して同じデータで検定すると、p 値は額面どおりの意味を持たない。
ここでは全部の検定に Benjamini-Hochberg の補正(q 値)を掛け、次の参加者で確かめる候補を
選ぶためだけに使う。確かめるのに要る人数の目安も出す。

調べる仮説の族:
  A. 条件の対比(人の中): 結果 × 対比(d=0 対 ほか、線形の傾き、逆 U など)
     p は、人ごとの対比の値に Wilcoxon の符号順位検定(全通り)。
  B. 人の中の関係: 過程の指標(待ち・挟んだ時間・ミス・L0 など)× 体験の評価
     人ごとに順位を中心化した相関。p は人の中での並べ替え。
  C. 人の特徴: 事前の期待・経験・端末など × 人ごとの平均や d=0 の効果
     人単位の順位相関(8 人)。p は並べ替え。

読むもの: results/experience1/analysis/pattern7/quant_merged.csv(analyze_pattern7 などで作った表)
書くもの: results/experience1/analysis/pattern7/hypothesis_search.csv
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_pattern7 as A  # noqa: E402

SRC = A.OUT / 'quant_merged.csv'
OUT = A.OUT / 'hypothesis_search.csv'
N_PERM_WITHIN = 4000
N_PERM_PERSON = 20000

PROCESS = {
    'スコア_かかった時間_秒': 'かかった時間',
    '計画とのずれ': '計画とのずれ',
    '最後のスープ': '最後のスープを入れた時刻',
    '2品目から最後まで': '2品目から最後まで',
    'hu_idle': '人が止まっていた時間',
    'ai_idle': 'AIが止まっていた時間',
    '人のミスの回数': '人のミス',
    'n_out': '挟んだ数(待ちの外)',
    'task_s_out': '挟んだ実時間(待ちの外)',
    'task_s_wait': '挟んだ実時間(待ちの間)',
    '即時実行の効率損失量L0_秒': 'L0',
    '解釈の回数': '指示の書き直し回数',
    '人がやったか': '人が指示の作業をやった',
    'AIがやり切った': 'AIが指示をやり切った',
    '切る指示': '「切って」の指示',
}
SURVEY = {
    'つながり平均': 'つながり', '協調平均': '協調', 'ラポール': 'ラポール',
    'オーダー尊重感': '9 指示を大切に', '意図理解感': '10 意図を分かっていた',
    '補完感': '11 必要な作業', '判断納得感': '13 納得', '予測可能感': '14 予想どおり',
    '後回しにされた感': '15 後回し', '楽しさ': '16 楽しさ', '自律性': '17 自由',
    '熟達': '18 うまく', '指示への自信': '指示への自信',
}
# 対比の重み(d=0, 1, 2, ∞ の順)。値は「重み付きの平均の差」
CONTRASTS = {
    'd=0 対 ほか': [-1, 1 / 3, 1 / 3, 1 / 3],
    'd=0 対 d=1・2': [-1, .5, .5, 0],
    '線形の傾き(0<1<2<∞)': [-1.5, -.5, .5, 1.5],
    'd=∞ 対 d=0・1・2': [-1 / 3, -1 / 3, -1 / 3, 1],
    'd=∞ 対 d=1・2': [0, -.5, -.5, 1],
    '逆U(1・2 対 0・∞)': [-.5, .5, .5, -.5],
    'd=0 対 d=1': [-1, 1, 0, 0],
    'd=0 対 d=2': [-1, 0, 1, 0],
    'd=0 対 d=∞': [-1, 0, 0, 1],
}


def bh(ps):
    ps = np.asarray(ps, float)
    n = len(ps)
    order = np.argsort(ps)
    q = np.empty(n)
    prev = 1.0
    for rank, i in reversed(list(enumerate(order, 1))):
        prev = min(prev, ps[i] * n / rank)
        q[i] = prev
    return q


def n_for_dz(dz):
    if not dz or not np.isfinite(dz):
        return None
    return math.ceil(((1.96 + 0.84) / abs(dz)) ** 2 / 0.955 + 2)


def n_people_for_r(r, per_person=3):
    """人の中の相関 r を 80% で検出するのに要る人数の目安(1 人あたり自由度 3)。"""
    if not r or not np.isfinite(r) or abs(r) >= 0.999:
        return None
    obs = ((1.96 + 0.84) / math.atanh(abs(r))) ** 2 + 3
    return math.ceil(obs / per_person)


def main():
    rng = np.random.default_rng(A.SEED)
    m = pd.read_csv(SRC, encoding='utf-8-sig', dtype={'d': str})
    m['AIがやり切った'] = m['done'].astype(int)
    m['切る指示'] = m['指示の動作'].eq('chop').astype(int)
    rows = []

    # A. 条件の対比
    for col, name in {**PROCESS, **SURVEY}.items():
        if col in ('切る指示',):
            continue
        w = m.pivot_table(index='参加者ID', columns='d', values=col, aggfunc='first')
        w = w.reindex(columns=A.D_ORDER).astype(float)
        if w.isna().any().any():
            continue
        for cname, wt in CONTRASTS.items():
            c = (w.to_numpy() * np.array(wt)).sum(axis=1)
            if np.allclose(c, 0):
                continue
            _, p, r, _ = A.wilcoxon_exact(np.zeros(len(c)), c)
            sd = c.std(ddof=1)
            dz = c.mean() / sd if sd > 0 else np.nan
            rows.append({'族': 'A 条件の対比', '仮説': f'{name}: {cname}',
                         '効果': round(float(c.mean()), 3), '効果量': round(float(dz), 2) if np.isfinite(dz) else None,
                         '同じ向きの人': f'{int((c > 0).sum())}+/{int((c < 0).sum())}-',
                         'p': p, '確かめるのに要る人数': n_for_dz(dz)})

    # B. 人の中の関係(過程 × 体験)、体験どうしも一部
    pairs = [(a, b) for a in PROCESS for b in SURVEY]
    for a, b in pairs:
        if m[a].nunique() < 2:
            continue
        r, p, n = A.spearman_within(m, a, b, rng, n_perm=N_PERM_WITHIN)
        if not np.isfinite(r):
            continue
        rows.append({'族': 'B 人の中の関係', '仮説': f'{PROCESS[a]} ↔ {SURVEY[b]}',
                     '効果': round(r, 3), '効果量': round(r, 2), '同じ向きの人': '',
                     'p': p, '確かめるのに要る人数': n_people_for_r(r)})

    # C. 人の特徴 × 人ごとの結果
    per = m.groupby('参加者ID').agg(
        **{'事前の期待_段取り': ('AIの能力の予想_段取り', 'first'),
           '事前の期待_意図': ('AIの能力の予想_意図の理解', 'first'),
           'ゲーム経験あり': ('ゲーム経験', lambda s: int(s.iloc[0] == 'yes')),
           'スマホ': ('端末', lambda s: int(s.iloc[0] == 'スマホ')),
           '年齢': ('年齢', 'first'),
           'L0の平均': ('即時実行の効率損失量L0_秒', 'mean')})
    outs = {}
    for col, name in [('スコア_かかった時間_秒', 'かかった時間の平均'), ('ラポール', 'ラポールの平均'),
                      ('判断納得感', '納得の平均'), ('楽しさ', '楽しさの平均'), ('意図理解感', '意図理解の平均')]:
        outs[name] = m.groupby('参加者ID')[col].mean()
    for col, name in [('スコア_かかった時間_秒', 'd=0 の遅さ'), ('協調平均', 'd=0 での協調の下がり'),
                      ('ラポール', 'd=0 でのラポールの下がり')]:
        w = m.pivot_table(index='参加者ID', columns='d', values=col, aggfunc='first')
        outs[name] = w['0'] - w[['1', '2', 'inf']].mean(axis=1)
        if col != 'スコア_かかった時間_秒':
            outs[name] = -outs[name]
    for mod in per.columns:
        x = per[mod].rank().to_numpy(float)
        if np.std(x) == 0:
            continue
        for oname, y in outs.items():
            yy = y.reindex(per.index).rank().to_numpy(float)
            r = float(np.corrcoef(x, yy)[0, 1])
            perm = np.array([np.corrcoef(x, rng.permutation(yy))[0, 1] for _ in range(N_PERM_PERSON)])
            p = float((np.abs(perm) >= abs(r) - 1e-9).mean())
            rows.append({'族': 'C 人の特徴', '仮説': f'{mod} ↔ {oname}', '効果': round(r, 3),
                         '効果量': round(r, 2), '同じ向きの人': '', 'p': p,
                         '確かめるのに要る人数': math.ceil(((1.96 + 0.84) / math.atanh(min(abs(r), .99))) ** 2 + 3) if r else None})

    df = pd.DataFrame(rows)
    df['q(全体)'] = bh(df['p'])
    df['q(族の中)'] = df.groupby('族')['p'].transform(lambda s: pd.Series(bh(s), index=s.index))
    df = df.sort_values('p')
    df.to_csv(OUT, index=False, encoding='utf-8-sig')
    print('検定の数:', len(df), df['族'].value_counts().to_dict())
    print('p<.05:', int((df.p < .05).sum()), ' 偶然でも出る数の目安:', round(len(df) * .05, 1))
    print('q<.10(全体):', int((df['q(全体)'] < .10).sum()), ' q<.10(族の中):', int((df['q(族の中)'] < .10).sum()))
    pd.set_option('display.width', 250)
    pd.set_option('display.max_colwidth', 60)
    print(df.head(40).to_string(index=False))
    print('出力:', OUT)


if __name__ == '__main__':
    main()
