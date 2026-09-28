"""実験用の記録ファイル2つの検証。

    定性 results/exp_qualitative.csv  : アンケート1回1行 + その回のスコア
    定量 results/exp_quantitative.csv : 指示1回1行 + L + その回のスコア

ここで確かめること:
    1. 見出しが全部日本語で、そのすぐ下に説明の行がある
    2. 定性にアンケートの全項目(つながり4/協調4/指示6/自由記述3)が入る
    3. 定性にもゲームスコア(提供数・失敗数・プレイ時間)が入る
    4. 定量に L と L0 が入る
    5. 定量は指示ごとに1行(縦軸が指示)
    6. 指示が1回も出なかった回も、スコアだけ1行残る
    7. 条件の欄は2つのファイルで同じ並び(突き合わせられる)
    8. Excel で読めるよう BOM が付く

実行方法:
    python tests/repro_experiment_files.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'agent'))

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

import csv
import io
import re
import tempfile
from pathlib import Path

import server as srv

results = []


def check(label, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}{(' -> ' + detail) if detail else ''}")


KANA = re.compile(r'[ぁ-んァ-ヶ一-龠]')


def is_japanese(name):
    # 「L」「L0」のような記号を含む見出しは、日本語が1文字でもあればよい
    return bool(KANA.search(name))


# 1. 見出しと説明
for label, fields, notes in (
        ('定性', srv.QUAL_FIELDS, srv.QUAL_NOTES),
        ('定量', srv.QUANT_FIELDS, srv.QUANT_NOTES)):
    bad = [f for f in fields if not is_japanese(f)]
    check(f'{label}: 見出しが日本語', not bad, f'日本語でない: {bad}')
    # 参加者ID など、説明が要らない欄は空でよい
    check(f'{label}: 説明が用意されている',
          all(f in notes for f in fields),
          f'説明が無い: {[f for f in fields if f not in notes]}')

# 2 + 3. 定性の中身
for i in range(1, 5):
    pass
need_qual = ([f'つながり{i}' for i in range(1, 5)]
             + [f'協調{i}' for i in range(1, 5)]
             + [f'指示{i}' for i in range(1, 7)]
             + ['つながり平均', '協調平均', 'ラポール', '指示平均',
                'うまく噛み合ったところ', '気になったところ', '指示に対するAIの動き'])
miss = [c for c in need_qual if c not in srv.QUAL_FIELDS]
check('定性: アンケートの全項目が入っている', not miss, f'足りない: {miss}')
miss = [c for c in ('提供数', '失敗数', 'プレイ時間_秒') if c not in srv.QUAL_FIELDS]
check('定性: ゲームスコアも入っている', not miss, f'足りない: {miss}')

# 4. 定量の中身
need_quant = ['効率損失量L_秒', 'L算出の可否',
              '即時実行の効率損失量L0_秒', 'L0算出の可否',
              '提供数', '失敗数', 'プレイ時間_秒',
              '指示の回数目', '実際の実行順位', '着手までの秒数']
miss = [c for c in need_quant if c not in srv.QUANT_FIELDS]
check('定量: L・L0・スコアが入っている', not miss, f'足りない: {miss}')

# 7. 条件の欄
# 順序統制のグループも条件の一部。どちらのファイルからも、その行が
# どの割り当ての人のものか分かるようにしておく。
cond = ['記録時刻', '参加者ID', 'パターン', 'グループ', 'セッション番号',
        '地図', '猶予', '注文の組み合わせ番号']
check('条件の欄が2つのファイルで同じ並び',
      srv.QUAL_FIELDS[:len(cond)] == cond
      and srv.QUANT_FIELDS[:len(cond)] == cond,
      f'定性={srv.QUAL_FIELDS[:len(cond)]} 定量={srv.QUANT_FIELDS[:len(cond)]}')

# 5 + 6 + 8. 実際に書いてみる
with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    q = d / 'quant.csv'
    for n in (1, 2, 3):
        srv.append_csv(q, srv.QUANT_FIELDS, {
            '記録時刻': '2026-09-28T14:00:00', '参加者ID': 'p1',
            'パターン': 3, 'セッション番号': 1, '地図': 'exp_ring', '猶予': 0,
            '指示の回数目': n, '指示の総回数': 3, '指示の動作': 'chop',
            '効率損失量L_秒': 0.4 * n, '即時実行の効率損失量L0_秒': 1.2,
            '提供数': 6, '失敗数': 1, 'プレイ時間_秒': 90,
        }, notes=srv.QUANT_NOTES)
    rows = list(csv.DictReader(io.open(q, encoding='utf-8-sig')))
    check('定量: 指示ごとに1行', len([r for r in rows[1:]]) == 3,
          f'{len(rows) - 1} 行')
    check('定量: 1行目が説明の行',
          rows[0]['効率損失量L_秒'].startswith('指示したせいで'),
          rows[0]['効率損失量L_秒'][:30])
    check('定量: 同じセッションの指示が続けて並ぶ',
          [r['指示の回数目'] for r in rows[1:]] == ['1', '2', '3'],
          str([r['指示の回数目'] for r in rows[1:]]))

    # 6. 指示が無かった回
    srv.append_csv(q, srv.QUANT_FIELDS, {
        '記録時刻': '2026-09-28T14:05:00', '参加者ID': 'p1',
        'パターン': 3, 'セッション番号': 2, '地図': 'exp_partition', '猶予': 'inf',
        '指示の回数目': 0, '指示の総回数': 0,
        '提供数': 4, '失敗数': 0, 'プレイ時間_秒': 90,
    }, notes=srv.QUANT_NOTES)
    rows = list(csv.DictReader(io.open(q, encoding='utf-8-sig')))
    last = rows[-1]
    check('指示が無かった回も、スコアだけ残る',
          last['指示の回数目'] == '0' and last['提供数'] == '4',
          f"回数目={last['指示の回数目']} 提供数={last['提供数']}")

    # 8. BOM
    check('Excel で読めるよう BOM が付く',
          q.read_bytes().startswith(b'\xef\xbb\xbf'))

    # 定性も1行書けること
    g = d / 'qual.csv'
    row = {c: '' for c in srv.QUAL_FIELDS}
    row.update({'記録時刻': '2026-09-28T14:10:00', '参加者ID': 'p1',
                'ラポール': 3.5, '指示平均': 4.0,
                '提供数': 6, '失敗数': 1, 'プレイ時間_秒': 90})
    srv.append_csv(g, srv.QUAL_FIELDS, row, notes=srv.QUAL_NOTES)
    rows = list(csv.DictReader(io.open(g, encoding='utf-8-sig')))
    check('定性: 1回答1行で書ける',
          len(rows) == 2 and rows[1]['ラポール'] == '3.5',
          f'{len(rows)} 行')
    check('定性: 1行目が説明の行',
          rows[0]['つながり1'] == 'つながりを感じた', rows[0]['つながり1'])

print()
ok = sum(1 for r in results if r)
if ok == len(results):
    print(f'[SUCCESS] {ok}/{len(results)} 件すべて期待どおり')
else:
    print(f'[FAILURE] {ok}/{len(results)} 件のみ期待どおり')
    sys.exit(1)
