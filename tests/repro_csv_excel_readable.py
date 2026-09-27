"""記録の CSV が Excel でそのまま読めることの検証。

これまでの問題:
    CSV を BOM なしの UTF-8 で書いていた。日本語 Windows の Excel は
    CSV を CP932 として読むので、日本語の見出しと説明行が
    「險倬鹸譎ょ綾 蜿ょ刈閠・」のように崩れた(報告あり)。

ここで確かめること:
    1. 新しく作った CSV の先頭に BOM がある
    2. 足した行の前には BOM が入らない(1回だけ)
    3. CP932 で読み直しても日本語が崩れない = Excel で読める
    4. 項目が変わったかの判定が BOM に引きずられない
       (BOM を見出しの一部と誤読すると、毎回ファイルを退避してしまう)
    5. BOM の無い古いファイルへ足しても、途中に BOM が混ざらない

実行方法:
    python tests/repro_csv_excel_readable.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'agent'))

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

import csv
import tempfile
from pathlib import Path

import server as srv

results = []


def check(name, ok, detail=''):
    results.append((name, bool(ok), detail))
    print(f'  [{"OK  " if ok else "FAIL"}] {name}' + (f'\n         -> {detail}' if detail and not ok else ''))


BOM = b'\xef\xbb\xbf'
FIELDS = ['記録時刻', '指示の動作', '効率損失量L_秒']
NOTES = {'記録時刻': 'この行を書いた日時', '指示の動作': 'chop=切る / cook=煮る',
         '効率損失量L_秒': "指示のせいで伸びた見込み時間"}
ROW = {'記録時刻': '2026-09-27 21:10:43', '指示の動作': 'chop', '効率損失量L_秒': '3.2'}

with tempfile.TemporaryDirectory() as d:
    d = Path(d)

    # 1-3. 新しいファイル
    p = d / 'new.csv'
    srv.append_csv(p, FIELDS, ROW, NOTES)
    srv.append_csv(p, FIELDS, dict(ROW, **{'指示の動作': 'cook'}))
    raw = p.read_bytes()

    check('新しい CSV の先頭に BOM がある', raw.startswith(BOM))
    check('BOM は1つだけ(足した行の前に入らない)', raw.count(BOM) == 1,
          f'{raw.count(BOM)} 個ある')

    # Excel(CP932)として読めるか。BOM があれば Excel は UTF-8 と判断するので、
    # ここでは「BOM を外した中身が UTF-8 として正しい」ことを確かめる。
    body = raw[len(BOM):] if raw.startswith(BOM) else raw
    try:
        text = body.decode('utf-8')
        ok_utf8 = True
    except UnicodeDecodeError as e:
        text, ok_utf8 = '', str(e)
    check('中身は UTF-8 として正しい', ok_utf8 is True, str(ok_utf8))

    # 崩れの再現: BOM が無いと CP932 読みで別の字になる(= 以前の症状)
    try:
        garbled = body.decode('cp932')
    except UnicodeDecodeError:
        garbled = ''
    check('BOM が無ければ崩れる状況だった(症状の再現)',
          '記録時刻' not in garbled,
          'CP932 で読んでも崩れない = この検証が意味を持たない')

    rows = list(csv.reader(p.open(encoding='utf-8-sig', newline='')))
    check('utf-8-sig で読むと見出しがそのまま取れる',
          rows and rows[0] == FIELDS, f'見出し: {rows[0] if rows else None}')
    check('見出しのすぐ下が説明の行', len(rows) > 1 and rows[1][0] == NOTES['記録時刻'],
          f'2行目: {rows[1] if len(rows) > 1 else None}')
    check('記録の行がある', len(rows) == 4, f'{len(rows)} 行')

    # 4. 退避が起きないこと
    before = sorted(x.name for x in d.iterdir())
    srv.append_csv(p, FIELDS, ROW, NOTES)
    after = sorted(x.name for x in d.iterdir())
    check('項目が同じなら退避しない(BOM を見出しと誤読しない)', before == after,
          f'増えた: {set(after) - set(before)}')

    # 項目が変わったときは今までどおり退避する
    srv.append_csv(p, FIELDS + ['提供数'], ROW, NOTES)
    check('項目が変われば今までどおり退避する',
          len([x for x in d.iterdir() if x.name.startswith('new-')]) == 1,
          f'{[x.name for x in d.iterdir()]}')

    # 5. BOM の無い古いファイルへ足す
    old = d / 'old.csv'
    with old.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerow(ROW)
    srv.append_csv(old, FIELDS, ROW, NOTES)
    raw_old = old.read_bytes()
    check('古いファイルへ足しても途中に BOM が混ざらない', BOM not in raw_old,
          '混ざっている')
    check('古いファイルは退避されない(見出しが同じ)',
          len(list(csv.reader(old.open(encoding='utf-8-sig', newline='')))) == 3,
          '行数が合わない')

print()
ng = [n for n, ok, _ in results if not ok]
if ng:
    print(f'[FAILURE] {len(results) - len(ng)}/{len(results)} 件のみ期待どおり')
    sys.exit(1)
print(f'[SUCCESS] {len(results)}/{len(results)} 件すべて期待どおり')
