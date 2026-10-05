"""複数人が同時に遊んでも、CSV を Excel で開いていても、記録が欠けないことの検証。

席の数だけサーバー(プロセス)が動き、同じ記録ファイルへ書く。確かめること:
    1. 4つのプロセスが同時に足しても、行が欠けない・崩れない
    2. CSV を Excel で開いている間に足した行は、閉じたあと本体へ戻る
    3. 参加者番号が重ならない
         - 名簿を Excel で開いている間に2人が同意しても
         - 2つのプロセスが同時に採番しても
    4. Excel で開いている間に終わったゲームも、統合ファイルの突き合わせで見つかる
    5. 割り当て(assignments.json)が読めないとき、空として上書きしない

本物の記録には触れない(一時フォルダへ向け直して動かす)。

実行方法:
    python tests/repro_concurrent_records.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'agent'))

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

import csv
import ctypes
import json
import subprocess
import tempfile
from pathlib import Path

results = []


def check(label, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}{(' -> ' + detail) if detail else ''}", flush=True)


class ExcelOpen:
    """Excel が開いているのと同じ状態にする(他からは読めるが書けない)。"""
    GENERIC_READ, GENERIC_WRITE = 0x80000000, 0x40000000
    FILE_SHARE_READ, OPEN_EXISTING = 0x1, 3

    def __init__(self, path):
        self.path = str(path)
        self.h = None

    def __enter__(self):
        k = ctypes.windll.kernel32
        k.CreateFileW.restype = ctypes.c_void_p
        self.h = k.CreateFileW(self.path, self.GENERIC_READ | self.GENERIC_WRITE,
                               self.FILE_SHARE_READ, None, self.OPEN_EXISTING, 0, None)
        assert self.h not in (None, ctypes.c_void_p(-1).value), '開けませんでした'
        return self

    def __exit__(self, *exc):
        ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(self.h))
        return False


def data_rows(path, key):
    """説明行を除いた記録の行。"""
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        return [r for r in csv.DictReader(f) if str(r.get(key, '')).startswith(('w', 'p', '20'))]


WORKER = r'''
import os, sys
sys.path.insert(0, r"{root}"); sys.path.insert(0, r"{root}\testbed-cooking"); sys.path.insert(0, r"{root}\agent")
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
from pathlib import Path
import server as srv
mode, who, tmp = sys.argv[1], sys.argv[2], Path(sys.argv[3])
if mode == 'append':
    for n in range(60):
        srv.append_csv(tmp / 'shared.csv', ['id', 'n', 'text'],
                       {{'id': 'w' + who, 'n': n, 'text': 'あ,い\n"う"' * 3}}, {{'id': '説明'}})
else:
    srv.ROSTER_PATH = tmp / 'participant_roster.csv'
    srv.ASSIGN_PATH = tmp / 'assignments.json'
    for n in range(15):
        print(srv.register_participant('名前%s_%d' % (who, n), False), flush=True)
'''


def run_workers(mode, tmp, count=4):
    code = WORKER.format(root=REPO_ROOT)
    procs = [subprocess.Popen([sys.executable, '-c', code, mode, str(i), str(tmp)],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              text=True, encoding='utf-8', errors='replace')
             for i in range(count)]
    return [p.communicate()[0] for p in procs]


def main():
    if os.name != 'nt':
        print('Windows 専用の検証です')
        return 0
    import server as srv
    tmp = Path(tempfile.mkdtemp(prefix='records_'))
    srv.ASSIGN_PATH = tmp / 'assignments.json'   # 本物の割り当ては読まない

    print('[1] 4つのプロセスが同時に足す')
    run_workers('append', tmp)
    rows = data_rows(tmp / 'shared.csv', 'id')
    check('行が欠けない(4 x 60)', len(rows) == 240, f'{len(rows)} 行')
    per = {w: sorted(int(r['n']) for r in rows if r['id'] == w) for w in ('w0', 'w1', 'w2', 'w3')}
    check('どのプロセスの分も 0〜59 がそろう', all(v == list(range(60)) for v in per.values()))
    check('改行やカンマを含む欄が崩れない', all(r['text'] == 'あ,い\n"う"' * 3 for r in rows))

    print('[2] CSV を Excel で開いている間に足す')
    p = tmp / 'excel.csv'
    F = ['id', 'v']
    srv.append_csv(p, F, {'id': 'p1', 'v': 1}, {'id': '説明'})
    with ExcelOpen(p):
        ok = srv.append_csv(p, F, {'id': 'p2', 'v': 2}, {'id': '説明'})
        check('開いている間は本体に書けない(症状の再現)', ok is False)
        check('そのぶんは隣のファイルへ逃がしてある', (tmp / 'excel-pending.csv').exists())
    srv.append_csv(p, F, {'id': 'p3', 'v': 3}, {'id': '説明'})
    ids = [r['id'] for r in data_rows(p, 'id')]
    check('閉じたあと、逃がした行が本体へ戻る', ids == ['p1', 'p2', 'p3'], str(ids))

    print('[2b] 項目が増えた直後に Excel で開いていた場合')
    p = tmp / 'grown.csv'
    srv.append_csv(p, ['id', 'v'], {'id': 'p1', 'v': 1}, {'id': '説明'})
    with ExcelOpen(p):
        srv.append_csv(p, ['id', 'v', 'w'], {'id': 'p2', 'v': 2, 'w': 9}, {'id': '説明'})
    srv.append_csv(p, ['id', 'v', 'w'], {'id': 'p3', 'v': 3, 'w': 9}, {'id': '説明'})
    srv._drain_pending(p)
    ids = sorted(r['id'] for r in data_rows(p, 'id'))
    check('新しい項目のファイルに、逃がした行も入る', ids == ['p2', 'p3'], str(ids))
    old = [q for q in tmp.glob('grown-2*.csv')]
    check('古い項目のぶんは退避されて残る', len(old) == 1 and
          [r['id'] for r in data_rows(old[0], 'id')] == ['p1'])

    print('[3] 参加者番号が重ならない')
    srv.ROSTER_PATH = tmp / 'participant_roster.csv'
    a = srv.register_participant('一人目', False)
    with ExcelOpen(srv.ROSTER_PATH):
        b = srv.register_participant('二人目', False)
        c = srv.register_participant('三人目', False)
    d = srv.register_participant('四人目', False)
    check('名簿を Excel で開いている間に同意しても重ならない',
          len({a, b, c, d}) == 4, f'{a} {b} {c} {d}')
    srv._drain_pending(srv.ROSTER_PATH)
    names = {r['参加者番号']: r['お名前'] for r in srv._read_roster()}
    check('閉じたあと、名簿に全員そろう', len(names) == 4, str(names))

    (tmp / 'participant_roster.csv').unlink()
    out = run_workers('ids', tmp)
    got = [x for o in out for x in o.split()]
    check('4つのプロセスが同時に採番しても重ならない(4 x 15)',
          len(got) == 60 and len(set(got)) == 60, f'{len(got)} 個中 {len(set(got))} 種類')

    print('[4] Excel で開いている間に終わったゲームも突き合わせで見つかる')
    srv.SESSION_LOG_PATH = tmp / 'web_sessions.csv'
    row = {'timestamp': '2026-01-01T00:00:00', 'participant_id': 'p01', 'session': 1,
           'instruction_confidence': 4}
    srv.append_csv(srv.SESSION_LOG_PATH, srv.SESSION_JA_FIELDS, srv.session_row_ja(row),
                   srv.SESSION_NOTES)
    with ExcelOpen(srv.SESSION_LOG_PATH):
        row2 = dict(row, participant_id='p02', instruction_confidence=5)
        srv.append_csv(srv.SESSION_LOG_PATH, srv.SESSION_JA_FIELDS, srv.session_row_ja(row2),
                       srv.SESSION_NOTES)
        found = {r.get('participant_id'): r.get('instruction_confidence')
                 for r in srv._read_sessions()}
        check('逃がしてある回も読める', found == {'p01': '4', 'p02': '5'}, str(found))

    print('[5] 割り当てが読めないとき、空として上書きしない')
    srv.ASSIGN_PATH = tmp / 'assignments.json'
    srv._save_assignments({'p01': {'order': [], 'done': 3}})
    check('書いたものが読める', srv._load_assignments() == {'p01': {'order': [], 'done': 3}})
    srv.ASSIGN_PATH.write_text('{"p01": {"order": [], "do', encoding='utf-8')   # 途中で切れた形
    try:
        srv._load_assignments()
        raised = False
    except Exception:
        raised = True
    check('壊れているときは空を返さず、止まる', raised)
    srv.ASSIGN_PATH.unlink()
    check('ファイルが無いだけなら空', srv._load_assignments() == {})

    print()
    print(f"[{'SUCCESS' if all(results) else 'FAIL'}] "
          f"{results.count(True)}/{len(results)} 件が期待どおり")
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
