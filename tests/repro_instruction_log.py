"""1回のゲームで出した指示を、全部残せていることの検証。

これまでの問題:
    セッションの記録(web_sessions.csv)は1回1行で、指示の欄には最初の
    1回ぶんしか入らなかった。n個の作業ごとに指示を出す設計では1回の
    ゲームで何度も指示が出る(実測: エンドレス90秒で5回)。残りは
    リプレイにしか残らず、L も挿入順も分析できなかった。

ここで確かめること:
    1. 指示の数だけ行が出る(縦軸が指示ごと)
    2. 1行に、その指示の L・割り込まれた作業数・実際の順位・着手までの
       秒数が入っている
    3. 見出しは日本語で、そのすぐ下に説明の行がある
    4. 取りかからずに終わった指示も、打ち切りとして残る
    5. 同じセッションの指示が続けて並ぶ(一望できる)

実行方法:
    python tests/repro_instruction_log.py
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
import tempfile
from pathlib import Path
from types import SimpleNamespace

import server as srv

results = []


def check(label, ok, detail=""):
    results.append(ok)
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}{(' -> ' + detail) if detail else ''}")


def pending(verb, obj, accepted, started, tasks_before, loss, status):
    return {
        'task': {'verb': verb, 'obj': obj},
        'accepted_env_time': accepted,
        'started_env_time': started,
        'tasks_before': tasks_before,
        'status': status,
        'time_loss': loss,
    }


# 90秒のセッションで3回指示を出した想定。3回目は最後まで取りかからなかった。
PEND = [
    pending('chop', 'onion', 0.0, 2.4, 0,
            {'loss_seconds': 0.0, 'baseline_seconds': 56.2,
             'constrained_seconds': 56.2, 'status': 'ok', 'num_tasks': 11},
            'done'),
    pending('cook', 'lettuce-onion soup', 31.8, 40.2, 2,
            {'loss_seconds': 6.6, 'baseline_seconds': 41.0,
             'constrained_seconds': 47.6, 'status': 'ok', 'num_tasks': 9},
            'done'),
    pending('serve', 'lettuce-onion soup', 70.8, None, None,
            {'status': 'no_constraint'}, 'canceled'),
]
SLOTS = [
    {'verb': 'chop', 'obj': 'onion', 'quality': 'soup', 'rank': 1},
    {'verb': 'cook', 'obj': 'lettuce-onion soup', 'quality': 'soup', 'rank': 5},
    {'verb': 'serve', 'obj': 'lettuce-onion soup', 'quality': 'soup', 'rank': None},
]

tmp = Path(tempfile.mkdtemp()) / 'web_instructions.csv'
srv.INSTRUCTION_LOG_PATH = tmp

s = srv.WebGamePlay.__new__(srv.WebGamePlay)
s.selection = {'participant': 'p01', 'pattern': 2, 'session': 3,
               'map': 'exp_ring', 'skip_budget': 1, 'case': 20}
s.result = {'served': 6, 'failed': 1, 'makespan_s': 90.0}
s.env = SimpleNamespace(_pending_instructions=PEND)
s.instruction_slots = SLOTS
s.game_id = 7
srv.WebGamePlay._log_instructions(s, '')

with io.open(tmp, encoding='utf-8', newline='') as f:
    rows = list(csv.reader(f))
head, note, data = rows[0], rows[1], rows[2:]

# 1
check('指示の数だけ行が出る', len(data) == 3, f'{len(data)} 行')

# 3
check('見出しが日本語', '効率損失量L_秒' in head and '割り込まれた作業数' in head)
check('見出しの下に説明の行がある',
      '指示したせいで伸びた見込み時間' in note[head.index('効率損失量L_秒')],
      note[head.index('効率損失量L_秒')])
check('説明の行はデータではない',
      data[0][head.index('参加者ID')] == 'p01')

d = [dict(zip(head, r)) for r in data]

# 2
check('1回目: L と順位が入っている',
      d[0]['効率損失量L_秒'] == '0.0' and d[0]['実際の実行順位'] == '1'
      and d[0]['指示なしの実行順位'] == '1', str(d[0]['効率損失量L_秒']))
check('2回目: L=6.6 / 割り込まれた作業 2 / 実際は3番目',
      d[1]['効率損失量L_秒'] == '6.6' and d[1]['割り込まれた作業数'] == '2'
      and d[1]['実際の実行順位'] == '3',
      f"L={d[1]['効率損失量L_秒']} 割込={d[1]['割り込まれた作業数']} 順位={d[1]['実際の実行順位']}")
check('2回目: 指示なしなら5番目 -> 2番手ぶん繰り上がった',
      d[1]['指示なしの実行順位'] == '5' and d[1]['繰り上がった順位'] == '2')
check('2回目: 着手までの秒数 = 40.2 - 31.8',
      d[1]['着手までの秒数'] == '8.4', d[1]['着手までの秒数'])

# 4
check('取りかからずに終わった指示も残る',
      d[2]['着手せず終了'] == '1' and d[2]['指示の結末'] == 'canceled',
      f"打ち切り={d[2]['着手せず終了']} 結末={d[2]['指示の結末']}")
check('取りかからなかった回の着手までの秒数は空',
      d[2]['着手までの秒数'] == '')

# 5
check('同じセッションの指示が続けて並ぶ',
      [r['指示の回数目'] for r in d] == ['1', '2', '3']
      and {r['セッション番号'] for r in d} == {'3'})
check('その回の成績も各行に入っている',
      d[0]['提供数'] == '6' and d[0]['プレイ時間_秒'] == '90.0')

ok = sum(1 for r in results if r)
print()
if ok == len(results):
    print(f'[SUCCESS] {ok}/{len(results)} 件が期待どおり')
else:
    print(f'[FAILURE] {ok}/{len(results)} 件のみ期待どおり')
    sys.exit(1)
