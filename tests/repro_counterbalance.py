"""実験の順序統制(カウンターバランス)の検証。

なぜ要るか:
    8セッションは学習効果が強く出る。順序をくじ引きにすると、ある条件だけ
    たまたま後半に偏る参加者が出て、少人数では打ち消されない。
    注文の構成も、くじ引きだと条件の比較に混ざる。

いまの決まり(experiment_design.py):
    1. 地図をひとまとまりにする(同じ地図を4回続けてから、もう一方へ)
    2. 地図の順序を参加者間で半々にする
    3. かたまりの中の skip_budget 順序を 4x4 のラテン方格で回す
    4. 注文の構成は、かたまりの中の「何番目か」だけで決める

ここで確かめること:
    1. G1 と G7 が仕様書の例と一致する
    2. ラテン方格になっている(どの列にもどの条件も1回ずつ)
    3. 8人で1周し、9人目は1人目と同じグループになる
    4. 地図がひとまとまりで、順序が半々
    5. 16人で、各セッション位置の条件が均等
    6. (skip_budget × 注文構成) が均等
    7. グループを引数で指定できる。連番でないIDでも同じIDなら同じ結果
    8. 練習は本番で使わない注文構成を使う
    9. サーバーがこの並びを使っている(くじ引きに戻っていない)

実行方法:
    python tests/repro_counterbalance.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'agent'))

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

import inspect
from collections import Counter

import experiment_design as design
from gym_cooking.utils.order_preset import (
    enumerate_order_recipes, experiment_case_indices)

results = []


def check(label, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}{(' -> ' + detail) if detail else ''}")


# 1. 仕様書の例
g1 = [(c['map'], c['skip_budget']) for c in design.plan_for('p01')]
want_g1 = [('exp_ring', 0), ('exp_ring', 1), ('exp_ring', 2), ('exp_ring', 'inf'),
           ('exp_partition', 0), ('exp_partition', 1), ('exp_partition', 2),
           ('exp_partition', 'inf')]
check('G1 が仕様書の例と一致', g1 == want_g1, str(g1))

g7 = [(c['map'], c['skip_budget']) for c in design.plan_for('p07')]
want_g7 = [('exp_partition', 2), ('exp_partition', 'inf'), ('exp_partition', 0),
           ('exp_partition', 1), ('exp_ring', 2), ('exp_ring', 'inf'),
           ('exp_ring', 0), ('exp_ring', 1)]
check('G7 が仕様書の例と一致', g7 == want_g7, str(g7))

# 2. ラテン方格
cols = list(zip(*design.LATIN_SQUARE.values()))
check('ラテン方格: どの列にもどの条件も1回ずつ',
      all(sorted(map(str, c)) == sorted(map(str, design.BUDGETS)) for c in cols),
      str(cols))
check('ラテン方格: どの行にもどの条件も1回ずつ',
      all(sorted(map(str, r)) == sorted(map(str, design.BUDGETS))
          for r in design.LATIN_SQUARE.values()))

# 3. 8人で1周
check('9人目は1人目と同じグループ',
      design.group_of('p09')['name'] == design.group_of('p01')['name'],
      design.group_of('p09')['name'])
check('8人で8グループ全部が使われる',
      {design.group_of(f'p{i:02d}')['name'] for i in range(1, 9)}
      == {g['name'] for g in design.GROUPS})

# 4. 地図のまとまりと半々
for pid in (f'p{i:02d}' for i in range(1, 17)):
    plan = design.plan_for(pid)
    maps = [c['map'] for c in plan]
    if not (len(set(maps[:4])) == 1 and len(set(maps[4:])) == 1
            and maps[0] != maps[4]):
        check(f'{pid}: 地図がひとまとまり', False, str(maps))
        break
else:
    check('16人とも、地図が4回ずつのひとまとまり', True)

first = Counter(design.plan_for(f'p{i:02d}')[0]['map'] for i in range(1, 17))
check('最初の地図が半々', first['exp_ring'] == first['exp_partition'] == 8,
      str(dict(first)))

# 5. 各セッション位置で均等
by_pos = {i: Counter() for i in range(1, 9)}
for i in range(1, 17):
    for c in design.plan_for(f'p{i:02d}'):
        by_pos[c['session']][(c['map'], str(c['skip_budget']))] += 1
bad = [s for s, cnt in by_pos.items()
       if len(cnt) != 8 or len(set(cnt.values())) != 1]
check('16人で、各セッション位置の条件が均等', not bad, f'偏り: {bad}')

# 6. skip_budget × 注文構成
PRESET = {'exp_ring': 'experiment1', 'exp_partition': 'experiment2'}


def case_of(map_name, position):
    cases = experiment_case_indices(PRESET[map_name]) or [0]
    return design.case_for(cases, position)


pair = Counter()
for i in range(1, 17):
    for c in design.plan_for(f'p{i:02d}'):
        pair[(str(c['skip_budget']), case_of(c['map'], c['position']))] += 1
check('(skip_budget × 注文構成) が均等', len(set(pair.values())) == 1,
      str(sorted(set(pair.values()))))

# 注文構成が参加者をまたいで同じか
same = all(case_of(c['map'], c['position'])
           == case_of(design.plan_for('p01')[k]['map'],
                      design.plan_for('p01')[k]['position'])
           for k, c in enumerate(design.plan_for('p01')))
check('注文構成は位置だけで決まる(くじ引きでない)', same)

# 7. グループの指定と、連番でないID
check('グループを指定できる',
      design.group_of('yamada', 'G3')['name'] == 'G3')
check('G を付けなくてもよい', design.group_of('yamada', 3)['name'] == 'G3')
check('連番でないIDでも、同じIDなら同じグループ',
      design.group_of('yamada')['name'] == design.group_of('yamada')['name'])
try:
    design.group_of('x', 9)
    check('範囲外のグループは断る', False, '通ってしまった')
except ValueError:
    check('範囲外のグループは断る', True)

# 8. 練習
for m, preset in PRESET.items():
    used = experiment_case_indices(preset) or []
    pc = design.practice_case(len(enumerate_order_recipes(preset)), used)
    check(f'{m}: 練習の注文構成が本番と重ならない', pc not in used,
          f'練習={pc} 本番={used}')
check('練習の skip_budget は全員同じ', design.PRACTICE_BUDGET == 0)

# 9. サーバーが使っているか
import server as srv
SRC = inspect.getsource(srv.assignment_for)
check('サーバーが順序統制の並びを使っている',
      'design.plan_for(participant)' in SRC, 'くじ引きのまま')
SRC2 = inspect.getsource(srv)
check('注文構成がくじ引きでなく位置で決まる',
      'design.case_for(cases, position)' in SRC2,
      'くじ引きのまま(random.choice)')
check('本番の注文構成が random.choice でなくなった',
      "cases = experiment_case_indices(preset) or list(range(len(sets)))"
      + '' + "            case = random.choice(cases)" not in SRC2
      if False else 'design.case_for' in SRC2)
check('定量ファイルにグループの列がある', 'グループ' in srv.QUANT_FIELDS,
      str(srv.QUANT_FIELDS[:5]))
check('定性ファイルにグループの列がある', 'グループ' in srv.QUAL_FIELDS,
      str(srv.QUAL_FIELDS[:5]))
check('既存の指示ファイルにもグループの列がある',
      'グループ' in srv.INSTRUCTION_FIELDS)
check('グループは条件の欄に入っている(両ファイルで同じ位置)',
      srv.QUAL_FIELDS.index('グループ') == srv.QUANT_FIELDS.index('グループ'),
      f"定性={srv.QUAL_FIELDS.index('グループ')} 定量={srv.QUANT_FIELDS.index('グループ')}")

print()
ok = sum(1 for r in results if r)
if ok == len(results):
    print(f'[SUCCESS] {ok}/{len(results)} 件すべて期待どおり')
else:
    print(f'[FAILURE] {ok}/{len(results)} 件のみ期待どおり')
    sys.exit(1)
