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


# 1. 並びが決まりどおりか
g1 = [(c['map'], c['skip_budget']) for c in design.plan_for('p01')]
want_g1 = [('exp_ring', 0), ('exp_ring', 1), ('exp_ring', 'inf'), ('exp_ring', 2),
           ('exp_partition', 0), ('exp_partition', 1), ('exp_partition', 'inf'), ('exp_partition', 2)]
check('G1: ring先・行A', g1 == want_g1, str(g1))

g6 = [(c['map'], c['skip_budget']) for c in design.plan_for('p06')]
want_g6 = [('exp_partition', 1), ('exp_partition', 2), ('exp_partition', 0), ('exp_partition', 'inf'),
           ('exp_ring', 1), ('exp_ring', 2), ('exp_ring', 0), ('exp_ring', 'inf')]
check('G6: partition先・行B', g6 == want_g6, str(g6))

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
# Williams 方格: 直前の条件の組も釣り合う
pairs = Counter()
for r in design.LATIN_SQUARE.values():
    for a, b in zip(r, r[1:]):
        pairs[(str(a), str(b))] += 1
check('直前の条件の組がすべて1回ずつ(Williams 方格)',
      len(pairs) == len(design.BUDGETS) * (len(design.BUDGETS) - 1)
      and set(pairs.values()) == {1}, str(sorted(pairs.items())))

# 4. 地図のまとまりと半々
B = design.BLOCK
for pid in (f'p{i:02d}' for i in range(1, 17)):
    plan = design.plan_for(pid)
    maps = [c['map'] for c in plan]
    if not (len(set(maps[:B])) == 1 and len(set(maps[B:])) == 1
            and maps[0] != maps[B]):
        check(f'{pid}: 地図がひとまとまり', False, str(maps))
        break
else:
    check(f'16人とも、地図が{B}回ずつのひとまとまり', True)

first = Counter(design.plan_for(f'p{i:02d}')[0]['map'] for i in range(1, 17))
check('最初の地図が半々', first['exp_ring'] == first['exp_partition'] == 8,
      str(dict(first)))

# 5. 各セッション位置で均等
by_pos = {i: Counter() for i in range(1, design.TOTAL + 1)}
for i in range(1, 17):
    for c in design.plan_for(f'p{i:02d}'):
        by_pos[c['session']][(c['map'], str(c['skip_budget']))] += 1
want_n = len(design.MAPS) * len(design.BUDGETS)
bad = [s for s, cnt in by_pos.items()
       if len(cnt) != want_n or len(set(cnt.values())) != 1]
check('16人で、各セッション位置の条件が均等', not bad, f'偏り: {bad}')

# 6. skip_budget × 注文構成
PRESET = {'exp_ring': 'experiment1', 'exp_partition': 'experiment2'}


def case_of(map_name, position):
    cases = experiment_case_indices(PRESET[map_name]) or [0]
    return design.fixed_case_for(PRESET[map_name], cases)


pair = Counter()
for i in range(1, 17):
    for c in design.plan_for(f'p{i:02d}'):
        pair[(str(c['skip_budget']), case_of(c['map'], c['position']))] += 1
check('(skip_budget × 注文構成) が均等', len(set(pair.values())) == 1,
      str(sorted(set(pair.values()))))

# 注文構成は地図ごとに1つ(条件が違っても同じ注文)
for m in PRESET:
    cs = {case_of(m, pos) for pos in range(1, design.BLOCK + 1)}
    check(f'{m}: 注文構成はどの位置でも同じ', len(cs) == 1, str(cs))
    check(f'{m}: 固定した構成は良い指示が決まる候補の中にある',
          next(iter(cs)) in (experiment_case_indices(PRESET[m]) or []), str(cs))

# 7. グループの指定と、連番でないID
check('グループを指定できる',
      design.group_of('yamada', 'G3')['name'] == 'G3')
check('G を付けなくてもよい', design.group_of('yamada', 3)['name'] == 'G3')
check('連番でないIDでも、同じIDなら同じグループ',
      design.group_of('yamada')['name'] == design.group_of('yamada')['name'])
try:
    design.group_of('x', 99)
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
check('本番の注文構成は地図ごとに固定(サーバーが fixed_case_for を使う)',
      'case = design.fixed_case_for(preset, cases)' in SRC2,
      '位置やくじ引きで決めたまま')
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
