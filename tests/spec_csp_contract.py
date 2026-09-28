"""CSP の仕様を固定する。

不具合を1つ直すたびに例外処理を足していくと、いつの間にか「CSP として
何を解いているのか」が変わってしまう。ここは、その形を書き留めて
見張るための場所。落ちたら「仕様を変えた」ということなので、意図した
変更なら、この表のほうを直してから進める。

見張っているもの:
    1. 目的関数の形
    2. 張られる制約の種類
    3. skip_budget の意味
    4. 番人(諦めたタスク)が、計画の制約へ漏れていないか
    5. 探索の上限が入っていること(遊ぶとき)と、L の計測では外すこと

実行方法:
    python tests/spec_csp_contract.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

import inspect

from gym_cooking.utils.replay import Replay
from agent.agent.myagent.CSPAgent import CSPAgent

results = []


def check(label, ok, detail=""):
    results.append(ok)
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}{(' -> ' + detail) if detail else ''}")


SRC = inspect.getsource(CSPAgent.solve_csp_scheduling)

print('--- 1. 目的関数 ---')
# makespan を辞書式に優先する。重みはタスク数 × 1000。
check('makespan の重みはタスク数×1000',
      'weight_makespan = num_tasks * 1000' in SRC)
check('時間の項は「時間*重み + 各タスクの終了時刻の和」',
      'objective = time_term * weight_makespan + end_sum' in SRC
      and 'model.Minimize(objective)' in SRC)

# 残り時間が迫ったとき、makespan だけでは「間に合わない品に手を付ける」
# 計画が最適になってしまう。出せる品数を第一に、そのうえで早く終える。
check('出せない品数が、時間より上の優先度で入っている',
      'missed_terms' in SRC and 'weight_missed' in SRC,
      '品数を最優先にする項が無い')
check('品数の重みは、時間の項が取り得る最大値より大きい',
      'weight_missed = ' in SRC and 'weight_makespan' in SRC
      and 'horizon' in SRC)
check('「間に合う」は、残り時間との比較で決める(時計の履歴では決めない)',
      'remaining_frames' in SRC and 'in_time' in SRC)
check('残り時間が分からないときは、これまでどおり makespan だけで解く',
      'if remaining_frames is None' in SRC)
check('余り時間は「出せる品の中で一番遅い終わり」を早めて最大化する',
      'served_makespan' in SRC)
check('担当替えの罰は、同点のときだけ効く形で足す',
      'switch_scale = len(switch_penalty_terms) + 1' in SRC
      and 'model.Minimize(objective * switch_scale + switch_penalty)' in SRC)
check('締切遅れは罰(ハード制約にしない)',
      'order_late_terms' in SRC and 'sum(order_late_terms) * (weight_makespan * 10)' in SRC
      and 'model.Add(_late >= finals[0][\'end\'] - _dl)' in SRC)

print('--- 2. 制約の種類 ---')
for label, mark in (
        ('鍋の占有(入れてから取り出すまで重ねない)', 'pot_usage_intervals'),
        ('ミキサーの占有', 'blender_usage_intervals'),
        ('まな板の占有', 'cutboard_intervals'),
        ('移動時間(同じ人が続けてやるなら距離ぶん空ける)', 'AddNoOverlap'),
        ('工程の前後関係(切る→煮る→出す)', 'cooking_frames'),
):
    check(f'{label} がある', mark in SRC)
check('占有は AddNoOverlap で表す(数を数える形にしない)',
      SRC.count('model.AddNoOverlap') >= 3 and 'AddCumulative' not in SRC)

print('--- 3. skip_budget の意味 ---')
SKIP = inspect.getsource(CSPAgent._apply_instruction_skip_budget_constraints)
check('「指示の作業より前に走る、前提でない作業の数は、割り込み許容数まで」',
      'budget_bound' in SKIP and 'OnlyEnforceIf' in SKIP)
check('前提の工程は割り込み許容数に数えない',
      '_dependency_ids_of' in SKIP or '依存' in SKIP)
check('同じ (動作,対象) が複数あるなら、どれか1つが守れればよい',
      '_find_group_task_indices' in SKIP)

print('--- 4. 番人が計画へ漏れていないか ---')
# _assignable_agents は「その作業をどの人に割り当てられるか」を返し、
# その結果が model.Add(is_a1[i] == ...) として CSP の割当制約になる
# (仕切りのある地図のとき)。ここで blocked_tasks を見ると、
# 「6秒進まなかった」という実時間の履歴が 30 秒ぶん制約として残る。
# CSP は盤面の状態から解くものなので、時計の都合は入れたくない。
ASSIGN = inspect.getsource(CSPAgent._assignable_agents)
check('割当を決めるところが、CSP の制約になっている(前提の確認)',
      'model.Add(is_a1[i] == next(iter(allowed)))' in SRC
      and '_assignable_agents' in SRC)
check('割当制約は、時計で減る blocked_tasks を見ていない',
      'blocked_tasks' not in ASSIGN,
      '見ている: 実時間の履歴が 30 秒ぶん制約に残る' if 'blocked_tasks' in ASSIGN else '')

print('--- 5. 探索の上限 ---')
ai = CSPAgent(10, Replay(), sc_2agent=True, skip_budget=0)
check('遊ぶときは上限が入っている',
      ai.solve_deterministic_limit is not None and ai.solve_deterministic_limit > 0,
      str(ai.solve_deterministic_limit))
check('上限は決定性時間(秒で切らない)',
      'max_deterministic_time' in SRC and 'max_time_in_seconds' not in SRC)
check('探索は1本(同点の解が実行のたびに変わらないように)',
      'num_search_workers = 1' in SRC and 'random_seed = 0' in SRC)
loss = ''
for name, fn in vars(CSPAgent).items():
    if callable(fn):
        try:
            body = inspect.getsource(fn)
        except Exception:
            continue
        if 'probe = _dcopy(self)' in body:
            loss = body
            break
check('L の計測では上限を外す(f と f\' の両方が最適解でないと意味がない)',
      'probe.solve_deterministic_limit = None' in loss)

ok = sum(1 for r in results if r)
print()
if ok == len(results):
    print(f'[SUCCESS] {ok}/{len(results)} 件が期待どおり')
else:
    print(f'[FAILURE] {ok}/{len(results)} 件のみ期待どおり')
    sys.exit(1)
