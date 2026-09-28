"""即時実行の効率損失量 L(0) の検証。

なぜ要るか:
    効率損失量 L = f'(d) - f は、その回の割り込み許容数 d に依存する。d=1 や 2 の
    回では、最適計画でもともと指示した作業が d 個以内に来ていることが
    多く、縛りが何も効かずに L=0 になる(実測: 出せた7件のうち6件が 0.0)。
    対照条件(inf)に至っては縛り自体が無いので、L は空欄になる。

    そこで、その回の条件に関わらず「いますぐやらせたら何秒損か」を
    必ず出す。盤面と指示だけで決まる量なので、条件をまたいで比べられる。

ここで確かめること:
    1. どの条件でも L(0) が出る(対照条件 inf でも出る)
    2. L(0) は条件によって変わらない(指示と盤面だけで決まる)
    3. 割り込み許容数=0 の回では L と L(0) が一致する
    4. 割り込み許容数が大きい回では L=0 でも L(0) は 0 でないことがある
    5. 指示ごとに L(0) は変わる(区別できる量になっている)
    6. 解き比べるときは、指示が持っている割り込み許容数も差し替える
       (ここを忘れると、割り込み許容数を変えたつもりで元の条件のまま解いてしまう)

実行方法:
    python tests/repro_immediate_loss.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

import inspect

from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from gym_cooking.utils.replay import Replay
from agent.agent.executor.low import EnvState
from agent.agent.myagent.CSPAgent import CSPAgent

results = []


def check(label, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}{(' -> ' + detail) if detail else ''}")


def state_of(env):
    i = env.get_ai_info()
    return EnvState(world=i['world'], agents=i['sim_agents'], agent_idx=0,
                    order=i['order_scheduler'], event_history=i['event_history'],
                    time=i['current_time'], chg_grid=i['chg_grid'])


ORDERS = ('OnionLettuceSalad', 'TomatoLettuceSoup', 'OnionTomatoSalad')


def make(budget):
    env = OvercookedEnvironment(MapSetting(level='exp_ring', order_recipes=ORDERS,
                                           max_num_timesteps=200))
    env.reset()
    ai = CSPAgent(10, Replay(), sc_2agent=True, skip_budget=budget)
    ai.human_counterpart_mode = True
    ai.own_agent_idx = 0
    ai.priority_weights = {}
    ai.gui_text_input = ''
    ai.gui_constraint_input = ''
    ai.active_constraints = []
    ai.time_limit_seconds = 200
    return env, ai


def measure(budget, key):
    env, ai = make(budget)
    cands = {p['verb'] + ' ' + p['obj']: (d, p)
             for d, p in ai.get_instruction_candidates(state_of(env))}
    if key not in cands:
        return None, cands
    d, p = cands[key]
    return ai.estimate_instruction_time_loss(
        state_of(env),
        {'task': (d, p), 'status': 'pending',
         'skip_budget': budget, 'remaining_skip_budget': budget},
        skip_budget=budget), cands


# 最適計画では先頭に来ない作業を選ぶ(縛ると損が出る)
TARGET = 'chop tomato'
out = {}
for budget in (0, 1, 2, None):
    r, cands = measure(budget, TARGET)
    if r is None:
        check(f'指示 {TARGET} が候補にある', False, str(list(cands)))
        break
    out[budget] = r

# 1
check('どの条件でも L(0) が出る',
      all(out[b]['immediate_loss_seconds'] is not None for b in out),
      str({b: out[b]['immediate_loss_seconds'] for b in out}))
check('対照条件(inf)でも L(0) が出る',
      out[None]['immediate_status'] == 'ok'
      and out[None]['immediate_loss_seconds'] is not None,
      f"{out[None]['immediate_status']} {out[None]['immediate_loss_seconds']}")
check('対照条件(inf)では L のほうは空のまま',
      out[None]['status'] == 'no_constraint'
      and out[None]['loss_seconds'] is None,
      f"{out[None]['status']} {out[None]['loss_seconds']}")

# 2
vals = {out[b]['immediate_loss_seconds'] for b in out}
check('L(0) は条件によって変わらない', len(vals) == 1, str(vals))

# 3
check('割り込み許容数=0 の回では L と L(0) が一致',
      out[0]['loss_seconds'] == out[0]['immediate_loss_seconds'],
      f"L={out[0]['loss_seconds']} L0={out[0]['immediate_loss_seconds']}")

# 4
l0 = out[0]['immediate_loss_seconds']
check('この指示は、いますぐやらせると損が出る', l0 and l0 > 0, f'L(0)={l0}')
check('割り込み許容数が大きいと L は 0 になる(L(0) は 0 でない)',
      out[1]['loss_seconds'] == 0.0 and out[1]['immediate_loss_seconds'] > 0,
      f"割り込み許容数1: L={out[1]['loss_seconds']} L0={out[1]['immediate_loss_seconds']}")

# 5
env, ai = make(None)
seen = {}
for d, p in ai.get_instruction_candidates(state_of(env)):
    env2, ai2 = make(None)
    r = ai2.estimate_instruction_time_loss(
        state_of(env2),
        {'task': (d, p), 'status': 'pending',
         'skip_budget': None, 'remaining_skip_budget': None},
        skip_budget=None)
    seen[p['verb'] + ' ' + p['obj']] = r['immediate_loss_seconds']
check('指示ごとに L(0) は変わる(区別できる)', len(set(seen.values())) > 1, str(seen))

# 残り時間に左右されないこと。L は「指示で段取りがどれだけ悪くなったか」で、
# ゲームがあと何秒あるかとは別の話。入れたままだと終盤で壊れる
# (実測: 残り30秒で L=-11.4、残り1秒で目的関数の時間項が 0 に潰れる)。
by_remaining = {}
for remaining in (None, 120, 30, 1):
    env, ai = make(0)
    ai.time_limit_seconds = remaining
    cands = {p['verb'] + ' ' + p['obj']: (d, p)
             for d, p in ai.get_instruction_candidates(state_of(env))}
    d, p = cands[TARGET]
    r = ai.estimate_instruction_time_loss(
        state_of(env),
        {'task': (d, p), 'status': 'pending',
         'skip_budget': 0, 'remaining_skip_budget': 0},
        skip_budget=0)
    by_remaining[remaining] = (r['loss_seconds'], r['status'])
check('残り時間に関係なく L は同じ',
      len({v[0] for v in by_remaining.values()}) == 1, str(by_remaining))
check('残り1秒でも L は負にならない',
      (by_remaining[1][0] or 0) >= 0, str(by_remaining[1]))
check('残り時間のせいで品数が変わった扱いにならない',
      all(v[1] == 'ok' for v in by_remaining.values()), str(by_remaining))

# 6
SRC = inspect.getsource(CSPAgent.estimate_instruction_time_loss)
check('L の計測では、残り時間の制約も外している',
      'probe.time_limit_seconds = None' in SRC,
      '残したままだと終盤で L が壊れる')
check('解き比べるとき、指示が持つ割り込み許容数も差し替えている',
      "_p['skip_budget'] = budget" in SRC
      and "_p['remaining_skip_budget'] = budget" in SRC,
      '差し替えていない: 元の条件のまま解いてしまう')

print()
ok = sum(1 for r in results if r)
if ok == len(results):
    print(f'[SUCCESS] {ok}/{len(results)} 件すべて期待どおり')
else:
    print(f'[FAILURE] {ok}/{len(results)} 件のみ期待どおり')
    sys.exit(1)
