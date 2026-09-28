"""残り時間が迫ったら、出せる品数を最大にすることの検証。

これまでの問題:
    目的関数は makespan(全部終わるまでの時間)の最小化だけだった。
    残り時間を知らないので、残りが 10 秒でも「鍋に材料を入れて煮る」
    計画が最適になりうる。実際には 1 品も増えないまま終わる。

いまの仕様:
    1. 出せる品数を第一に置く(時間より上の優先度)
    2. そのうえで「出せる品の中で一番遅い終わり」を早める(= 余り時間を最大に)
    3. 残り時間が分からないときは、これまでどおり makespan だけで解く
    4. 間に合わない注文も計画からは消さない(順番が後ろへ回るだけ)
       消すと makespan の見積もりから消え、L の意味が変わる
    5. 指示された注文は、他を全部落とすより重く扱う。ただしハード制約には
       しない(間に合わない指示で解が無くならないように)

実行方法:
    python tests/repro_deadline_max_dishes.py
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


def new_ai(limit=None):
    ai = CSPAgent(10, Replay(), sc_2agent=True, skip_budget=0)
    ai.human_counterpart_mode = True
    ai.own_agent_idx = 0
    ai.priority_weights = {}
    ai.gui_text_input = ''
    ai.gui_constraint_input = ''
    ai.active_constraints = []
    ai.time_limit_seconds = limit
    ai.fixed_objective = False   # この式そのものを試すテストなので有効にする
    return ai


def fresh(orders=('OnionTomatoSoup', 'TomatoLettuceSalad', 'OnionLettuceSalad')):
    env = OvercookedEnvironment(MapSetting(level='exp_ring', order_recipes=orders,
                                           max_num_timesteps=200))
    env.reset()
    return env


# 3. 制限時間が分からないときは、これまでどおり
env = fresh()
ai = new_ai(limit=None)
st = state_of(env)
check('制限時間が分からなければ None を返す', ai._remaining_frames(st) is None,
      str(ai._remaining_frames(st)))
ai(st)
base = dict(ai._last_solve_metrics or {})
check('その場合でも解ける', base.get('makespan_frames') is not None, str(base))

# 残り時間の計算
ai90 = new_ai(limit=90)
check('残り時間は 制限時間 - 盤面の時刻',
      ai90._remaining_frames(st) == int(90 * ai90.fps),
      str(ai90._remaining_frames(st)))

# 4. 間に合わない注文も計画から消えない
ai_short = new_ai(limit=8)          # どう頑張っても 3 品は無理
env2 = fresh()
st2 = state_of(env2)
ai_short(st2)
sched = [t.get('id') for lst in (ai_short.schedule_per_agent or {}).values()
         for t in lst]
orders_in_plan = {tid[2] for tid in sched if tid}
check('間に合わなくても、注文は計画に残っている', len(orders_in_plan) == 3,
      f'計画に残った注文: {sorted(orders_in_plan)}')
m = dict(ai_short._last_solve_metrics or {})
check('makespan は全部終わるまでの時間のまま(L の土台を壊さない)',
      m.get('makespan_frames') is not None
      and m['makespan_frames'] > int(8 * ai_short.fps),
      f"makespan={m.get('makespan_frames')} 残り={int(8 * ai_short.fps)}")

# 1+2. 目的関数の形
SRC = inspect.getsource(CSPAgent.solve_csp_scheduling)
check('品数の重みは、時間の項の最大値より大きい',
      'weight_missed = (' in SRC and 'horizon * weight_makespan' in SRC
      and 'sum(missed_terms) * weight_missed' in SRC)
check('縮めるのは「出せる品の中で一番遅い終わり」',
      "time_term = served_makespan if served_makespan is not None else makespan" in SRC)
check('「間に合う」は残り時間との比較で決める',
      'model.Add(_end <= remaining_frames).OnlyEnforceIf(in_time)' in SRC)

# 5. 指示された注文の扱い
check('指示された注文は、他を全部落とすより重い',
      'weight = (n_orders + 1) if _uid in instructed_uids else 1' in SRC)
check('指示はハード制約にしない(解が無くならないように)',
      'model.Add(in_time == 1)' not in SRC)

# 時間制限つきでも、ちゃんと解けて指示も通る
ai_lim = new_ai(limit=60)
env3 = fresh()
ai_lim(state_of(env3))
m3 = dict(ai_lim._last_solve_metrics or {})
check('時間制限つきでも最適解が出る',
      str(m3.get('status', '')).upper() in ('OPTIMAL', 'FEASIBLE'), str(m3.get('status')))

print()
ok = sum(1 for r in results if r)
if ok == len(results):
    print(f'[SUCCESS] {ok}/{len(results)} 件すべて期待どおり')
else:
    print(f'[FAILURE] {ok}/{len(results)} 件のみ期待どおり')
    sys.exit(1)
