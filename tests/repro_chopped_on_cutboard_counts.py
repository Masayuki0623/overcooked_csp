"""まな板に残った「切り終わった食材」を、済みとして数えることの検証。

これまでの問題:
    残っている工程を数える get_remaining_tids は、まな板の上の物を
    まるごと読み飛ばしていた。切り終わったトマトがまな板に載っていても
    「まだ切っていない」と数え、('chop','tomato') が残り続ける。
    その結果、切る必要が無いのに指示の候補へ「トマトを切って」が並ぶ。
    選ぶと AI はまな板から切ったトマトを拾って別の場所へ置くだけで
    終わる(報告あり: 「ただただ別の場所へトマトを持って行っただけ」)。

    読み飛ばしていたのは、刻みかけ(Chopping)を「もうある」と数えないため。
    そこは残す必要があるが、刻み終わりまで巻き添えにしていた。

ここで確かめること:
    1. まな板の上の「切り終わった」食材は、済みとして数える
    2. 切る必要が無いなら、指示の候補にも「切って」を出さない
    3. 刻みかけ(Chopping)は、これまでどおり数えない
    4. ふつうの台の上のふるまいは変わらない
    5. 足りない分はちゃんと残る(2品要るのに1つしか無ければ1つ残る)

実行方法:
    python tests/repro_chopped_on_cutboard_counts.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from gym_cooking.utils.core import Object, Tomato, FoodState
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


def new_ai():
    ai = CSPAgent(10, Replay(), sc_2agent=True, skip_budget=0)
    ai.human_counterpart_mode = True
    ai.own_agent_idx = 0
    ai.priority_weights = {}
    ai.gui_text_input = ''
    ai.gui_constraint_input = ''
    ai.active_constraints = []
    return ai


def fresh(orders=('OnionTomatoSalad',)):
    env = OvercookedEnvironment(MapSetting(level='exp_partition',
                                           order_recipes=orders,
                                           max_num_timesteps=200))
    env.reset()
    return env


def put_tomato(env, loc, state):
    t = Tomato()
    t.set_state(state)
    o = Object(location=loc, contents=[t])
    env.world.insert(o)
    env.world.get_gridsquare_at(loc).acquire(o)


def tomato_tasks(env):
    ai = new_ai()
    st = state_of(env)
    rem = ai.get_remaining_tids(st, ai._build_order_tasks(st))
    verbs = {p['verb'] + ' ' + p['obj']
             for _d, p in ai.get_instruction_candidates(st)}
    return sorted(t for t in rem if t[1] == 'tomato'), verbs


_e = fresh()
_st = state_of(_e)
BOARDS = [tuple(l) for l in _st.get_pos_by_obj_gs(gs='Cutboard')]
FREE = [tuple(l) for l in _st.get_pos_by_obj_gs(gs='Counter')
        if _st.pos_obj.get(tuple(l)) is None and tuple(l)[0] < 6]

# 1 + 2
env = fresh()
put_tomato(env, BOARDS[0], FoodState.CHOPPED)
rem, cands = tomato_tasks(env)
check('まな板の切り終わったトマトは、済みとして数える', not rem, f'残っている: {rem}')
check('切る必要が無いので「トマトを切って」は候補に出ない',
      'chop tomato' not in cands, f'候補: {sorted(cands)}')

# 3
env = fresh()
put_tomato(env, BOARDS[0], FoodState.CHOPPING)
rem, _c = tomato_tasks(env)
check('刻みかけは、これまでどおり数えない',
      any(t[0] == 'chop' for t in rem), f'残っている: {rem}')

# 4
env = fresh()
put_tomato(env, FREE[0], FoodState.CHOPPED)
rem, cands = tomato_tasks(env)
check('ふつうの台の上は、これまでどおり済みとして数える', not rem, f'残っている: {rem}')

env = fresh()
put_tomato(env, FREE[0], FoodState.FRESH)
rem, _c = tomato_tasks(env)
check('切っていないトマトを数えたりはしない',
      any(t[0] == 'chop' for t in rem), f'残っている: {rem}')

# 5
env = fresh(orders=('OnionTomatoSalad', 'TomatoLettuceSalad'))
put_tomato(env, BOARDS[0], FoodState.CHOPPED)
rem, _c = tomato_tasks(env)
chops = [t for t in rem if t[0] == 'chop']
check('2品要るのに1つしか無ければ、1つぶん残る', len(chops) == 1, f'残っている: {rem}')

env = fresh(orders=('OnionTomatoSalad', 'TomatoLettuceSalad'))
put_tomato(env, BOARDS[0], FoodState.CHOPPED)
put_tomato(env, FREE[0], FoodState.CHOPPED)
rem, _c = tomato_tasks(env)
check('2つそろえば残らない', not [t for t in rem if t[0] == 'chop'], f'残っている: {rem}')

print()
ok = sum(1 for r in results if r)
if ok == len(results):
    print(f'[SUCCESS] {ok}/{len(results)} 件すべて期待どおり')
else:
    print(f'[FAILURE] {ok}/{len(results)} 件のみ期待どおり')
    sys.exit(1)
