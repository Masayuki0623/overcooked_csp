"""ジュースの「混ぜて」「提供して」を指示できることの検証。

これまでの問題:
    残っている工程を数える get_remaining_tids は、盤面の食材を
    extract_food_names で拾う。その中身が ('Lettuce','Onion','Tomato')
    固定で、果物(Apple/Orange/Banana)が1つも入っていなかった。

    このため果物は「切り終えた」と数えられず、('chop', 果物) が永久に
    残る。mix は「材料を全部刻み終えていること」を前提にしているので
    永久に前提未達となり、指示の候補に一度も出てこない。serve_juice も
    「mix が終わっていること」が前提なので同じく出てこない。
    ミキサーの中身も拾えないため、混ぜ終わりも認識できなかった。

    実測: 記録に残った指示 69 件はすべて chop。mix と serve 系は0件。
    90秒×3回まわしても候補に出たのは chop と cook だけだった。

ここで確かめること:
    1. 切った果物が「切り終えた」と数えられる
    2. 切った果物がそろえば「混ぜて」が指示の候補に出る
    3. 混ぜ終わっていれば「提供して」が指示の候補に出る
    4. 野菜だけの注文のふるまいは変わらない

実行方法:
    python tests/repro_juice_instruction.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from gym_cooking.utils.core import Object, Apple, Orange, Onion, FoodState
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


def fresh(orders=('AppleOrangeJuice', 'OnionTomatoSoup'), level='exp_ring'):
    # リングで見る。仕切りは AI と人間の開始位置を入れ替えたので
    # (2026-09-28)、ミキサー・提供口のある左側が人間の側になり、AI には
    # ジュースの工程を指示できない(最後にそれも確かめる)。
    env = OvercookedEnvironment(MapSetting(level=level, order_recipes=orders))
    env.reset()
    return env


def put(env, loc, *foods):
    obj = Object(location=loc, contents=[foods[0]])
    for extra in foods[1:]:
        obj.contents.append(extra)
    env.world.insert(obj)
    env.world.get_gridsquare_at(loc).acquire(obj)
    return obj


def ai_side_counters(env, side='left'):
    st = state_of(env)
    keep = (lambda x: x < 6) if side == 'left' else (lambda x: x > 6)
    return [tuple(l) for l in st.get_pos_by_obj_gs(gs='Counter')
            if keep(tuple(l)[0]) and st.pos_obj.get(tuple(l)) is None]


def verbs_of(env):
    ai = new_ai()
    return {p['verb']: p['obj']
            for _d, p in ai.get_instruction_candidates(state_of(env))}


# 1. 切った果物が「切り終えた」と数えられる
env = fresh()
spots = ai_side_counters(env)
a, o = Apple(), Orange()
a.set_state(FoodState.CHOPPED)
o.set_state(FoodState.CHOPPED)
put(env, spots[0], a)
put(env, spots[1], o)
ai = new_ai()
st = state_of(env)
remaining = ai.get_remaining_tids(st, ai._build_order_tasks(st))
still_chop = {t for t in remaining if t[0] == 'chop' and t[1] in ('apple', 'orange')}
check('切った果物は「切り終えた」と数えられる', not still_chop, f'まだ残っている: {still_chop}')

# 2. そろえば「混ぜて」が候補に出る
got = verbs_of(env)
check('切った果物がそろうと「混ぜて」が指示の候補に出る', 'mix' in got, f'候補: {got}')
check('その対象はジュース', got.get('mix', '').endswith('juice'), str(got.get('mix')))

# 3. 混ぜ終わっていれば「提供して」が出る
env = fresh()
blender = [tuple(l) for l in state_of(env).get_pos_by_obj_gs(gs='Blender')][0]
a2, o2 = Apple(), Orange()
a2.set_state(FoodState.MIXED)
o2.set_state(FoodState.MIXED)
put(env, blender, a2, o2)
got = verbs_of(env)
check('混ぜ終わると「提供して」が指示の候補に出る', 'serve_juice' in got, f'候補: {got}')
check('混ぜ終わったあとに「混ぜて」は残らない', 'mix' not in got, f'候補: {got}')

# 4. 野菜だけの注文は変わらない
env = fresh(orders=('OnionTomatoSoup',))
spots = ai_side_counters(env)
on = Onion()
on.set_state(FoodState.CHOPPED)
put(env, spots[0], on)
ai = new_ai()
st = state_of(env)
remaining = ai.get_remaining_tids(st, ai._build_order_tasks(st))
check('切った玉ねぎも、これまでどおり数えられる',
      not any(t[0] == 'chop' and t[1] == 'onion' for t in remaining),
      str({t for t in remaining if t[0] == 'chop'}))
# 仕切り: AI は右側(レタス・トマト・オレンジ・バナナ・まな板)にいて、
# ミキサーとコップは左側(人間の側)。切った果物を AI の側に置いても、
# 「混ぜて」は指示の候補に出ない(AI が物理的にできないため)。
env_p = fresh(level='exp_partition')
spots_p = ai_side_counters(env_p, side='right')
a2, o2 = Apple(), Orange()
a2.set_state(FoodState.CHOPPED)
o2.set_state(FoodState.CHOPPED)
put(env_p, spots_p[0], a2, o2)
got_p = verbs_of(env_p)
check('仕切りでは AI にジュースの工程を指示できない(右側にミキサーが無い)',
      'mix' not in got_p and 'serve_juice' not in got_p, f'候補: {got_p}')

check('野菜の注文にジュースの工程は混ざらない',
      not any(t[0] in ('mix', 'serve_juice') for t in remaining),
      str(remaining))

print()
ok = sum(1 for r in results if r)
if ok == len(results):
    print(f'[SUCCESS] {ok}/{len(results)} 件すべて期待どおり')
else:
    print(f'[FAILURE] {ok}/{len(results)} 件のみ期待どおり')
    sys.exit(1)
