"""手に持っている材料を、計画が正しく数えることの検証。

想定シナリオ(報告: 「人間の位置と持ち物を正確に把握できていない。
レタスを持っているなら、切るための費用はまな板への移動と刻む分だけ
のはず」):
    手に持っている物は世界の一覧に無いので、計画からは見えていなかった。
    「切る」は必ず供給口から取りに行く前提で見積もられ、持っている人に
    固定もされなかった。

ここで見るのは3点(人間でも AI でも同じ):
    1. 出発点が持っている人の位置になる
    2. 所要時間が「取りに行く」ぶん短くなる
    3. その工程が持っている人に固定される(held_by)

実行方法:
    python tests/repro_chop_from_hand.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from gym_cooking.utils.core import Lettuce, Tomato, Plate, Object, FoodState
from agent.agent.executor.low import EnvState
from agent.agent.myagent.CSPAgent import CSPAgent
from gym_cooking.utils.replay import Replay

results = []


def check(label, ok, detail=''):
    results.append(bool(ok))
    print(f"  [{'OK  ' if ok else 'NG  '}] {label}" + (f" -> {detail}" if detail else ''))


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


def fresh():
    env = OvercookedEnvironment(MapSetting(level='exp_ring',
                                           order_recipes=('TomatoLettuceSoup',)))
    env.reset()
    return env


def give(env, agent_idx, *foods):
    ag = env.sim_agents[agent_idx]
    obj = Object(location=tuple(ag.location), contents=[foods[0]])
    for extra in foods[1:]:
        obj.contents.append(extra)
    obj.is_held = True
    env.world.insert(obj)
    ag.acquire(obj)
    return obj


def chopped(cls):
    f = cls()
    f.set_state(FoodState.CHOPPED)
    return f


def task_of(ai, env, verb, obj):
    st = state_of(env)
    orders = ai._build_order_tasks(st)
    tasks = [t for o in orders for t in o.get('tasks', [])]
    ai._annotate_task_geometry(st, tasks, tuple(env.sim_agents[0].location))
    for t in tasks:
        if t['id'][0] == verb and t['id'][1] == obj:
            return t
    return None


def chop_lettuce_task(ai, env):
    st = state_of(env)
    orders = ai._build_order_tasks(st)
    tasks = [t for o in orders for t in o.get('tasks', [])]
    ai._annotate_task_geometry(st, tasks, tuple(env.sim_agents[0].location))
    for t in tasks:
        if t['id'][0] == 'chop' and t['id'][1] == 'lettuce':
            return t
    return None


def main():
    # 基準: 誰も持っていない
    env0 = fresh()
    base = chop_lettuce_task(new_ai(), env0)
    check('基準の「レタスを切る」がある', base is not None)
    check('基準は誰にも固定されない', base.get('held_by') is None)

    # 人間(2人目)がレタスを持っている
    env1 = fresh()
    give(env1, 1, Lettuce())
    human_pos = tuple(env1.sim_agents[1].location)
    t1 = chop_lettuce_task(new_ai(), env1)
    check('出発点が人間の位置', tuple(t1['start_pos']) == human_pos,
          f"start={t1['start_pos']} 人間={human_pos}")
    check('人間に固定される(held_by=1)', t1.get('held_by') == 1, str(t1.get('held_by')))
    check('取りに行く分だけ短くなる', t1['dur'] < base['dur'],
          f"持っている={t1['dur']} / 基準={base['dur']}")

    # AI(1人目)が持っている場合も同じ
    env2 = fresh()
    give(env2, 0, Lettuce())
    ai_pos = tuple(env2.sim_agents[0].location)
    t2 = chop_lettuce_task(new_ai(), env2)
    check('AI が持っていれば AI の位置から', tuple(t2['start_pos']) == ai_pos,
          f"start={t2['start_pos']} AI={ai_pos}")
    check('AI に固定される(held_by=0)', t2.get('held_by') == 0, str(t2.get('held_by')))

    # 煮る: 人間が刻んだ材料をそろえて持っている
    print('[煮る・提供も同じ]')
    env3 = fresh()
    base_cook = task_of(new_ai(), env3, 'cook', 'lettuce-tomato soup')
    base_serve = task_of(new_ai(), env3, 'serve', 'lettuce-tomato soup')
    env4 = fresh()
    give(env4, 1, chopped(Lettuce), chopped(Tomato))
    hp = tuple(env4.sim_agents[1].location)
    c4 = task_of(new_ai(), env4, 'cook', 'lettuce-tomato soup')
    check('煮る: 出発点が持っている人間の位置', tuple(c4['start_pos']) == hp,
          f"start={c4['start_pos']} 人間={hp}")
    check('煮る: 人間に固定', c4.get('held_by') == 1, str(c4.get('held_by')))
    check('煮る: 集めに行く分が無いので短い', c4['dur'] < base_cook['dur'],
          f"持っている={c4['dur']} / 基準={base_cook['dur']}")

    # 提供: AI が空の皿を持っている
    env5 = fresh()
    give(env5, 0, Plate())
    ap = tuple(env5.sim_agents[0].location)
    s5 = task_of(new_ai(), env5, 'serve', 'lettuce-tomato soup')
    check('提供: 皿を持つ AI の位置から', tuple(s5['start_pos']) == ap,
          f"start={s5['start_pos']} AI={ap}")
    check('提供: AI に固定', s5.get('held_by') == 0, str(s5.get('held_by')))
    check('提供: 皿を取りに行く分が無いので短い', s5['dur'] < base_serve['dur'],
          f"持っている={s5['dur']} / 基準={base_serve['dur']}")

    ng = results.count(False)
    if ng:
        print(f'[FAILED] {ng}/{len(results)} 件が期待どおりではありません')
        return 1
    print(f'[SUCCESS] {len(results)}/{len(results)} 件すべて期待どおり')
    return 0


if __name__ == '__main__':
    sys.exit(main())
