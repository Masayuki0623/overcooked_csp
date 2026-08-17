"""切り直さずに「運ぶだけ」で済ませられることの検証と、テーブル間の移動合戦が
起きないことの確認。

想定シナリオ:
    レタス・玉ねぎ・トマトスープの注文があり、
    置き場には ChoppedLettuce-ChoppedOnion、別のテーブルには ChoppedTomato。
    -> トマトを切り直すのではなく、運んで合流させるだけでよい。

あわせて、他注文の置き場にある食材を横取りしない(=テーブル間で食材を
行ったり来たりさせない)ことも確認する。

実行方法:
    python tests/repro_carry_instead_of_rechop.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from gym_cooking.utils.core import Lettuce, Onion, Tomato, Object, FoodState
from agent.agent.executor.low import EnvState
from agent.agent.myagent.CSPAgent import CSPAgent

FOOD = {'onion': Onion, 'lettuce': Lettuce, 'tomato': Tomato}
results = []


def chopped(name):
    f = FOOD[name]()
    f.set_state(FoodState.CHOPPED)
    return f


def place(env, pos, names):
    obj = Object(location=pos, contents=[chopped(n) for n in names])
    env.world.insert(obj)
    env.world.get_gridsquare_at(pos).acquire(obj)
    return obj


def check(label, ok, detail=""):
    results.append(ok)
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}{(' -> ' + detail) if detail else ''}")


def main():
    env = OvercookedEnvironment(MapSetting(level="new1", order_file="sample"))
    env.reset()

    agent = CSPAgent(speed=10, sc_2agent=True)
    agent.human_counterpart_mode = True
    agent.own_agent_idx = 0
    agent.debug_counter_trace = False

    def build():
        i = env.get_ai_info()
        return EnvState(world=i['world'], agents=i['sim_agents'], agent_idx=0,
                        order=i['order_scheduler'], event_history=i['event_history'],
                        time=i['current_time'], chg_grid=i['chg_grid'])

    agent(build())

    # --- 検証A: 3種スープの注文だけに絞り、材料が別々のテーブルにある状況 ---
    # (他の注文もトマトを必要とすると、その注文が先にトマトを確保するのが正しい挙動
    #  なので、運ぶ経路そのものを見るために注文を1つに絞る)
    all_orders = list(env.order_scheduler.current_orders)
    three = next((o for o in all_orders
                  if sum(i in getattr(o[0], 'full_name', '').lower()
                         for i in ('lettuce', 'onion', 'tomato')) == 3), None)
    if three is None:
        print("[FAIL] 3種スープの注文が見つからなかった")
        return
    env.order_scheduler.current_orders = [three]

    agent(build())
    target_uid = agent.active_order_entries[0]['uid']
    counter = agent.counter_policy_by_order.get(target_uid, {}).get('counter')
    print(f"[SETUP] order_uid={target_uid} 置き場={counter}")

    es = build()
    reserved = {e.get('counter') for e in agent.counter_policy_by_order.values()}
    other = next(p for p in agent._get_resources(es).get('counters', [])
                 if p not in reserved and es.pos_obj.get(p) is None)
    place(env, counter, ['lettuce', 'onion'])
    place(env, other, ['tomato'])
    print(f"[SETUP] 置き場に lettuce+onion / 別テーブル {other} に tomato")

    agent._mark_reschedule_needed('test_carry')
    orders = agent._build_order_tasks(build())
    tomato_tasks = [t for o in orders for t in o['tasks']
                    if t['id'] == ('chop', 'tomato', target_uid)]
    check("トマトのタスクが1つ生成される", len(tomato_tasks) == 1, f"{len(tomato_tasks)}件")
    if tomato_tasks:
        carry_from = tomato_tasks[0].get('carry_from')
        check("切り直さず『運ぶだけ』になっている", carry_from == other, f"carry_from={carry_from}")

    # --- 検証B: 他注文の置き場にある食材は横取りしない ---
    # (横取りを許すとテーブル間で食材を行ったり来たりさせる合戦になる)
    gs = env.world.get_gridsquare_at(other)
    env.world.remove(gs.release())
    env.order_scheduler.current_orders = all_orders
    agent._mark_reschedule_needed('test_no_steal')
    agent(build())

    victim_uid = next((e['uid'] for e in agent.active_order_entries
                       if e['uid'] != target_uid
                       and agent.counter_policy_by_order.get(e['uid'], {}).get('counter')), None)
    if victim_uid is None:
        print("  [SKIP] 比較用の別注文が無いため横取り検証は省略")
    else:
        victim_counter = agent.counter_policy_by_order[victim_uid]['counter']
        if build().pos_obj.get(victim_counter) is None:
            place(env, victim_counter, ['tomato'])
        agent._mark_reschedule_needed('test_no_steal2')
        orders2 = agent._build_order_tasks(build())
        t2 = [t for o in orders2 for t in o['tasks']
              if t['id'] == ('chop', 'tomato', target_uid)]
        stolen = any(t.get('carry_from') == victim_counter for t in t2)
        check("他注文の置き場からは横取りしない", not stolen, f"victim_counter={victim_counter}")

    print()
    print(f"[{'SUCCESS' if all(results) else 'FAIL'}] {results.count(True)}/{len(results)} 件が期待どおり")


if __name__ == '__main__':
    main()
