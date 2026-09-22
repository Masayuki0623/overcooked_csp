"""刻んである材料の取り合いを、注文の並び順ではなく makespan で裁くことの検証。

想定シナリオ:
    トマト・レタスのサラダ と 玉ねぎ・レタスのスープ が同時に出ていて、
    刻んだレタスが1つだけ空いているテーブルに置いてある。
    -> レタスは両方が欲しがるが、在庫は1つしかない。

    以前は注文の並び順(サラダが常に先)で先着順に決まっていたため、
    サラダがレタスを確保し、スープは切り直しになっていた。スープは
    煮込み15秒があるぶん全体の所要時間を決めているので、これは損。
    いまはどちらが取るかを両方解いて makespan の短い方を採る。

実行方法:
    python tests/test_stock_claim_by_makespan.py
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


def rechops(orders, uid, ing):
    """その注文が『切り直す』ことになっている工程の数。

    carry_from が入っている chop は「切ってある物を運ぶだけ」なので数えない。
    """
    return [t for o in orders for t in o['tasks']
            if t['id'] == ('chop', ing, uid) and t.get('carry_from') is None]


def main():
    env = OvercookedEnvironment(MapSetting(
        level="new1",
        order_recipes=('TomatoLettuceSalad', 'OnionLettuceSoup')))
    env.reset()

    agent = CSPAgent(speed=10, sc_2agent=True)
    agent.human_counterpart_mode = True
    agent.own_agent_idx = 0
    agent.debug_counter_trace = bool(os.environ.get('CSP_DEBUG'))

    def build():
        i = env.get_ai_info()
        return EnvState(world=i['world'], agents=i['sim_agents'], agent_idx=0,
                        order=i['order_scheduler'], event_history=i['event_history'],
                        time=i['current_time'], chg_grid=i['chg_grid'])

    agent(build())

    uids = {}
    for entry in agent.active_order_entries:
        uids['soup' if 'cooked' in entry['name'] else 'salad'] = entry['uid']
    if len(uids) != 2:
        print(f"[FAIL] サラダとスープの注文が揃わなかった: {agent.active_order_entries}")
        return
    print(f"[SETUP] salad_uid={uids['salad']} soup_uid={uids['soup']}")

    # スープ側は玉ねぎを済ませてあり、あとはレタスだけ、という状況にする。
    # こうすると「レタスをスープに回せば、そのぶん早く鍋に入れられる」ので、
    # どちらが在庫を取るかが全体の所要時間に効く。
    soup_counter = agent.counter_policy_by_order[uids['soup']]['counter']
    place(env, soup_counter, ['onion'])
    print(f"[SETUP] スープの置き場 {soup_counter} に刻んだ玉ねぎを置いた")

    # どの注文の置き場でもない空きテーブルに、刻んだレタスを1つ置く。
    es = build()
    reserved = {e.get('counter') for e in agent.counter_policy_by_order.values()}
    free = next(p for p in agent._get_resources(es).get('counters', [])
                if p not in reserved and es.pos_obj.get(p) is None)
    place(env, free, ['lettuce'])
    print(f"[SETUP] 空きテーブル {free} に刻んだレタスを1つ置いた")

    # --- 1) 取り合いが検出されること ---
    orders = agent._build_order_tasks(build())
    check("レタスの取り合いを検出する", 'lettuce' in agent._stock_contest,
          f"contest={agent._stock_contest}")
    before = {k: len(rechops(orders, v, 'lettuce')) for k, v in uids.items()}
    print(f"[BEFORE] 切り直し: {before}")

    # --- 2) makespan で裁いた結果、煮込みのあるスープ側が在庫を取ること ---
    agent._claim_decision_key = None
    _, solved = agent._solve_best_claim_priority(build(), orders)
    after = {k: len(rechops(solved, v, 'lettuce')) for k, v in uids.items()}
    print(f"[AFTER ] 切り直し: {after} 確保順={agent._claim_priority}")

    check("在庫を取るのは片方だけ(合計の切り直しは1件)",
          after['salad'] + after['soup'] == 1, f"{after}")
    check("煮込みのあるスープが在庫を取る", after['soup'] == 0, f"{after}")

    # --- 3) 決めた後は、同じ取り合いについて決め直さないこと ---
    key = agent._claim_decision_key
    orders3 = agent._build_order_tasks(build())
    _, solved3 = agent._solve_best_claim_priority(build(), orders3)
    after3 = {k: len(rechops(solved3, v, 'lettuce')) for k, v in uids.items()}
    check("再計算しても裁き方が入れ替わらない", after3 == after, f"{after3}")
    check("決着済みの取り合いは解き直さない",
          agent._claim_decision_key == key, f"{agent._claim_decision_key} vs {key}")

    print()
    print(f"[{'SUCCESS' if all(results) else 'FAIL'}] {results.count(True)}/{len(results)} 件が期待どおり")
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main() or 0)
