"""もう切ってある材料を、自分で運びに行くことの検証。

想定シナリオ(バグ報告 20260924_180512「動かなくなった」の場面):
    仕切りの地図。共有台に「刻んだレタス」と「刻んだ玉ねぎ」が別々に
    乗っている。スープ(レタス+玉ねぎ)は玉ねぎの台を置き場にしていて、
    レタスを運んで合流させれば、すぐ煮始められる。

    ところが AI は鍋の前で「不足分がそろうのを待機中」のまま動かなかった。
    運ぶ工程が人間の担当になっていたためで、レタスは台に乗ったまま、
    人間が刻み直すのを全員で待っていた。

    「切らずに運ぶだけ」の工程は、まな板も材料の供給口も要らない。
    供給口の無い側にも割り当てられるようにする。

実行方法:
    python tests/repro_fetch_chopped_not_left_to_partner.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from gym_cooking.utils.core import Lettuce, Onion, Object, FoodState
from gym_cooking.utils.interact import resolve_action
from gym_cooking.utils.replay import Replay
from agent.agent.executor.low import EnvState
from agent.agent.myagent.CSPAgent import CSPAgent

ORDERS = ('TomatoLettuceSalad', 'OnionLettuceSoup', 'BananaOrangeJuice')
LETTUCE_AT = (6, 6)
ONION_AT = (6, 7)
results = []


def check(label, ok, detail=""):
    results.append(ok)
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}{(' -> ' + detail) if detail else ''}")


def chopped(cls):
    f = cls()
    f.set_state(FoodState.CHOPPED)
    return f


def state_of(env):
    i = env.get_ai_info()
    return EnvState(world=i['world'], agents=i['sim_agents'], agent_idx=0,
                    order=i['order_scheduler'], event_history=i['event_history'],
                    time=i['current_time'], chg_grid=i['chg_grid'])


def main():
    env = OvercookedEnvironment(MapSetting(level='exp_partition',
                                           order_recipes=ORDERS))
    env.reset()
    for pos, cls in ((LETTUCE_AT, Lettuce), (ONION_AT, Onion)):
        obj = Object(location=pos, contents=[chopped(cls)])
        env.world.insert(obj)
        env.world.get_gridsquare_at(pos).acquire(obj)
    print(f'[SETUP] 注文={ORDERS}')
    print(f'[SETUP] 共有台 {LETTUCE_AT} に刻んだレタス / {ONION_AT} に刻んだ玉ねぎ')
    print('[SETUP] レタスの供給口は人間側のみ。運べばスープはすぐ煮始められる')

    ai = CSPAgent(10, Replay(), sc_2agent=True, skip_budget=2)
    ai.human_counterpart_mode = True
    ai.own_agent_idx = 0
    ai.priority_weights = {}
    ai.gui_text_input = ''
    ai.gui_constraint_input = ''
    ai.active_constraints = []
    move, reason = ai(state_of(env))

    def fetch_tasks(who):
        return [t for t in ((ai.schedule_per_agent or {}).get(who) or [])
                if t.get('carry_from') is not None]

    mine, theirs = fetch_tasks(0), fetch_tasks(1)
    print('  AI の運搬工程:', [(t['id'], t['carry_from']) for t in mine])
    print('  人の運搬工程:', [(t['id'], t['carry_from']) for t in theirs])
    check('運ぶだけの工程は AI の担当になる', bool(mine) and not theirs)

    own = move.get('ai_0') if isinstance(move, dict) else move
    check('待たずに動き出す', bool(own) and tuple(own) != (0, 0),
          f'手={own} 理由={reason}')

    cooking_at = None
    for _step in range(1, 151):
        move, _reason = ai(state_of(env))
        own = move.get('ai_0') if isinstance(move, dict) else move
        acts = {a.name: (0, 0) for a in env.sim_agents}
        if own:
            acts[env.sim_agents[0].name] = own
        for a in env.sim_agents:
            acts[a.name] = resolve_action(a, env.world, acts[a.name])
        env.step(acts, passed_time=0.2)
        st = state_of(env)
        for pot in st.get_pos_by_obj_gs(gs='Pot'):
            name = str(getattr(st.pos_obj.get(tuple(pot)), 'full_name', '') or '')
            if 'Lettuce' in name and 'Onion' in name:
                cooking_at = env.current_time
                break
        if cooking_at:
            break
    check('レタスを運んでスープを煮始められる', cooking_at is not None,
          f'{cooking_at:.1f}秒' if cooking_at else '30秒回しても鍋に入らず')

    print()
    print(f"[{'SUCCESS' if all(results) else 'FAIL'}] "
          f"{results.count(True)}/{len(results)} 件が期待どおり")
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
