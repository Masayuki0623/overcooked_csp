"""相手待ちの工程を抱えたまま、できる作業を放置して止まらないことの検証。

想定シナリオ(実測で起きた場面):
    仕切りの地図。ジュースの材料(オレンジ・バナナ)は人間側にしか無く、
    AI は人が刻んで渡してくれるまで何もできない。
    一方サラダの材料(刻んだ玉ねぎ・刻んだトマト)は置き場にそろっていて、
    提供口は AI 側にあるので、AI はいますぐサラダを出せる。

    それなのに AI は「ジュースの材料待ち」を抱えたまま受け渡し台の前で
    14 秒間止まっていた(リプレイから再構成して確認)。その間にサラダを
    出せたはずで、出さないまま鍋が煮上がるのを待っていた。

実行方法:
    python tests/repro_idle_while_waiting_partner.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from gym_cooking.utils.core import Lettuce, Onion, Tomato, Object, FoodState
from gym_cooking.utils.replay import Replay
from agent.agent.executor.low import EnvState
from agent.agent.myagent.CSPAgent import CSPAgent

ORDERS = ('OnionTomatoSalad', 'OnionLettuceSoup', 'BananaOrangeJuice')
results = []


def check(label, ok, detail=""):
    results.append(ok)
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}{(' -> ' + detail) if detail else ''}")


def chopped(cls):
    f = cls()
    f.set_state(FoodState.CHOPPED)
    return f


def place(env, pos, obj):
    env.world.insert(obj)
    env.world.get_gridsquare_at(pos).acquire(obj)
    return obj


def state_of(env):
    i = env.get_ai_info()
    return EnvState(world=i['world'], agents=i['sim_agents'], agent_idx=0,
                    order=i['order_scheduler'], event_history=i['event_history'],
                    time=i['current_time'], chg_grid=i['chg_grid'])


def main():
    env = OvercookedEnvironment(MapSetting(level='exp_partition',
                                           order_recipes=ORDERS))
    env.reset()

    # AI に見せる前に、その場面の盤面を作っておく。
    # (先に AI を動かすと置き場の割り当てが決まってしまい、あとから置いた
    #  食材を別注文のものと見て使えなくなる)
    pot = state_of(env).get_pos_by_obj_gs(gs='Pot')[0]
    soup = Object(location=pot, contents=[chopped(Lettuce), chopped(Onion)])
    place(env, pot, soup)
    soup.cook(0.1)   # 0 だと環境側の時刻判定に引っかかる
    print(f'[SETUP] 鍋 {pot} でスープを煮ている (cooking={soup.is_cooking()})')

    counter = (6, 5)       # 仕切りの共有台
    place(env, counter, Object(location=counter,
                               contents=[chopped(Onion), chopped(Tomato)]))
    print(f'[SETUP] 共有台 {counter} に 刻んだ玉ねぎ+トマト = サラダはすぐ出せる')
    print('[SETUP] ジュースの材料(オレンジ・バナナ)は刻まれていない = 相手待ち')

    ai = CSPAgent(10, Replay(), sc_2agent=True, skip_budget=2)
    ai.human_counterpart_mode = True
    ai.own_agent_idx = 0
    ai.priority_weights = {}
    ai.gui_text_input = ''
    ai.gui_constraint_input = ''
    ai.active_constraints = []
    ai(state_of(env))

    sched = (ai.schedule_per_agent or {}).get(0) or []
    print('  AI の計画:', [t.get('id') for t in sched])

    # 実測の場面では「ジュースを混ぜる」が手前に来ていた。そこを再現する
    # ため、現在の作業を mix に合わせる。
    mix_idx = next((i for i, t in enumerate(sched)
                    if (t.get('id') or (None,))[0] == 'mix'), None)
    if mix_idx is None:
        print('  [SKIP] この構成では mix が AI の担当にならなかった')
        return 0
    ai.current_task_idx = dict(ai.current_task_idx or {})
    ai.current_task_idx[0] = mix_idx
    print(f'  現在の作業を {sched[mix_idx].get("id")} に合わせた (相手待ちの工程)')

    # AI の行動を実際に世界へ流し込み、サラダが出るかどうかで判定する。
    # (何の「つもり」かは current_task_idx からは読めない。割り込みで
    #  実行する作業だけ差し替わることがあるため)
    from gym_cooking.utils.interact import resolve_action
    waiting = 0
    reasons = []
    served_at = None
    for step in range(1, 61):
        move, reason = ai(state_of(env))
        own = move.get('ai_0') if isinstance(move, dict) else move
        reasons.append(str(reason))
        if '待機' in str(reason) or '待つ' in str(reason):
            waiting += 1
        acts = {a.name: (0, 0) for a in env.sim_agents}
        if own:
            acts[env.sim_agents[0].name] = own
        for a in env.sim_agents:
            acts[a.name] = resolve_action(a, env.world, acts[a.name])
        env.step(acts, passed_time=0.2)
        if env.order_scheduler.successful_orders and served_at is None:
            served_at = step * 0.2
            break
    for r in reasons[:3]:
        print(f'  reason={r}')

    check('相手待ちのまま止まらない', waiting == 0,
          f'{waiting}/{len(reasons)} フレームが待機')
    check('サラダを出せる', served_at is not None,
          f'{served_at:.1f}秒で提供' if served_at else
          f'{len(reasons)}フレーム回しても提供できず')

    print()
    print(f"[{'SUCCESS' if all(results) else 'FAIL'}] "
          f"{results.count(True)}/{len(results)} 件が期待どおり")
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
