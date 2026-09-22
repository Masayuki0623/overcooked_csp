"""ミキサーを回すべきときに、コップを持ったまま止まらないことの検証。

想定シナリオ:
    ジュースの材料がミキサーに入っていて、まだ混ざりきっていない。
    AI はコップを持っている。

    ミキサーは鍋と違って、放っておいても進まない。手ぶらで向かって
    インタラクトした回数だけ混ざる。コップを持ったままだとインタラクトは
    「注ぐ」に化けるが、混ざりきるまでは注げないので何も起きない。
    -> コップを置いて混ぜに戻らなければ、永久に止まる。

実行方法:
    python tests/repro_mix_with_cup_in_hand.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from gym_cooking.utils.core import Apple, Orange, Cup, Object, FoodState
from agent.agent.executor.low import EnvState
from agent.agent.myagent.TaskAgent import TaskAgent

results = []


def check(label, ok, detail=""):
    results.append(ok)
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}{(' -> ' + detail) if detail else ''}")


def chopped(cls):
    f = cls()
    f.set_state(FoodState.CHOPPED)
    return f


def main():
    env = OvercookedEnvironment(MapSetting(
        level='exp_ring', order_recipes=('AppleOrangeJuice',)))
    env.reset()

    blender = env.world.get_object_locs_by_gs(gs='Blender')[0] \
        if hasattr(env.world, 'get_object_locs_by_gs') else None
    if blender is None:
        i = env.get_ai_info()
        blender = i['world'].get_gridsquare_locations('Blender')[0] \
            if hasattr(i['world'], 'get_gridsquare_locations') else None
    if blender is None:
        # 環境の API 名に依存しないよう、盤面から直接探す。
        for gs in env.world.get_object_list():
            if type(gs).__name__ == 'Blender':
                blender = gs.location
                break
    if blender is None:
        print('[SKIP] ミキサーのある地図ではありません')
        return 0

    # 刻んだフルーツを入れて、混ぜ始めた状態にする(まだ混ざりきっていない)。
    juice = Object(location=blender, contents=[chopped(Apple), chopped(Orange)])
    env.world.insert(juice)
    env.world.get_gridsquare_at(blender).acquire(juice)
    juice.mix(env.current_time)
    print(f'[SETUP] ミキサー {blender} に {juice.full_name}'
          f' (mixing={juice.is_mixing()}, mixed={juice.is_mixed()})')

    # AI の手にコップを持たせる。
    ai_agent = env.sim_agents[0]
    cup = Object(location=ai_agent.location, contents=[Cup()])
    env.world.insert(cup)
    ai_agent.acquire(cup)
    print(f'[SETUP] AI は {cup.full_name} を持っている')

    info = env.get_ai_info()
    es = EnvState(world=info['world'], agents=info['sim_agents'], agent_idx=0,
                  order=info['order_scheduler'], event_history=info['event_history'],
                  time=info['current_time'], chg_grid=info['chg_grid'])

    ta = TaskAgent(speed=10, replay=None, task_name='serve_juice')
    ta.task_name = 'serve_juice'
    action, reason = ta.process_serve_juice_task(
        es, ingredients=['Apple', 'Orange'], assigned_blender=blender)
    print(f'[RESULT] serve_juice: action={action} reason={reason}')
    check('コップを持ったまま「完了待ち」で止まらない',
          '待ち' not in reason, reason)
    check('コップを置きに行く', ('置' in reason or 'Cup' in reason), reason)

    # mix タスク側は、もともと置きに行く作りになっている(退行していないか確認)
    action2, reason2 = ta.process_mix_task(
        es, ingredients=['Apple', 'Orange'], assigned_blender=blender)
    print(f'[RESULT] mix       : action={action2} reason={reason2}')
    check('mix 側もコップを置きに行く', '置' in reason2, reason2)

    # 手ぶらのときは、コップを取りに行く前に回しきる。先に取ると
    # 「持ったまま待つ」状態に戻ってしまう。
    env.world.remove(ai_agent.holding)
    ai_agent.release()
    info2 = env.get_ai_info()
    es2 = EnvState(world=info2['world'], agents=info2['sim_agents'], agent_idx=0,
                   order=info2['order_scheduler'], event_history=info2['event_history'],
                   time=info2['current_time'], chg_grid=info2['chg_grid'])
    action3, reason3 = ta.process_serve_juice_task(
        es2, ingredients=['Apple', 'Orange'], assigned_blender=blender)
    print(f'[RESULT] 手ぶら    : action={action3} reason={reason3}')
    check('手ぶらならコップより先にミキサーを回す', 'ミキサーを回す' in reason3, reason3)

    # 鍋(放っておいても煮える)は、これまでどおり待ってよい。
    ta2 = TaskAgent(speed=10, replay=None, task_name='serve')
    action4, reason4 = ta2.process_serve_task(
        es, ingredients=['Onion', 'Tomato'])
    print(f'[RESULT] 鍋(退行確認): action={action4} reason={reason4}')
    check('鍋の振る舞いは変えていない', '置' not in reason4 or '鍋を回す' not in reason4, reason4)

    print()
    print(f"[{'SUCCESS' if all(results) else 'FAIL'}] "
          f"{results.count(True)}/{len(results)} 件が期待どおり")
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
