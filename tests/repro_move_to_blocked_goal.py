"""目的地の前に相手が立っていても、そこへ向かって待つことの検証。

想定シナリオ(報告):
    経路探索が相手のいるマスを壁として扱っていたため、相手が目的地の前
    (唯一の隣接マス)や一本道に立つと経路が無くなり、AI はその場で止まって
    いた。相手はいずれ動くので、目的地の手前まで進んで待つのが正しい。
    以前は 15 フレーム待つと無作為に一歩離れていたが、目的地の前に相手が
    いるだけなら離れない。

実行方法:
    python tests/repro_move_to_blocked_goal.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from agent.agent.executor.low import EnvState
from agent.agent.myagent.TaskAgent import TaskAgent

results = []


def check(label, ok, detail=''):
    results.append(bool(ok))
    print(f"  [{'OK  ' if ok else 'NG  '}] {label}" + (f" -> {detail}" if detail else ''))


def state_of(env):
    i = env.get_ai_info()
    return EnvState(world=i['world'], agents=i['sim_agents'], agent_idx=0,
                    order=i['order_scheduler'], event_history=i['event_history'],
                    time=i['current_time'], chg_grid=i['chg_grid'])


def main():
    env = OvercookedEnvironment(MapSetting(level='exp_ring',
                                           order_recipes=('OnionTomatoSalad',)))
    env.reset()
    ai_agent, human = env.sim_agents[0], env.sim_agents[1]
    # 目的地: 左列の台 (0,3)。唯一の隣接マスは (1,3)。そこに人間を立たせる。
    target = (0, 3)
    human.location = (1, 3)
    ai_agent.location = (1, 6)
    print(f'[SETUP] AI={ai_agent.location} 人間={human.location} 目的地={target}')

    ta = TaskAgent(speed=10)
    trail = []
    for _ in range(30):
        st = state_of(env)
        move = ta.move_to(st, target, dynamic_obstacles={tuple(human.location)})
        x, y = ai_agent.location
        nxt = (x + move[0], y + move[1])
        # 人間のマスには入れない。それ以外の床なら進む。
        if move != (0, 0) and nxt != tuple(human.location) and env.world.get_gridsquare_at(nxt).__class__.__name__ == 'Floor':
            ai_agent.location = nxt
        trail.append(tuple(ai_agent.location))

    check('目的地の手前 (1,4) まで進む', (1, 4) in trail, f'通った所={sorted(set(trail))}')
    last10 = trail[-10:]
    check('手前に着いたら離れずに待つ', all(p == (1, 4) for p in last10),
          f'最後の10フレーム={last10}')

    # 人間がどいたら、すぐ目的地に手を出せる
    human.location = (1, 1)
    st = state_of(env)
    move = ta.move_to(st, target, dynamic_obstacles={tuple(human.location)})
    ai_agent.location = (ai_agent.location[0] + move[0], ai_agent.location[1] + move[1])
    st = state_of(env)
    move2 = ta.move_to(st, target, dynamic_obstacles={tuple(human.location)})
    check('人間がどいたら目的地へ進んで手を出す', tuple(ai_agent.location) == (1, 3) and move2 == (-1, 0),
          f'位置={ai_agent.location} 次の一手={move2}')

    ng = results.count(False)
    if ng:
        print(f'[FAILED] {ng}/{len(results)} 件が期待どおりではありません')
        return 1
    print(f'[SUCCESS] {len(results)}/{len(results)} 件すべて期待どおり')
    return 0


if __name__ == '__main__':
    sys.exit(main())
