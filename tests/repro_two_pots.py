"""鍋を2つにした地図(パターン3)の検証。

やりたいこと:
    鍋が1つだと、スープが2品出た時点で必ず順番待ちになる。鍋を2つに
    して、同時に2品まで煮られるようにする。地図は器具も材料も配置は
    そのままで、いまの鍋のすぐ下にもう1つ置いてある。

ここで確かめること:
    1. 鍋2つの地図に、鍋がちょうど2つある(置き場所も隣どうし)
    2. 鍋1つの地図はこれまでどおり
    3. 「いま調理を始められるか」の判定が、1つでも空いていれば通る
       (以前は「全部空いていること」を求めていた。鍋を2つにすると、
        片方が煮えているだけで始められない判定になってしまう)
    4. 全部ふさがっていれば始められない
    5. 注文ごとに違う鍋が割り当たる(同時に2品煮るための前提)
    6. パターン3の設定が、鍋2つ・パターン2と同じ条件になっている

実行方法:
    python tests/repro_two_pots.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from gym_cooking.play_test import MAP_SETTINGS
from gym_cooking.utils.core import Object, Onion
from agent.agent.myagent.CSPAgent import CSPAgent

results = []


def check(label, ok, detail=""):
    results.append(ok)
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}{(' -> ' + detail) if detail else ''}")


def pots_of(map_key):
    env = OvercookedEnvironment(MapSetting(**dict(MAP_SETTINGS[map_key])))
    env.reset()
    return env, [tuple(o.location) for o in env.world.objects.get('Pot', [])]


# 1 + 2
for key in ('exp_ring_2pot', 'exp_ring_2pot_veg',
            'exp_partition_2pot', 'exp_partition_2pot_veg'):
    _env, locs = pots_of(key)
    near = (len(locs) == 2
            and abs(locs[0][0] - locs[1][0]) + abs(locs[0][1] - locs[1][1]) == 1)
    check(f'{key}: 鍋が2つ、隣どうし', near, str(locs))

for key in ('exp_ring', 'exp_partition', 'exp_ring_veg', 'exp_partition_veg'):
    _env, locs = pots_of(key)
    check(f'{key}: 鍋はこれまでどおり1つ', len(locs) == 1, str(locs))


class FakePot:
    pass


FakePot.__name__ = 'Pot'


class FakeEnv:
    def __init__(self, busy):
        self.pos_gs = {(0, 7): FakePot(), (0, 8): FakePot()}
        self.pos_obj = {loc: object() for loc in busy}


ai = CSPAgent.__new__(CSPAgent)
# 3
check('片方が空いていれば調理を始められる',
      ai._station_is_free(FakeEnv([(0, 7)]), 'Pot') is True)
check('両方空いていれば始められる',
      ai._station_is_free(FakeEnv([]), 'Pot') is True)
# 4
check('全部ふさがっていれば始められない',
      ai._station_is_free(FakeEnv([(0, 7), (0, 8)]), 'Pot') is False)

# 5. 注文ごとに違う鍋が割り当たる(スープを2品出して確かめる)
kw = dict(MAP_SETTINGS['exp_ring_2pot_veg'])
kw['order_recipes'] = ('TomatoLettuceSoup', 'OnionTomatoSoup', 'OnionLettuceSalad')
env = OvercookedEnvironment(MapSetting(**kw))
env.reset()
locs = [tuple(o.location) for o in env.world.objects.get('Pot', [])]
# 鍋を使う注文が2件ある状態にする
probe = CSPAgent(10, None, sc_2agent=True, skip_budget=0)
probe.human_counterpart_mode = True
probe.own_agent_idx = 0
probe.priority_weights = {}
probe.active_constraints = []
probe.debug_counter_trace = False
from agent.agent.executor.low import EnvState
info = env.get_ai_info()
st = EnvState(world=info['world'], agents=info['sim_agents'], agent_idx=0,
              order=info['order_scheduler'], event_history=info['event_history'],
              time=info['current_time'], chg_grid=info['chg_grid'])
res = probe._get_resources(st)
check('AI から鍋が2つ見えている', len(res.get('pots', [])) == 2,
      str(res.get('pots')))

orders = probe._build_order_tasks(st)
probe._annotate_task_geometry(st, [t for o in orders for t in o['tasks']], st.self_pos)
cook_pots = [t.get('fixed_res') for o in orders for t in o['tasks']
             if t.get('verb') == 'cook']
uniq = {r[1] for r in cook_pots if r and r[0] == 'pot'}
check('スープが2品ぶん計画に入っている', len(cook_pots) >= 2, f'{cook_pots}')
check('調理の工程が鍋を使い分けている', len(uniq) == 2, f'{cook_pots}')

# 6. パターン3の設定
import server as srv
p3 = srv.EXPERIMENT_PATTERNS.get(3) or {}
check('パターン3がある', bool(p3))
check('パターン3は鍋2つ', p3.get('pots') == 2, str(p3.get('pots')))
check('パターン3はエンドレス90秒・3工程ごと',
      p3.get('endless') and p3.get('seconds') == 90
      and p3.get('instruct_every') == 3)
check('パターン3の条件はパターン2と同じ4つ',
      srv.PATTERN_SKIP_BUDGETS.get(3) == srv.PATTERN_SKIP_BUDGETS.get(2),
      str(srv.PATTERN_SKIP_BUDGETS.get(3)))
check('パターン1・2は鍋1つのまま',
      srv.EXPERIMENT_PATTERNS[1].get('pots', 1) == 1
      and srv.EXPERIMENT_PATTERNS[2].get('pots', 1) == 1)

ok = sum(1 for r in results if r)
print()
if ok == len(results):
    print(f'[SUCCESS] {ok}/{len(results)} 件が期待どおり')
else:
    print(f'[FAILURE] {ok}/{len(results)} 件のみ期待どおり')
    sys.exit(1)
