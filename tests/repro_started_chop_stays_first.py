"""切り始めた材料を放置して別の材料を取りに行かないことの検証。

想定(2026-10-06、実際のリプレイ 26 試合で 102 回の切り始めのうち 20 回で発生):
    リングの地図で、AI がレタスをまな板に置いて 1 回切った直後に再計画が走る。
    同じ注文の「切る」3つは所要も締切も同じで、ソルバーにとっては並びが
    どれでも同点なので、並びが「トマト → たまねぎ → レタス」に入れ替わる。
    実行側は「その注文で計画に残っている一番手前の工程」しか手をつけない
    ので、切りかけのレタスを置いたままトマトを取りに行く。

    直し: 手をつけた「切る」(手に持っている / まな板に切りかけがある)は、
    再計画のときに AI の先頭に固定する(CSPAgent._started_chop_index)。

実行方法:
    python tests/repro_started_chop_stays_first.py
"""
import os
import sys
from copy import deepcopy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'agent'))
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from gym_cooking.utils.replay import Replay
from gym_cooking.utils.interact import resolve_action
from gym_cooking.utils.core import Cutboard
from gym_cooking.utils import config as game_config
from agent.executor.low import EnvState
from agent.myagent.CSPAgent import CSPAgent

results = []


def check(label, ok, detail=''):
    results.append(bool(ok))
    print(f"  [{'OK  ' if ok else 'NG  '}] {label}" + (f" -> {detail}" if detail else ''))


def state_of(env):
    i = env.get_ai_info()
    return EnvState(world=i['world'], agents=i['sim_agents'], agent_idx=0,
                    order=i['order_scheduler'], event_history=i['event_history'],
                    time=i['current_time'], chg_grid=i['chg_grid'])


def new_ai(budget, pin):
    ai = CSPAgent(10, Replay(), sc_2agent=True, skip_budget=budget)
    ai.human_counterpart_mode = True
    ai.two_agent_assignment = True
    ai.own_agent_idx = 0
    ai.priority_weights = {}
    ai.gui_text_input = ''
    ai.gui_constraint_input = ''
    ai.active_constraints = []
    ai.debug_counter_trace = False
    ai.pin_started_chop = pin
    return ai


def play(pin, seconds=20.0):
    """相方は動かない。AI がまな板に切りかけを残して 2 秒以上離れた回数を数える。"""
    env = OvercookedEnvironment(MapSetting(level='exp_ring_veg',
                                           order_recipes=('FullSalad', 'FullSalad', 'TomatoLettuceSoup'),
                                           max_num_timesteps=60.0))
    env.reset()
    ai = new_ai(0, pin)
    st = state_of(env)
    cands = {str(d): p for d, p in ai.get_instruction_candidates(deepcopy(st))}
    p = cands['serve_salad_lettuce_onion_tomatosalad_x1']
    ai._pending_instructions = [{'id': 1.0, 'task': (p['verb'], p), 'target_idx': 0,
                                 'accepted_env_time': 0.0, 'status': 'pending',
                                 'skip_budget': 0, 'remaining_skip_budget': 0}]
    boards = [(x, y) for x in range(env.world.width) for y in range(env.world.height)
              if isinstance(env.world.get_gridsquare_at((x, y)), Cutboard)]
    step = game_config.seconds_per_step()
    me = env.sim_agents[0]

    def near(loc, b):
        return abs(loc[0] - b[0]) + abs(loc[1] - b[1]) == 1

    def chopping_on(b):
        for o in env.world.get_object_list():
            if (tuple(getattr(o, 'location', ())) == b and not getattr(o, 'is_held', False)
                    and 'Chopping' in (getattr(o, 'full_name', '') or '')):
                return o.full_name
        return None

    last_adj = {b: None for b in boards}
    abandons = []
    starts = 0
    seen = {b: False for b in boards}
    for _ in range(int(seconds / step)):
        move, _reason = ai(deepcopy(state_of(env)))
        acts = {a.name: (0, 0) for a in env.sim_agents}
        own = move.get('ai_0') if isinstance(move, dict) else move
        if own:
            acts[me.name] = own
        for a in env.sim_agents:
            acts[a.name] = resolve_action(a, env.world, acts[a.name])
        env.step(acts, passed_time=step)
        t = env.current_time
        loc = tuple(me.location)
        for b in boards:
            name = chopping_on(b)
            if name:
                if not seen[b]:
                    seen[b] = True
                    starts += 1
                if near(loc, b):
                    last_adj[b] = t
                elif last_adj[b] is not None and t - last_adj[b] > 2.0:
                    abandons.append((round(t, 1), b, name))
                    last_adj[b] = None
            else:
                seen[b] = False
        if not env.order_scheduler.current_orders:
            break
    return starts, abandons, env.current_time


def main():
    print('[1] 固定なし(以前の動き): 切りかけを放置する')
    starts, ab, t = play(pin=False)
    print(f'  切り始め {starts} 回 / 放置 {len(ab)} 回 {ab} / {t:.1f} 秒')
    # 同名の工程を割り込みと数えなくした(2026-10-06)あとは、この構成では
    # 並びが安定して再現しないことがある。参考値として出すだけで、検査しない
    print('  (参考: 固定なしで放置が起きれば、以前の動きの再現)')

    print('[2] 固定あり: 切り始めた材料は切り終えるまで離れない')
    starts, ab, t = play(pin=True)
    print(f'  切り始め {starts} 回 / 放置 {len(ab)} 回 {ab} / {t:.1f} 秒')
    check('放置が無い', len(ab) == 0, str(ab))
    check('切る作業は進んでいる', starts >= 2, str(starts))

    ng = results.count(False)
    if ng:
        print(f'[FAILED] {ng}/{len(results)} 件が期待どおりではありません')
        return 1
    print(f'[SUCCESS] {len(results)}/{len(results)} 件すべて期待どおり')
    return 0


if __name__ == '__main__':
    sys.exit(main())
