"""指示(工程の鎖)のために切った材料が、別の注文に回らないこと。回って切り直したら
割り込みとして数えること。

想定(2026-10-07 のデモ):
    注文: たまねぎレタスサラダ / たまねぎトマトスープ / トマトレタススープ。
    「レタス玉ねぎサラダを作って」(d=0)。AI がサラダ用にレタスを切ったのに、
    在庫の取り合い(どの注文がその在庫を使うかを所要時間で選び直す仕組み)で
    そのレタスがトマトレタススープに回り、AI はサラダ用にレタスをもう一度切った。
    2 回目は鎖の工程として数えられ、挟んだ数は 0 のままだった。

  [1] 守る: AI が鎖のために切ったレタスを持っているとき、それはサラダに充てる
      (サラダのレタスを切る工程は計画に残らず、スープのレタスは別に切る)
  [2] 数える: 鎖のために切った物が使えなくなり、鎖の工程として切り直したら、
      鎖に要る数を超えた分を挟んだ数に入れる

実行方法:
    python tests/repro_chain_lost_ingredient.py
"""
import os
import sys
from copy import deepcopy
from types import SimpleNamespace

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'agent'))
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from gym_cooking.utils import core
from gym_cooking.utils.core import FoodState
from gym_cooking.utils.event import Event
from gym_cooking.utils.replay import Replay
from agent.executor.low import EnvState
from agent.myagent.CSPAgent import CSPAgent

results = []
ORDERS = ('OnionLettuceSalad', 'OnionTomatoSoup', 'TomatoLettuceSoup')
INSTR = 'serve_salad_lettuce_onionsalad'


def check(label, ok, detail=''):
    results.append(bool(ok))
    print(f"  [{'OK  ' if ok else 'NG  '}] {label}" + (f" -> {detail}" if detail else ''))


def state_of(env):
    i = env.get_ai_info()
    return EnvState(world=i['world'], agents=i['sim_agents'], agent_idx=0,
                    order=i['order_scheduler'], event_history=i['event_history'],
                    time=i['current_time'], chg_grid=i['chg_grid'])


def new_ai():
    # 1 つの脳で 2 人を動かす(デモと同じ)
    ai = CSPAgent(10, Replay(), sc_2agent=True, skip_budget=0)
    ai.human_counterpart_mode = False
    ai.partner_is_external = False
    ai.own_agent_idx = 0
    ai.priority_weights = {}
    ai.gui_text_input = ''
    ai.gui_constraint_input = ''
    ai.active_constraints = []
    ai.debug_counter_trace = False
    return ai


def instruct(ai, st):
    cands = {str(d): p for d, p in ai.get_instruction_candidates(deepcopy(st))}
    p = cands[INSTR]
    pend = {'id': 1.0, 'task': (p['verb'], p), 'target_idx': 0, 'accepted_env_time': 0.0,
            'status': 'pending', 'skip_budget': 0, 'remaining_skip_budget': 0}
    ai._pending_instructions = [deepcopy(pend)]
    return p


def main():
    print('[1] デモと同じ条件で、サラダを出すまでに AI がレタスを切るのは 1 回だけ')
    import io
    from contextlib import redirect_stdout
    from gym_cooking.utils.interact import resolve_action
    env = OvercookedEnvironment(MapSetting(level='exp_ring_2pot_veg', order_recipes=ORDERS,
                                           max_num_timesteps=60.0))
    env.reset()
    ai = new_ai()
    instruct(ai, state_of(env))
    names = [a.name for a in env.sim_agents]
    chops, served_at = 0, None
    seen = set()
    for _ in range(int(40.0 / 0.2)):
        with redirect_stdout(io.StringIO()):
            move, _r = ai(deepcopy(state_of(env)))
        acts = {n: (0, 0) for n in names}
        for k, n in (('ai_0', names[0]), ('ai_1', names[1])):
            if isinstance(move, dict) and move.get(k):
                acts[n] = move[k]
        for a in env.sim_agents:
            acts[a.name] = resolve_action(a, env.world, acts[a.name])
        env.step(acts, passed_time=0.2)
        for ev in env._event_history:
            if id(ev) in seen:
                continue
            seen.add(id(ev))
            if ev.playerA != names[0]:
                continue
            if ev.event == 'Pickup_ChoppedLettuce_from_Cutboard':
                chops += 1
            if str(ev.event).startswith('Deliver_ChoppedLettuce-ChoppedOnion'):
                served_at = ev.time
        if served_at is not None:
            break
    pd = (ai._pending_instructions or [{}])[0]
    check('サラダを出した', served_at is not None, served_at)
    check('サラダを出すまでに AI がレタスを切ったのは 1 回', chops == 1, chops)
    check('挟んだ数は 0(d=0)', pd.get('_consumed_tasks', 0) == 0, pd.get('_consumed_tasks'))

    print('[2] 鎖の工程として切り直した分は挟んだ数に入る')
    env2 = OvercookedEnvironment(MapSetting(level='exp_ring_2pot_veg', order_recipes=ORDERS))
    env2.reset()
    ai2 = new_ai()
    p2 = instruct(ai2, state_of(env2))
    tid2 = next((c[1], c[2], c[3]) for c in p2['chain'] if c[1] == 'chop' and c[2] == 'lettuce')
    pd = ai2._pending_instructions[0]
    agents = [SimpleNamespace(name=a.name) for a in env2.sim_agents]
    me = agents[0].name
    fake = SimpleNamespace(agents=agents, event_history=[], time=0.0)
    ai2._note_exec_tid(SimpleNamespace(time=0.0), tid2)
    tasks = [t for o in ai2._build_order_tasks(deepcopy(state_of(env2))) for t in o['tasks']]
    ai2._track_instruction_progress(fake, tasks)          # 最初の呼び出し(基準)
    fake.event_history.append(Event(me, 'Pickup_ChoppedLettuce_from_Cutboard', (6, 6), 1.0))
    fake.time = 1.0
    ai2._track_instruction_progress(fake, tasks)
    check('1 回目(鎖に要る分)は数えない', pd.get('_consumed_tasks', 0) == 0, pd.get('_consumed_tasks'))
    # 1 回目のレタスは別の料理に混ぜられて使えなくなり、鎖の工程として切り直した
    fake.event_history.append(Event(me, 'Pickup_ChoppedLettuce_from_Cutboard', (6, 6), 8.0))
    fake.time = 8.0
    ai2._track_instruction_progress(fake, tasks)
    check('切り直した 2 回目は挟んだ数に入る', pd.get('_consumed_tasks', 0) == 1, pd.get('_consumed_tasks'))
    ai2._track_instruction_progress(fake, tasks)
    check('同じ切り直しを二重に数えない', pd.get('_consumed_tasks', 0) == 1, pd.get('_consumed_tasks'))

    ng = results.count(False)
    if ng:
        print(f'[FAILED] {ng}/{len(results)} 件が期待どおりではありません')
        return 1
    print(f'[SUCCESS] {len(results)}/{len(results)} 件すべて期待どおり')
    return 0


if __name__ == '__main__':
    sys.exit(main())
