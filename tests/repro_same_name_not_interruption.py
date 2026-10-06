"""同名の工程(別の注文の「レタスを切る」)は割り込みと数えないことの検証。

想定(2026-10-06):
    注文: トマトレタスサラダ / たまねぎレタススープ / たまねぎトマトスープ。
    「サラダを最後まで作って」(d=0)。鎖は 切る(レタス, 注文0)・切る(トマト, 注文0)・盛る。
    スープ用の id が付いた「レタスを切る」「トマトを切る」を先にやっても、切った
    材料はサラダに使えるので割り込みではない。
      - ソルバー: d=0 でも同名の工程は鎖の前に置ける(縛られない)
      - 同名でない工程(たまねぎを切る・煮る)は d=0 では鎖の前に置けない
      - 数え方: 同名の工程が消えても「挟んだ数」は増えない

実行方法:
    python tests/repro_same_name_not_interruption.py
"""
import os
import sys
from copy import deepcopy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'agent'))
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from gym_cooking.utils.replay import Replay
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


def new_ai(budget):
    ai = CSPAgent(10, Replay(), sc_2agent=True, skip_budget=budget)
    ai.human_counterpart_mode = True
    ai.two_agent_assignment = True
    ai.own_agent_idx = 0
    ai.priority_weights = {}
    ai.gui_text_input = ''
    ai.gui_constraint_input = ''
    ai.active_constraints = []
    ai.debug_counter_trace = False
    return ai


def main():
    env = OvercookedEnvironment(MapSetting(level='exp_ring_2pot_veg',
                                           order_recipes=('TomatoLettuceSalad', 'OnionLettuceSoup', 'OnionTomatoSoup')))
    env.reset()
    st = state_of(env)
    ai = new_ai(0)
    cands = {str(d): p for d, p in ai.get_instruction_candidates(deepcopy(st))}
    p = cands['serve_salad_lettuce_tomatosalad']
    pend = {'id': 1.0, 'task': (p['verb'], p), 'target_idx': 0, 'accepted_env_time': 0.0,
            'status': 'pending', 'skip_budget': 0, 'remaining_skip_budget': 0}
    ai._pending_instructions = [deepcopy(pend)]
    ai(deepcopy(st))
    own = ai.schedule_per_agent.get(0) or []
    chain = {(c[1], c[2], c[3]) for c in p['chain']}
    names = {(c[1], c[2]) for c in p['chain']}
    print('  AI の計画:', [f"{t['id'][0]} {t['id'][1].split()[0]} u{t['id'][2]}" for t in own])

    print('[1] d=0: 鎖の「切る」は同名ならどの注文の id でもよく、必要数(各1つ)だけ切って盛る')
    serve_k = next((k for k, t in enumerate(own) if t['id'][0] == 'serve_salad'), None)
    check('盛る工程が AI の計画にある', serve_k is not None)
    pre = [t['id'] for t in own[:serve_k or 0]]
    foreign = [t for t in pre if (t[0], t[1]) not in names]
    check('盛る前に同名でない工程(たまねぎ・煮る)は無い', not foreign, str(foreign))
    check('盛る前の切るはレタス1つ・トマト1つだけ(同名を余分に切らない)',
          sorted((t[0], t[1]) for t in pre) == sorted(n for n in names if n[0] == 'chop'), str(pre))

    print('[2] 数え方: 同名の工程が消えても挟んだ数は増えない')
    pd = ai._pending_instructions[0]
    # 前回の計画の名前を覚えた状態で、スープ用の「レタスを切る」が消えた世界を見せる
    ai._track_instruction_progress(st, [t for o in ai._build_order_tasks(deepcopy(st)) for t in o['tasks']])
    before_cnt = pd.get('_consumed_tasks', 0)
    tasks = [t for o in ai._build_order_tasks(deepcopy(st)) for t in o['tasks']]
    tasks2 = [t for t in tasks if not (t['id'][0] == 'chop' and t['id'][1] == 'lettuce' and t['id'][2] != 0)][:]
    # レタスの切るを1つ減らした一覧(どの注文の id かは問わない)
    ai._track_instruction_progress(st, tasks2)
    check('同名(レタスを切る)が1つ消えても挟んだ数はそのまま', pd.get('_consumed_tasks', 0) == before_cnt,
          f"{before_cnt} -> {pd.get('_consumed_tasks', 0)}")
    tasks3 = [t for t in tasks2 if not (t['id'][0] == 'chop' and t['id'][1] == 'onion')]
    ai._track_instruction_progress(st, tasks3)
    # たまねぎを切るは同名でない。AI の計画に入っていれば数える(相手の分と見た分は除く)
    print('  (参考) たまねぎを切るが消えたあとの挟んだ数:', pd.get('_consumed_tasks', 0))

    ng = results.count(False)
    if ng:
        print(f'[FAILED] {ng}/{len(results)} 件が期待どおりではありません')
        return 1
    print(f'[SUCCESS] {len(results)}/{len(results)} 件すべて期待どおり')
    return 0


if __name__ == '__main__':
    sys.exit(main())
