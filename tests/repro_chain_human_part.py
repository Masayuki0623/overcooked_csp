"""鎖の中に人にしかできない工程があっても指示でき、人の側にも縛りが入る。

想定(2026-09-30):
    仕切りで AI が鍋側にいるとき、「スープを調理して」の鎖は
    切る(あなた) → 切る(あなた) → 煮る(AI)。人の工程は「AI が必要とする
    時までに済ませる」ものとして計画に入れ、人の側にも割り込み許容数を課す
    (人の工程が終わるまでに、人が鎖の外の工程を挟んでよいのは d 個まで)。

実行方法:
    python tests/repro_chain_human_part.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from gym_cooking.utils.replay import Replay
from agent.agent.executor.low import EnvState
from agent.agent.myagent.CSPAgent import CSPAgent

results = []


def check(label, ok, detail=''):
    results.append(bool(ok))
    print(f"  [{'OK  ' if ok else 'NG  '}] {label}" + (f" -> {detail}" if detail else ''))


def state_of(env):
    i = env.get_ai_info()
    return EnvState(world=i['world'], agents=i['sim_agents'], agent_idx=0,
                    order=i['order_scheduler'], event_history=i['event_history'],
                    time=i['current_time'], chg_grid=i['chg_grid'])


def new_ai(budget=0):
    ai = CSPAgent(10, Replay(), sc_2agent=True, skip_budget=budget)
    ai.human_counterpart_mode = True
    ai.own_agent_idx = 0
    ai.priority_weights = {}
    ai.gui_text_input = ''
    ai.gui_constraint_input = ''
    ai.active_constraints = []
    ai.solve_deterministic_limit = None
    return ai


def plan_with(env, payload, budget):
    ai = new_ai(budget)
    ai._pending_instructions = [{'task': (payload['verb'], payload), 'status': 'pending',
                                 'skip_budget': budget, 'remaining_skip_budget': budget, 'id': 'i1'}]
    ai(state_of(env))
    return ai, ai.schedule_per_agent.get(0) or [], ai.schedule_per_agent.get(1) or []


def main():
    env = OvercookedEnvironment(MapSetting(level='exp_partition',
                                           order_recipes=('TomatoLettuceSalad', 'OnionLettuceSoup', 'AppleBananaJuice')))
    env.reset()
    # AI と人の位置を入れ替える(AI が鍋側)
    a0, a1 = env.sim_agents[0], env.sim_agents[1]
    a0.location, a1.location = a1.location, a0.location
    st = state_of(env)
    cands = {d: p for d, p in new_ai().get_instruction_candidates(st)}
    print('  候補:', sorted(cands))

    print('[1] 「スープを調理して」の鎖に、人が切る工程が入る')
    cook = cands.get('cook_lettuce_onionsoup')
    check('候補にある', cook is not None)
    if cook is None:
        return 1
    chain = cook['chains'][0]
    human = {tuple(c) for c in cook.get('human_ids') or []}
    check('鎖は 切る2 + 煮る', [c[1] for c in chain] == ['chop', 'chop', 'cook'], str([c[1:3] for c in chain]))
    check('切る1つ(レタス)が人の工程', len(human) == 1 and all(c[1] == 'chop' for c in human), str(sorted(human)))

    print('[2] d=0: 人の計画は切るが最初、AI は煮るまで鎖の外の工程をやらない')
    ai, own, other = plan_with(env, cook, 0)
    print('  AI  :', [f'{t["id"][0]} {t["id"][1].split()[0]}' for t in own])
    print('  あなた:', [f'{t["id"][0]} {t["id"][1].split()[0]}' for t in other])
    htids = {(c[1], c[2], c[3]) for c in human}
    hpos = [i for i, t in enumerate(other) if t['id'] in htids]
    check('人の計画に切るがある', len(hpos) == 1, str(hpos))
    check('人の計画で、切るより前に他の工程が無い', hpos and max(hpos) == 0, str(hpos))
    cook_tid = (chain[-1][1], chain[-1][2], chain[-1][3])
    cpos = next((i for i, t in enumerate(own) if t['id'] == cook_tid), None)
    check('AI の計画に煮るがある', cpos is not None)
    ctids = {(c[1], c[2], c[3]) for c in chain}
    foreign_ai = [t['id'] for i, t in enumerate(own) if cpos is not None and i < cpos and t['id'] not in ctids]
    check('AI は煮るより前に鎖の外の工程をやらない', cpos is not None and not foreign_ai, str(foreign_ai))
    ct = next((t for t in own if t['id'] == cook_tid), None)
    hend = max((t['end'] for t in other if t['id'] in htids), default=None)
    check('煮るの開始は人の切るが終わった後', ct is not None and hend is not None and ct['start'] >= hend,
          f"煮る開始={ct and ct.get('start')} 切る終了={hend}")

    print('[3] d=1: 人も AI も1つまで挟める')
    ai, own, other = plan_with(env, cook, 1)
    print('  AI  :', [f'{t["id"][0]} {t["id"][1].split()[0]}' for t in own])
    print('  あなた:', [f'{t["id"][0]} {t["id"][1].split()[0]}' for t in other])
    hpos = [i for i, t in enumerate(other) if t['id'] in htids]
    foreign_h = [t['id'] for i, t in enumerate(other) if i < max(hpos) and t['id'] not in htids] if hpos else ['x', 'x']
    check('人が切るを終えるまでに挟んだ工程は1つ以下', len(foreign_h) <= 1, str(foreign_h))
    cpos = next((i for i, t in enumerate(own) if t['id'] == cook_tid), None)
    foreign_ai = [t['id'] for i, t in enumerate(own) if cpos is not None and i < cpos and t['id'] not in ctids]
    check('AI が煮るまでに挟んだ鎖の外の工程は1つ以下', cpos is not None and len(foreign_ai) <= 1, str(foreign_ai))

    ng = results.count(False)
    if ng:
        print(f'[FAILED] {ng}/{len(results)} 件が期待どおりではありません')
        return 1
    print(f'[SUCCESS] {len(results)}/{len(results)} 件すべて期待どおり')
    return 0


if __name__ == '__main__':
    sys.exit(main())
