"""文章の指示で工程の一部を除く(「スープを作って。材料は切らないで」)。

想定(2026-10-06):
    LLM が exclude_steps=['chop'] を返したら、鎖の切る工程は人の分
    (human_ids / exclude_verbs)になる。計画では AI は切らず、人の切るが
    終わってから煮る → 提供をやる。除く工程が無ければ、いままでどおり。

実行方法:
    python tests/repro_chain_exclude_steps.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'agent'))      # server.py と同じ通し方
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from gym_cooking.utils.replay import Replay
from agent.executor.low import EnvState
from agent.myagent.CSPAgent import CSPAgent
from agent import instruction_nl as NL

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


def tid(c):
    return (c[1], c[2], c[3])


def main():
    env = OvercookedEnvironment(MapSetting(level='exp_ring_2pot',
                                           order_recipes=('OnionTomatoSalad', 'OnionLettuceSoup', 'TomatoLettuceSoup')))
    env.reset()
    st = state_of(env)
    cands = new_ai().get_instruction_candidates(st)
    by_id = {str(d): (d, p) for d, p in cands}
    print('  候補:', sorted(by_id))

    print('[1] normalize: 個数違いを両方選んだら曖昧')
    r = NL.normalize({'valid': True, 'ambiguous': False, 'message': '',
                      'tasks': ['chop_onion_x1', 'chop_onion_x2']}, list(by_id))
    check('曖昧として却下', r['decision'] == 'reject' and r['reject_reason'] == 'ambiguous', r['message'])

    print('[2] 「材料は切らないで」(下の工程を人に残す)は無効。除けるのは提供だけ')
    sid = 'serve_lettuce_tomatosoup'
    check('候補にある', sid in by_id)
    if sid not in by_id:
        return 1
    r = NL.normalize({'valid': True, 'ambiguous': False, 'message': '', 'tasks': [sid],
                      'exclude_steps': ['chop']}, list(by_id), cands)
    check('切るを除くと無効', r['decision'] == 'reject' and r['reject_reason'] == 'invalid', r['message'])
    r = NL.normalize({'valid': True, 'ambiguous': False, 'message': '', 'tasks': [sid],
                      'exclude_steps': ['cook']}, list(by_id), cands)
    check('煮るを除くと無効', r['decision'] == 'reject' and r['reject_reason'] == 'invalid')
    r = NL.normalize({'valid': True, 'ambiguous': False, 'message': '', 'tasks': [sid],
                      'exclude_steps': ['serve']}, list(by_id), cands)
    check('提供を除くのは受ける', r['decision'] == 'accept' and r['exclude'] == ['serve'], str(r))

    print('[2b] compose(計画側の仕組み): 除いた工程は人の分になる')
    label, p = NL.compose(cands, [sid], ['chop'])
    chain = [tuple(c) for c in p['chains'][0]]
    human = {tuple(c) for c in p.get('human_ids') or []}
    check('exclude_verbs に chop', p.get('exclude_verbs') == ['chop'], str(p.get('exclude_verbs')))
    check('切る工程が全部、人の分', human == {c for c in chain if c[1] == 'chop'} and len(human) == 2,
          str(sorted(human)))
    steps = dict(NL.expand_steps(cands, [sid], ['chop']))[sid]
    check('工程の表示に「(あなた)」', sum('(あなた)' in s for s in steps) == 2, str(steps))

    print('[3] 計画: AI は切らず、人の切るのあとに煮る')
    ai, own, other = plan_with(env, p, 0)
    print('  AI  :', [f'{t["id"][0]} {t["id"][1].split()[0]}' for t in own])
    print('  あなた:', [f'{t["id"][0]} {t["id"][1].split()[0]}' for t in other])
    htids = {tid(c) for c in human}
    ai_chops = [t['id'] for t in own if t['id'] in htids]
    check('AI の計画に、その鎖の切るが無い', not ai_chops, str(ai_chops))
    h_chops = [t for t in other if t['id'] in htids]
    check('人の計画に、その鎖の切るが2つ', len(h_chops) == 2, str([t['id'] for t in h_chops]))
    cook_tid = next((tid(c) for c in chain if c[1] == 'cook'), None)
    ct = next((t for t in own if t['id'] == cook_tid), None)
    hend = max((t['end'] for t in h_chops), default=None)
    check('AI が煮る', ct is not None)
    check('煮るの開始は人の切るが終わった後', ct is not None and hend is not None and ct['start'] >= hend,
          f"煮る開始={ct and ct.get('start')} 切る終了={hend}")

    print('[4] 除く工程が無ければ、AI が切る')
    d0, p0 = by_id[sid]
    ai, own, other = plan_with(env, p0, 0)
    print('  AI  :', [f'{t["id"][0]} {t["id"][1].split()[0]}' for t in own])
    chops_ai = [t['id'] for t in own if t['id'] in {tid(c) for c in chain if c[1] == 'chop'}]
    check('AI の計画に切るがある', len(chops_ai) >= 1, str(chops_ai))

    print('[5] 量の下限・上限: AI の工程 2 つ〜全体の半分')
    total = NL.total_steps(cands)
    print('  全体の工程:', total)
    check('全体は 11 工程(サラダ3 + スープ4 + スープ4)', total == 11, str(total))

    def size(tasks, exclude=()):
        r = NL.normalize({'valid': True, 'ambiguous': False, 'tasks': tasks,
                          'exclude_steps': list(exclude), 'message': ''}, list(by_id), cands)
        return r['decision'], r['reject_reason'], NL.ai_steps(cands, tasks, exclude)
    check('たまねぎ1つ(1工程)は少なすぎ', size(['chop_onion_x1'])[1] == 'too_few', str(size(['chop_onion_x1'])))
    check('たまねぎ2つ(2工程)は受ける', size(['chop_onion_x2'])[0] == 'accept', str(size(['chop_onion_x2'])))
    check('スープ1品(4工程)は受ける', size([sid])[0] == 'accept', str(size([sid])))
    check('スープ + たまねぎ1つ(5工程)は受ける', size([sid, 'chop_onion_x1'])[0] == 'accept',
          str(size([sid, 'chop_onion_x1'])))
    two = [sid, 'serve_lettuce_onionsoup']
    check('スープ2品(8工程)は多すぎ', size(two)[1] == 'too_many', str(size(two)))
    check('スープ、提供は人(3工程)は受ける', size([sid], ['serve'])[0] == 'accept', str(size([sid], ['serve'])))
    check('サラダ、提供は人(2工程)は受ける', size(['serve_salad_onion_tomatosalad'], ['serve'])[0] == 'accept',
          str(size(['serve_salad_onion_tomatosalad'], ['serve'])))

    ng = results.count(False)
    if ng:
        print(f'[FAILED] {ng}/{len(results)} 件が期待どおりではありません')
        return 1
    print(f'[SUCCESS] {len(results)}/{len(results)} 件すべて期待どおり')
    return 0


if __name__ == '__main__':
    sys.exit(main())
