"""工程の鎖としての指示(「作って」/ まだできない工程 / 個数つき)の検証。

想定(2026-09-30 の設計、パターン5):
    指示は工程の集合(鎖)として扱う。「スープを作って」はその注文の工程
    すべて、「煮て」は前提の刻む工程も含む。同じ工程が複数あれば個数を
    選ぶ(「トマトを1つ/2つ切って」)。割り込み許容数 d は
      - 指示を受けてから鎖の最後が終わるまでに、AI が鎖の外の工程を
        挟んでよい数(鎖の前でも間でも数える)
      - 煮える待ちの中に丸ごと収まる工程は数えない
      - 煮る と 出す が両方あれば、煮上がった瞬間に鍋から取る

実行方法:
    python tests/repro_chain_instruction.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from gym_cooking.utils.config import COOKING_TIME_SECONDS
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
    sched = ai.schedule_per_agent.get(0) or []
    return ai, sched, [t['id'] for t in sched]


def foreign_before_chain_end(ids, chain_tids, sched, fps):
    """鎖の最後が終わるまでに挟まった、鎖の外の工程(煮える待ちの中は除く)。"""
    last = max(i for i, t in enumerate(ids) if t in chain_tids)
    cook = next((t for t in sched if t['id'] in chain_tids and t['id'][0] == 'cook'), None)
    serve = next((t for t in sched if t['id'] in chain_tids and t['id'][0] == 'serve'), None)
    win = None
    if cook and 'end' in cook:
        w1 = cook['end'] + int(COOKING_TIME_SECONDS * fps)
        if serve and 'start' in serve:
            w1 = min(w1, serve['start'])
        win = (cook['end'], w1)
    out = []
    for i in range(last):
        t = sched[i]
        if t['id'] in chain_tids:
            continue
        if win and t.get('start') is not None and t['start'] >= win[0] and t['end'] <= win[1]:
            continue
        out.append(t['id'])
    return out


def main():
    env = OvercookedEnvironment(MapSetting(level='exp_ring',
                                           order_recipes=('OnionTomatoSoup', 'TomatoLettuceSalad')))
    env.reset()
    ai = new_ai()
    st = state_of(env)
    cands = {d: p for d, p in ai.get_instruction_candidates(st)}
    print('  候補:', sorted(cands))

    print('[1] 候補: 提供まで任せる鎖、まだできない工程、個数つき')
    make = next((p for d, p in cands.items() if p.get('verb') == 'serve' and 'onion' in p['obj']), None)
    cook = next((p for d, p in cands.items() if p.get('verb') == 'cook'), None)
    check('「オニオントマトスープを提供まで任せて」(まだ材料が無い)がある',
          make is not None and make.get('startable') is False)
    check('「作って」は出ない(提供の鎖と同じ中身なので)',
          not any(p.get('verb') == 'make' for p in cands.values()))
    check('「煮て」(まだ材料が無い)が出る', cook is not None and cook.get('startable') is False,
          str(cook and cook.get('startable')))
    check('提供の鎖は 切る2 + 煮る + 出す', make is not None and
          [c[1] for c in make['chain']] == ['chop', 'chop', 'cook', 'serve'],
          str(make and [tuple(c[1:3]) for c in make['chain']]))
    check('「煮て」の鎖には前提の刻む工程が入る', cook is not None and
          [c[1] for c in cook['chain']] == ['chop', 'chop', 'cook'],
          str(cook and [tuple(c[1:3]) for c in cook['chain']]))
    t1 = cands.get('chop_tomato_x1')
    t2 = cands.get('chop_tomato_x2')
    check('トマトは2つあるので「1つ切って」「2つ切って」が出る',
          t1 is not None and t2 is not None and t1['count'] == 1 and t2['count'] == 2 and t2['total'] == 2)
    check('「2つ切って」は2かたまり', t2 is not None and len(t2['chains']) == 2)

    print('[2] 「トマトを2つ切って」 d=0: 2つとも AI、間に何も挟まらない')
    _, sched, ids = plan_with(env, t2, 0)
    print('  AI の計画:', [f'{t[0]} {t[1].split()[0]}' for t in ids])
    tids2 = {(c[1], c[2], c[3]) for c in t2['chain']}
    n_ai = sum(1 for t in ids if t in tids2)
    check('2つの切る工程が両方 AI の計画にある', n_ai == 2, f'{n_ai}/2')
    fb = foreign_before_chain_end(ids, tids2, sched, ai.fps) if n_ai else ['(なし)']
    check('d=0: 2つ目が終わるまでに他の工程が無い', not fb, str(fb))

    print('[3] 「トマトを2つ切って」 d=1: 挟まるのは1つまで')
    _, sched, ids = plan_with(env, t2, 1)
    print('  AI の計画:', [f'{t[0]} {t[1].split()[0]}' for t in ids])
    n_ai = sum(1 for t in ids if t in tids2)
    fb = foreign_before_chain_end(ids, tids2, sched, ai.fps) if n_ai else ['x', 'x']
    check('d=1: 2つ目が終わるまでに挟まった工程は1つ以下', n_ai == 2 and len(fb) <= 1, str(fb))

    print('[4] 「トマトを1つ切って」 d=0: どちらか1つを最初に')
    _, sched, ids = plan_with(env, t1, 0)
    print('  AI の計画:', [f'{t[0]} {t[1].split()[0]}' for t in ids])
    check('最初の工程がトマトを切る', bool(ids) and ids[0] in tids2, str(ids[:1]))

    print('[5] 「提供まで任せて」 d=0: 鎖の外の工程は煮える待ちの中だけ、煮上がった瞬間に取る')
    ai2, sched, ids = plan_with(env, make, 0)
    print('  AI の計画:', [f'{t[0]} {t[1].split()[0]}' for t in ids])
    chain_tids = {(c[1], c[2], c[3]) for c in make['chain']}
    n_ai = sum(1 for t in ids if t in chain_tids)
    check('鎖の工程が全部 AI の計画にある', n_ai == len(chain_tids), f'{n_ai}/{len(chain_tids)}')
    fb = foreign_before_chain_end(ids, chain_tids, sched, ai2.fps) if n_ai else ['x']
    check('煮える待ちの外には他の工程が挟まらない', not fb, str(fb))
    check('指示の記録に鎖の制約が入ったと印が付く',
          ai2._pending_instructions[0].get('chain_constraint_applied') is True)
    cook_t = next((t for t in sched if t['id'][0] == 'cook'), None)
    serve_t = next((t for t in sched if t['id'][0] == 'serve'), None)
    if cook_t and serve_t and 'end' in cook_t and 'start' in serve_t:
        ready = cook_t['end'] + int(COOKING_TIME_SECONDS * ai2.fps)
        _orders = ai2._build_order_tasks(state_of(env))
        _tasks = [x for o in _orders for x in o.get('tasks', [])]
        ai2._annotate_task_geometry(state_of(env), _tasks, tuple(env.sim_agents[0].location))
        _serve = next((x for x in _tasks if x['id'] == serve_t['id']), {})
        off = int(serve_t.get('pot_offset') or _serve.get('pot_offset') or 0)
        check('鍋に着く時刻が煮上がりと一致', serve_t['start'] + off == ready,
              f"着く={serve_t['start'] + off} 煮上がり={ready}")
    else:
        check('煮る・出すが計画にある', False)

    print('[6] L も鎖の制約で測れる')
    ai3 = new_ai()
    r = ai3.estimate_instruction_time_loss(state_of(env), {'task': ('serve', make), 'status': 'pending',
                                                            'skip_budget': 0, 'remaining_skip_budget': 0},
                                           skip_budget=0)
    check('L が出る', r.get('status') == 'ok' and r.get('loss_seconds') is not None,
          f"L={r.get('loss_seconds')} f={r.get('baseline_seconds')} f'={r.get('constrained_seconds')} {r.get('status')}")

    ng = results.count(False)
    if ng:
        print(f'[FAILED] {ng}/{len(results)} 件が期待どおりではありません')
        return 1
    print(f'[SUCCESS] {len(results)}/{len(results)} 件すべて期待どおり')
    return 0


if __name__ == '__main__':
    sys.exit(main())
