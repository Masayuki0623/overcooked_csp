"""工程の鎖としての指示(「作って」/ まだできない工程)の検証。

想定(2026-09-29 の設計):
    指示は工程の集合(鎖)として扱う。「スープを作って」はその注文の工程
    すべて、「煮て」は前提の刻む工程も含む。割り込み許容数は使わず、
      - 鎖の工程は全部 AI がやる
      - 鎖の最初から最後まで、AI は他の工程を挟まない
        (煮える待ちの中に丸ごと収まる工程は自由)
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


def new_ai():
    ai = CSPAgent(10, Replay(), sc_2agent=True, skip_budget=0)
    ai.human_counterpart_mode = True
    ai.own_agent_idx = 0
    ai.priority_weights = {}
    ai.gui_text_input = ''
    ai.gui_constraint_input = ''
    ai.active_constraints = []
    ai.solve_deterministic_limit = None
    return ai


def main():
    env = OvercookedEnvironment(MapSetting(level='exp_ring',
                                           order_recipes=('OnionTomatoSoup', 'TomatoLettuceSalad')))
    env.reset()
    ai = new_ai()
    st = state_of(env)
    cands = {d: p for d, p in ai.get_instruction_candidates(st)}
    print('  候補:', sorted(cands))

    print('[1] 候補に「作って」と、まだできない工程が出る')
    make = next((p for d, p in cands.items() if p.get('verb') == 'make' and 'onion' in p['obj']), None)
    cook = next((p for d, p in cands.items() if p.get('verb') == 'cook'), None)
    check('「オニオントマトスープを作って」がある', make is not None)
    check('「煮て」(まだ材料が無い)が出る', cook is not None and cook.get('startable') is False,
          str(cook and cook.get('startable')))
    check('「作って」の鎖は 切る2 + 煮る + 出す', make is not None and
          [c[1] for c in make['chain']] == ['chop', 'chop', 'cook', 'serve'],
          str(make and [tuple(c[1:3]) for c in make['chain']]))
    check('「煮て」の鎖には前提の刻む工程が入る', cook is not None and
          [c[1] for c in cook['chain']] == ['chop', 'chop', 'cook'],
          str(cook and [tuple(c[1:3]) for c in cook['chain']]))

    print('[2] 「作って」を指示すると、鎖が AI に固定され、間に他の工程が挟まらない')
    ai._pending_instructions = [{'task': ('make', make), 'status': 'pending', 'skip_budget': 0,
                                 'remaining_skip_budget': 0, 'id': 'i1'}]
    ai(st)
    sched = ai.schedule_per_agent.get(0) or []
    ids = [t['id'] for t in sched]
    chain_tids = {(c[1], c[2], c[3]) for c in make['chain']}
    pos = [i for i, tid in enumerate(ids) if tid in chain_tids]
    print('  AI の計画:', [f'{t[0]} {t[1].split()[0]}' for t in ids])
    check('鎖の工程が全部 AI の計画にある', len(pos) == len(chain_tids), f'{len(pos)}/{len(chain_tids)}')
    inside = [ids[i] for i in range(min(pos), max(pos) + 1)] if pos else []
    foreign = [t for t in inside if t not in chain_tids]
    # 煮える待ちの中の工程は許す: 煮る の直後〜出す の直前だけ
    cook_i = next((i for i, t in enumerate(ids) if t[0] == 'cook'), None)
    serve_i = next((i for i, t in enumerate(ids) if t[0] == 'serve'), None)
    foreign_outside_wait = [ids[i] for i in range(min(pos), max(pos) + 1)
                            if ids[i] not in chain_tids and not (cook_i is not None and serve_i is not None and cook_i < i < serve_i)]
    check('鎖の途中に他の工程が挟まらない(煮える待ちの中は除く)', not foreign_outside_wait, str(foreign_outside_wait))
    st2 = state_of(env)
    check('指示の記録に鎖の制約が入ったと印が付く',
          ai._pending_instructions[0].get('chain_constraint_applied') is True)

    print('[3] 煮上がった瞬間に取る(出す工程の鍋到着 = 煮る終了 + 煮込み時間)')
    tasks = [t for t in sched]
    cook_t = next((t for t in tasks if t['id'][0] == 'cook'), None)
    serve_t = next((t for t in tasks if t['id'][0] == 'serve'), None)
    if cook_t and serve_t and 'end' in cook_t and 'start' in serve_t:
        ready = cook_t['end'] + int(COOKING_TIME_SECONDS * ai.fps)
        # 皿を取ってから鍋に着くまで(pot_offset)は工程の位置計算で決まる
        _orders = ai._build_order_tasks(state_of(env))
        _tasks = [x for o in _orders for x in o.get('tasks', [])]
        ai._annotate_task_geometry(state_of(env), _tasks, tuple(env.sim_agents[0].location))
        _serve = next((x for x in _tasks if x['id'] == serve_t['id']), {})
        off = int(serve_t.get('pot_offset') or _serve.get('pot_offset') or 0)
        arrive = serve_t['start'] + off
        check('鍋に着く時刻が煮上がりと一致', arrive == ready, f'着く={arrive} 煮上がり={ready}')
    else:
        print('  (計画に start が無いので時刻の確認は飛ばす)', {k: cook_t and cook_t.get(k) for k in ('start', 'dur')})

    print('[4] L も鎖の制約で測れる')
    ai2 = new_ai()
    r = ai2.estimate_instruction_time_loss(state_of(env), {'task': ('make', make), 'status': 'pending',
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
