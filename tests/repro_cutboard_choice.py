"""「切る」工程でどのまな板を使うかを CSP が決めることの検証。

以前は「材料の置き場からいちばん近いまな板」に決め打ちだった。リング
(鍋2つ)の地図では、どの材料から見ても同じまな板がいちばん近いので、
7つの「切る」が全部1つのまな板に並び、もう1つは計画の上で使われなかった
(同じまな板の工程は時間を重ねられないので、2人いても順番待ちになる)。

確かめること:
    1. まな板を選べる工程がある
    2. 計画が2つのまな板を使い分ける
    3. 同じまな板を使う工程は、時間が重ならない
    4. 決め打ち(以前の形)より、完了時間が長くならない
    5. 決め打ちに戻すと、以前と同じ計画になる(切り替えが効く)
    6. 指示(鎖)で縛ったときも解ける

実行方法:
    python tests/repro_cutboard_choice.py
"""
import os
import sys
from copy import deepcopy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'tools'))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
import survey_chain_loss as S

ORDERS = ('TomatoLettuceSoup', 'FullSoup', 'OnionLettuceSalad')
results = []


def check(label, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}{(' -> ' + detail) if detail else ''}")


def plan(state, orders, choose, payload=None, budget=None):
    ai = S.new_ai(budget)
    ai.choose_cutboard = choose
    st = deepcopy(state)
    if payload is not None:
        pend = {'task': (payload['verb'], deepcopy(payload)), 'status': 'pending', 'id': 'x',
                'skip_budget': budget, 'remaining_skip_budget': budget, 'target_idx': 0}
        ai._pending_instructions = [pend]
        st._pending_instructions = [pend]
    ai.solve_csp_scheduling(st, orders=deepcopy(orders))
    chops = [t for sched in (ai.schedule_per_agent or {}).values() for t in sched
             if t['id'][0] == 'chop']
    return ai, chops


def main():
    env = OvercookedEnvironment(MapSetting(level='exp_ring_2pot_veg', order_recipes=ORDERS))
    env.reset()
    state = S.state_of(env)
    orders = S.new_ai(None)._build_order_tasks(deepcopy(state))

    ai, chops = plan(state, orders, True)
    m = ai._last_solve_metrics
    print(f'[SETUP] 注文={ORDERS} 切る工程={len(chops)}')
    check('まな板を選べる工程がある', (m.get('board_choice_tasks') or 0) > 0,
          str(m.get('board_choice_tasks')))
    boards = sorted({tuple(t['res'][1]) for t in chops})
    check('計画が2つのまな板を使い分ける', len(boards) == 2, str(boards))
    clash = []
    for i, a in enumerate(chops):
        for b in chops[i + 1:]:
            if a['res'][1] == b['res'][1] and a['start'] < b['end'] and b['start'] < a['end']:
                clash.append((a['id'], b['id']))
    check('同じまな板を使う工程は、時間が重ならない', not clash, str(clash))

    old, old_chops = plan(state, orders, False)
    f_new, f_old = m['makespan_frames'], old._last_solve_metrics['makespan_frames']
    check('決め打ちより完了時間が長くならない', f_new <= f_old,
          f'選ばせる {f_new / ai.fps:.1f}秒 / 決め打ち {f_old / old.fps:.1f}秒')
    check('決め打ちに戻すと、まな板は1つだけ(以前と同じ)',
          len({tuple(t['res'][1]) for t in old_chops}) == 1
          and not old._last_solve_metrics.get('board_choice_tasks'))

    probe = S.new_ai(0)
    probe.instruction_scope = 'dish'
    cands = probe.get_instruction_candidates(deepcopy(state))
    for display, payload in cands:
        a, _c = plan(state, orders, True, payload, 0)
        b, _c = plan(state, orders, False, payload, 0)
        fa = a._last_solve_metrics.get('makespan_frames')
        fb = b._last_solve_metrics.get('makespan_frames')
        check(f'指示 {display} (d=0) でも解けて、決め打ちより長くならない',
              fa is not None and fb is not None and fa <= fb,
              f'{fa} / {fb}')

    print()
    print(f"[{'SUCCESS' if all(results) else 'FAIL'}] "
          f"{results.count(True)}/{len(results)} 件が期待どおり")
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
