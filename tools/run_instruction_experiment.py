"""指示の質・skip_budget と効率損失の関係を、人間なしで自動計測する(段階A)。

仮説H2:
  対象タスクの「自然順位」(指示なしの最適スケジュールで、担当エージェントの
  作業列の何番目に予定されていたか)が skip_budget を上回ったときにだけ効率損失
  が発生し、その超過分が大きいほど損失も大きくなる。

計測はソルバー水準で行う。1試行につき同じ初期条件を2回解くだけ:
  f      = 指示制約なしの最適 makespan   -> 自然順位もここから読む
  f'(d)  = 「対象タスクの前に同じエージェントが実行してよい他タスクは d 個まで」
           という制約ありの最適 makespan
  L(d)   = f'(d) - f

エピソードを最後まで走らせる必要があるのは「全注文を提供できたか」だけで、
H2 自体はスケジューリングの仮説なのでここまでで完結する(1試行 0.1 秒程度)。

    python tools/run_instruction_experiment.py --map ring --seeds 30

ゲーム側のコードには一切手を入れず、観測だけを行う。
"""
import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT / 'agent', ROOT / 'testbed-cooking'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from copy import deepcopy as dcopy  # noqa: E402

from ortools.sat.python import cp_model  # noqa: E402

from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting  # noqa: E402
from gym_cooking.play_test import MAP_SETTINGS  # noqa: E402
from gym_cooking.utils.order_preset import generate_order_recipes  # noqa: E402
from gym_cooking.utils.replay import Replay  # noqa: E402
from agent.executor.low import EnvState  # noqa: E402
from agent.myagent.CSPAgent import CSPAgent  # noqa: E402

QUALITIES = ('good', 'bad', 'random')
# 実測(段階A・第1回)で d=1 と d=2 はLが変わる試行が9%しかなく、
# 水準として機能していなかった。自然順位の中央値が
# 良い=0.5 / ランダム=2.0 / 悪い=3.0 なので、その分布を跨ぐ 0/2/4 にする。
SKIP_BUDGETS = (0, 2, 4)

# ソルバーの解を再現可能にする。既定は複数ワーカーで走るため、同点の最適解の
# うちどれが返るかが実行ごとに変わりうる。makespan は変わらないが「自然順位」は
# 割り当てに依存するので、試行の再現性のためワーカー1本・固定シードで解く。
_SOLVE_TIMES = []
_orig_solve = cp_model.CpSolver.Solve


def _deterministic_solve(self, model, *a, **kw):
    self.parameters.num_workers = 1
    self.parameters.random_seed = 0
    t = time.perf_counter()
    status = _orig_solve(self, model, *a, **kw)
    _SOLVE_TIMES.append(time.perf_counter() - t)
    return status


cp_model.CpSolver.Solve = _deterministic_solve


def make_agent():
    ai = CSPAgent(10, Replay(), sc_2agent=True)
    # 人間役スロットの計画も立てる(実験と同じ2エージェント構成)。
    ai.human_counterpart_mode = True
    ai.own_agent_idx = 0
    ai.priority_weights = {}
    ai.gui_text_input = ""
    ai.gui_constraint_input = ""
    ai.active_constraints = []
    ai.debug_counter_trace = False
    return ai


def make_env(map_name, seed, preset='experiment1'):
    kw = dict(MAP_SETTINGS[map_name])
    kw['order_recipes'] = generate_order_recipes(preset, random.Random(seed))
    env = OvercookedEnvironment(MapSetting(**kw))
    env.reset()
    return env


def snapshot(env):
    i = env.get_ai_info()
    return EnvState(world=i['world'], agents=i['sim_agents'], agent_idx=0,
                    order=i['order_scheduler'], event_history=i['event_history'],
                    time=i['current_time'], chg_grid=i['chg_grid'])


def solve(agent, env_state, orders, skip_budget, pending=None):
    """1回解いて、スケジュールと評価値を返す。orders は毎回複製して渡す。"""
    agent.skip_budget = skip_budget
    agent._pending_instructions = [dcopy(pending)] if pending else []
    env_copy = dcopy(env_state)
    env_copy._pending_instructions = [dcopy(pending)] if pending else []

    n_before = len(_SOLVE_TIMES)
    agent.solve_csp_scheduling(env_copy, orders=dcopy(orders))
    elapsed = sum(_SOLVE_TIMES[n_before:])

    metrics = dict(getattr(agent, '_last_solve_metrics', {}) or {})
    metrics['solve_seconds'] = elapsed
    metrics['schedule_per_agent'] = dcopy(getattr(agent, 'schedule_per_agent', None) or {})
    return metrics


def find_rank(agent, schedule_per_agent, fixed_task_ids):
    """担当エージェントの作業列で、対象タスクが何番目かを返す。

    候補は (verb, obj) 単位でまとめられており、複数の注文にまたがることが
    ある。制約は「そのうちどれか1つを d 以内に」なので、最も早いものを取る。
    """
    best = None
    for agent_idx in (0, 1):
        sched = schedule_per_agent.get(agent_idx, [])
        for fid in fixed_task_ids:
            idx = agent._find_schedule_index_by_fixed_id(sched, fid)
            if idx is None:
                continue
            if best is None or idx < best[0]:
                best = (idx, agent_idx)
    return best if best else (None, None)


def order_uids_with_verb(orders, verb):
    """その工程を含む注文の order_uid 集合。"""
    return {t['order'] for o in orders for t in o['tasks'] if t['verb'] == verb}


def soup_order_uids(orders):
    """スープ(鍋を使う注文)の order_uid 集合。良い指示の判定に使う。"""
    return order_uids_with_verb(orders, 'cook')


def juice_order_uids(orders):
    """ジュース(ミキサーを使う注文)の order_uid 集合。悪い指示の判定に使う。"""
    return order_uids_with_verb(orders, 'mix')


def pick_target(candidates, quality, soup_uids, rng, bad_uids=None):
    """指示の質に応じて対象タスクを1つ選ぶ。

    good : スープ(ボトルネック)に寄与する下ごしらえ(chop)。全体最適の指示。
    bad  : スープに寄与しない下ごしらえ。目先の都合を優先した指示。
           ジュースがある構成では「ジュースの注文に属する chop」を選ぶ
           (人間が自分のミキサー作業を早く始めたい、という動機のある悪い指示)。
           ジュースが無い構成では、従来どおりサラダ専用の chop。
    random: 候補から一様に1つ。
    """
    if quality == 'random':
        return rng.choice(candidates) if candidates else None

    # good/bad はどちらも「下ごしらえ(chop)」に揃える。盛り付け(serve)は
    # 依存関係の都合で自然順位が常に最後になり、指示の質ではなくタスク種別で
    # 損失が決まってしまうため、good/bad の対象からは外す。
    chops = [(d, p) for d, p in candidates if p['verb'] == 'chop']
    if quality == 'good':
        pool = [c for c in chops if soup_uids & set(c[1]['order_uids'])]
    elif bad_uids:
        pool = [c for c in chops
                if (bad_uids & set(c[1]['order_uids']))
                and not (soup_uids & set(c[1]['order_uids']))]
    else:
        pool = [c for c in chops if not (soup_uids & set(c[1]['order_uids']))]
    return rng.choice(pool) if pool else None


def seed_is_usable(map_name, seed, preset='experiment1'):
    """good と bad の両方を chop で選べる初期条件かどうか。

    スープが3種の具材を使う(FullSoup)と、すべての chop がスープにも寄与する
    ため「サラダ専用の下ごしらえ」が存在せず、bad が定義できない。
    そういう注文構成は条件として成立しないので、シードごと除外する。
    """
    env = make_env(map_name, seed, preset)
    agent = make_agent()
    state = snapshot(env)
    orders = agent._build_order_tasks(dcopy(state))
    if not any(o['tasks'] for o in orders):
        return False
    candidates = agent.get_instruction_candidates(dcopy(state))
    soup_uids = soup_order_uids(orders)
    bad_uids = juice_order_uids(orders)
    rng = random.Random(0)
    return all(pick_target(candidates, q, soup_uids, rng, bad_uids) is not None
               for q in ('good', 'bad'))


def run_trial(map_name, seed, quality, skip_budget, preset='experiment1'):
    env = make_env(map_name, seed, preset)
    agent = make_agent()
    state = snapshot(env)
    orders = agent._build_order_tasks(dcopy(state))
    tasks_total = sum(len(o['tasks']) for o in orders)

    row = {
        'map': map_name, 'preset': preset, 'seed': seed, 'quality': quality,
        'skip_budget': skip_budget,
        'orders': '|'.join(env.arglist.order_recipes),
        'num_tasks': tasks_total,
    }

    if tasks_total == 0:
        row['status'] = 'no_tasks'
        return row

    # --- f: 指示制約なし。自然順位もここから読む -------------------------
    base = solve(agent, state, orders, skip_budget=None)
    row['baseline_status'] = base.get('status')
    row['baseline_solve_s'] = round(base.get('solve_seconds', 0), 4)
    if base.get('makespan_frames') is None:
        row['status'] = f"baseline_{base.get('status')}"
        return row
    makespan_base = base['makespan_frames'] / 10.0
    row['makespan_baseline_s'] = round(makespan_base, 2)

    # --- 指示対象を選ぶ ---------------------------------------------------
    candidates = agent.get_instruction_candidates(dcopy(state))
    soup_uids = soup_order_uids(orders)
    bad_uids = juice_order_uids(orders)
    rng = random.Random(f'{map_name}-{seed}-{quality}')
    target = pick_target(candidates, quality, soup_uids, rng, bad_uids)
    if target is None:
        row['status'] = 'no_candidate_for_quality'
        row['num_candidates'] = len(candidates)
        return row

    display, payload = target
    row['target'] = display
    row['target_verb'] = payload['verb']
    row['target_obj'] = payload['obj']
    row['target_serves_soup'] = int(bool(soup_uids & set(payload['order_uids'])))
    row['num_candidates'] = len(candidates)

    rank, rank_agent = find_rank(agent, base['schedule_per_agent'], payload['fixed_task_ids'])
    row['natural_rank'] = rank
    row['natural_rank_agent'] = rank_agent
    row['excess'] = None if rank is None else max(0, rank - skip_budget)

    # --- f'(d): 指示制約あり ----------------------------------------------
    pending = {
        'id': float(seed), 'task': payload, 'target_idx': 0,
        'accepted_env_time': state.time, 'status': 'pending',
        'skip_budget': skip_budget, 'remaining_skip_budget': skip_budget,
    }
    cons = solve(agent, state, orders, skip_budget=skip_budget, pending=pending)
    row['constrained_status'] = cons.get('status')
    row['constrained_solve_s'] = round(cons.get('solve_seconds', 0), 4)
    if cons.get('makespan_frames') is None:
        row['status'] = 'infeasible' if cons.get('status') == 'INFEASIBLE' \
            else f"constrained_{cons.get('status')}"
        return row

    makespan_cons = cons['makespan_frames'] / 10.0
    row['makespan_constrained_s'] = round(makespan_cons, 2)
    row['loss_s'] = round(makespan_cons - makespan_base, 2)

    # --- 実際に前へ挟まった他タスク ---------------------------------------
    idx, a_idx = find_rank(agent, cons['schedule_per_agent'], payload['fixed_task_ids'])
    row['constrained_rank'] = idx
    if idx is not None:
        before = cons['schedule_per_agent'][a_idx][:idx]
        row['inserted_count'] = len(before)
        durs = [(t['end'] - t['start']) / 10.0 for t in before]
        row['inserted_total_s'] = round(sum(durs), 2)
        row['inserted_durations_s'] = '|'.join(f'{d:.1f}' for d in durs)
    row['status'] = 'ok'
    return row


FIELDS = ['map', 'preset', 'seed', 'quality', 'skip_budget', 'status', 'orders', 'num_tasks',
          'num_candidates', 'target', 'target_verb', 'target_obj', 'target_serves_soup',
          'natural_rank', 'natural_rank_agent', 'excess',
          'makespan_baseline_s', 'makespan_constrained_s', 'loss_s',
          'constrained_rank', 'inserted_count', 'inserted_total_s', 'inserted_durations_s',
          'baseline_status', 'constrained_status', 'baseline_solve_s', 'constrained_solve_s']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', default='ring', help='比較するマップ名(カンマ区切りで複数可)')
    ap.add_argument('--seeds', type=int, default=30, help='各条件あたりの試行数')
    ap.add_argument('--preset', default='experiment1',
                    help='注文プリセット(experiment1=サラダ2+スープ1 / experiment2=サラダ1+スープ1+ジュース1)')
    ap.add_argument('--out', default=str(ROOT / 'results' / 'instruction_experiment.csv'))
    args = ap.parse_args()

    maps = [m.strip() for m in args.map.split(',') if m.strip()]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    started = time.time()
    total = len(maps) * args.seeds * len(QUALITIES) * len(SKIP_BUDGETS)
    done = 0
    for map_name in maps:
        # 条件として成立するシードだけを、必要数そろうまで拾う。
        seeds, probe, skipped = [], 0, 0
        while len(seeds) < args.seeds and probe < args.seeds * 20:
            if seed_is_usable(map_name, probe, args.preset):
                seeds.append(probe)
            else:
                skipped += 1
            probe += 1
        print(f'[{map_name}] 使用シード {len(seeds)} 件 '
              f'(bad が定義できず除外 {skipped} 件)', flush=True)

        for seed in seeds:
            for quality in QUALITIES:
                for d in SKIP_BUDGETS:
                    rows.append(run_trial(map_name, seed, quality, d, args.preset))
                    done += 1
                    if done % 25 == 0 or done == total:
                        el = time.time() - started
                        print(f'  {done}/{total} 件 ({el:.0f}秒経過, '
                              f'残り約{el / done * (total - done):.0f}秒)', flush=True)

    with out.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

    ok = sum(1 for r in rows if r.get('status') == 'ok')
    print(f'\n{len(rows)} 試行 (正常 {ok} / それ以外 {len(rows) - ok})')
    print(f'所要 {time.time() - started:.0f} 秒 -> {out}')


if __name__ == '__main__':
    main()
