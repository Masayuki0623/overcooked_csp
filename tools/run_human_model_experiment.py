"""人間役の行動モデルを変えて、skip_budget と効率の関係を実測する。

前回のシミュレーションは CSP が両方のキャラクターを動かしていた
(human_counterpart_mode=False)ため、2体が同じ最適計画を共有しており、
AI の予測が外れることが原理的に無かった。skip_budget を大きくするほど
効率が良くなるのは、その設定の必然だった。

ここでは human_counterpart_mode=True にして CSP には自分の担当ぶんだけを
動かさせ、人間側は tools/human_models.py の方策で別に動かす。これにより
「AIは人間が最適に動くと予測して計画する / 実際の人間はそう動かない」という
ズレが生じる。このズレが、skip_budget が大きすぎる側で体験が悪化する機序
(仮説H1の右側)になり得るかを見る。

エピソードを実際に走らせる必要がある(ズレは時間発展の中でしか生じない)。

    python tools/run_human_model_experiment.py --map experiment --seeds 30
    python tools/run_human_model_experiment.py --shard 0/8   # 8並列で分割実行
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
for extra in (ROOT / 'agent', ROOT / 'testbed-cooking', ROOT / 'tools'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from copy import deepcopy as dcopy  # noqa: E402

from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting  # noqa: E402
from gym_cooking.play_test import MAP_SETTINGS  # noqa: E402
from gym_cooking.utils.order_preset import generate_order_recipes  # noqa: E402
from gym_cooking.utils.replay import Replay  # noqa: E402
from agent.executor.low import EnvState  # noqa: E402
from agent.myagent.CSPAgent import CSPAgent  # noqa: E402

from human_models import MODELS, HumanModel  # noqa: E402

QUALITIES = ('good', 'bad', 'random')
SKIP_BUDGETS = (0, 2, 4)
MAX_STEPS = 1000   # ゲーム内100秒。max_num_timesteps と同じ。


def make_env(map_name, seed, preset):
    kw = dict(MAP_SETTINGS[map_name])
    kw['order_recipes'] = generate_order_recipes(preset, random.Random(seed))
    env = OvercookedEnvironment(MapSetting(**kw))
    env.reset()
    return env


def make_ai(skip_budget):
    ai = CSPAgent(10, Replay(), sc_2agent=True, skip_budget=skip_budget)
    # 人間側は別方策で動かすので、AI は自分の担当ぶんだけを動かす。
    ai.human_counterpart_mode = True
    ai.own_agent_idx = 0
    ai.priority_weights = {}
    ai.gui_text_input = ""
    ai.gui_constraint_input = ""
    ai.active_constraints = []
    ai.debug_counter_trace = False
    return ai


def state_for(env, idx):
    i = env.get_ai_info()
    return EnvState(world=i['world'], agents=i['sim_agents'], agent_idx=idx,
                    order=i['order_scheduler'], event_history=i['event_history'],
                    time=i['current_time'], chg_grid=i['chg_grid'])


def pick_instruction(ai, state, orders, quality, rng):
    """指示の質に応じて対象タスクを選ぶ(前回のハーネスと同じ規則)。"""
    candidates = ai.get_instruction_candidates(dcopy(state))
    if not candidates:
        return None
    soup = {t['order'] for o in orders for t in o['tasks'] if t['verb'] == 'cook'}
    juice = {t['order'] for o in orders for t in o['tasks'] if t['verb'] == 'mix'}
    if quality == 'random':
        return rng.choice(candidates)
    chops = [c for c in candidates if c[1]['verb'] == 'chop']
    if quality == 'good':
        pool = [c for c in chops if soup & set(c[1]['order_uids'])]
    else:
        pool = [c for c in chops
                if (juice & set(c[1]['order_uids'])) and not (soup & set(c[1]['order_uids']))]
        if not pool:
            pool = [c for c in chops if not (soup & set(c[1]['order_uids']))]
    return rng.choice(pool) if pool else None


def run_trial(map_name, preset, seed, human_model, quality, skip_budget):
    env = make_env(map_name, seed, preset)
    ai = make_ai(skip_budget)
    human_idx = 1
    human = HumanModel(human_model, ai, human_idx, Replay(), seed=seed * 31 + 7)

    row = {
        'map': map_name, 'preset': preset, 'seed': seed,
        'human_model': human_model, 'quality': quality, 'skip_budget': skip_budget,
        'orders': '|'.join(env.arglist.order_recipes),
    }

    state = state_for(env, 0)
    orders = ai._build_order_tasks(dcopy(state))
    if not any(o['tasks'] for o in orders):
        row['status'] = 'no_tasks'
        return row

    # --- 開始直後に1回だけ指示を出す(実験と同じ once_at_start 相当) ------
    rng = random.Random(f'{map_name}-{seed}-{quality}')
    target = pick_instruction(ai, state, orders, quality, rng)
    if target is None:
        row['status'] = 'no_candidate'
        return row
    display, payload = target
    row['target'] = display
    row['target_verb'] = payload['verb']
    row['target_obj'] = payload['obj']

    pending = {
        'id': float(seed), 'task': payload, 'target_idx': 0,
        'accepted_env_time': 0.0, 'status': 'pending',
        'skip_budget': skip_budget, 'remaining_skip_budget': skip_budget,
    }
    env._pending_instructions = [dcopy(pending)]
    ai._pending_instructions = [dcopy(pending)]

    # --- 効率損失 L: 指示ありとなしの最適 makespan の差(前回と同じ定義) ---
    loss = ai.estimate_instruction_time_loss(dcopy(state), pending, skip_budget=skip_budget)
    row['loss_s'] = loss.get('loss_seconds')
    row['makespan_plan_baseline_s'] = loss.get('baseline_seconds')
    row['makespan_plan_constrained_s'] = loss.get('constrained_seconds')
    row['loss_status'] = loss.get('status')

    # --- エピソード実行 ---------------------------------------------------
    started = time.time()
    steps = 0
    for steps in range(1, MAX_STEPS + 1):
        move, _ = ai(dcopy(state_for(env, 0)))
        actions = {a.name: (0, 0) for a in env.sim_agents}
        if isinstance(move, dict):
            for key, m in move.items():
                idx = int(str(key).split('_')[-1])
                if idx < len(env.sim_agents):
                    actions[env.sim_agents[idx].name] = m or (0, 0)
        elif move:
            actions[env.sim_agents[0].name] = move

        h_action, _tid = human.act(state_for(env, human_idx),
                                   env.sim_agents[0].location)
        actions[env.sim_agents[human_idx].name] = h_action

        env.step(actions, passed_time=0.1)
        if not env.order_scheduler.current_orders:
            break

    sched = env.order_scheduler
    row.update({
        'served': sched.successful_orders,
        'reward': sched.reward,
        'orders_left': len(sched.current_orders),
        'makespan_actual_s': round(env.current_time, 1),
        'completed': int(len(sched.current_orders) == 0),
        'timed_out': int(steps >= MAX_STEPS and len(sched.current_orders) > 0),
        'prediction_match': (None if human.prediction_match_rate is None
                             else round(human.prediction_match_rate, 4)),
        'prediction_samples': human.pred_total,
        'wall_seconds': round(time.time() - started, 1),
        'status': 'ok',
    })
    return row


FIELDS = ['map', 'preset', 'seed', 'human_model', 'quality', 'skip_budget', 'status',
          'orders', 'target', 'target_verb', 'target_obj',
          'served', 'reward', 'orders_left', 'completed', 'timed_out',
          'makespan_actual_s', 'makespan_plan_baseline_s', 'makespan_plan_constrained_s',
          'loss_s', 'loss_status', 'prediction_match', 'prediction_samples', 'wall_seconds']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', default='experiment')
    ap.add_argument('--preset', default='experiment2')
    ap.add_argument('--seeds', type=int, default=30)
    ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--shard', default=None,
                    help='"i/n" 形式。試行を n 個に分けて i 番目だけ実行する(並列用)')
    ap.add_argument('--out', default=str(ROOT / 'results' / 'human_model_experiment.csv'))
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(',') if m.strip()]
    combos = [(s, m, q, d)
              for s in range(args.seeds)
              for m in models
              for q in QUALITIES
              for d in SKIP_BUDGETS]

    if args.shard:
        i, n = (int(x) for x in args.shard.split('/'))
        combos = [c for k, c in enumerate(combos) if k % n == i]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    started = time.time()
    for k, (seed, model, quality, budget) in enumerate(combos, 1):
        rows.append(run_trial(args.map, args.preset, seed, model, quality, budget))
        if k % 5 == 0 or k == len(combos):
            el = time.time() - started
            print(f'  {k}/{len(combos)} 件 ({el:.0f}秒経過, '
                  f'残り約{el / k * (len(combos) - k):.0f}秒)', flush=True)

    with out.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

    ok = sum(1 for r in rows if r.get('status') == 'ok')
    done = sum(1 for r in rows if r.get('completed'))
    print(f'\n{len(rows)} 試行 (正常 {ok} / 全注文を提供できた {done})')
    print(f'所要 {time.time() - started:.0f} 秒 -> {out}')


if __name__ == '__main__':
    main()
