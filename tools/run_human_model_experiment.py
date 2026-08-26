"""人間役の行動モデルを変えて、skip_budget と効率の関係を実測する。

前回のシミュレーションは CSP が両方のキャラクターを動かしていたため、
2体が同じ最適計画を共有しており、計画が外れることが原理的に無かった。
skip_budget を大きくするほど効率が良くなるのは、その設定の必然だった。

ここでは AI に「相手も CSP として全体最適に動く」と仮定させたまま
(human_counterpart_mode=False で2体分の計画を立てる)、実行に使うのは
AI 自身の分だけにして、人間側は tools/human_models.py の方策で動かす。
つまり「AIは相手が賢いと信じて計画を立てるが、実際の相手はそう動かない」。
このズレが、skip_budget が大きすぎる側で体験が悪化する機序(仮説H1の右側)に
なり得るかを見る。

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
from gym_cooking.utils.order_preset import (  # noqa: E402
    enumerate_order_recipes, generate_order_recipes)
from gym_cooking.utils.replay import Replay  # noqa: E402
from agent.executor.low import EnvState  # noqa: E402
from agent.myagent.CSPAgent import CSPAgent  # noqa: E402

from human_models import MODELS, HumanModel  # noqa: E402

QUALITIES = ('good', 'bad', 'random')
SKIP_BUDGETS = (0, 2, 4)
MAX_STEPS = 1000   # ゲーム内100秒。max_num_timesteps と同じ。


MAX_SECONDS_OVERRIDE = None


def make_env(map_name, seed, preset, recipes=None):
    kw = dict(MAP_SETTINGS[map_name])
    if MAX_SECONDS_OVERRIDE:
        kw['max_num_timesteps'] = MAX_SECONDS_OVERRIDE
    kw['order_recipes'] = (list(recipes) if recipes is not None
                           else generate_order_recipes(preset, random.Random(seed)))
    env = OvercookedEnvironment(MapSetting(**kw))
    env.reset()
    return env


def make_ai(skip_budget, partner_is_external=True):
    ai = CSPAgent(10, Replay(), sc_2agent=True, skip_budget=skip_budget)
    # 「相手も CSP として全体最適に動く」と仮定して2体分の計画を立てさせる。
    # 実行に使うのは AI 自身(0番)の行動だけで、1番は人間役の方策で上書きする。
    ai.human_counterpart_mode = False
    # 相手は別方策で動くので、相手の担当タスクが実行される保証はない。
    # 計画は2体分のまま(相手も賢いと信じる)だが、手待ちになったら
    # 相手の担当も引き受ける。そうしないと永久に待ち続けて何も完成しない。
    # follow_plan のときだけ False。相手は計画どおりに動くので、手待ちで
    # 相手の担当を奪いに行くと二人が同じタスクへ殺到して壊れる。
    ai.partner_is_external = partner_is_external
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
    """指示の質に応じて対象タスクを選ぶ。

    どの注文がスープ/ジュースかは、候補一覧そのものから決める。
    _build_order_tasks は呼ぶたびに置き場の割り当てなど内部状態が進むため、
    別々に呼んだ結果を突き合わせると食い違うことがある(実際、1回目には
    cook タスクが出ず、スープの判定が空になっていた)。
    """
    candidates = ai.get_instruction_candidates(dcopy(state))
    if not candidates:
        return None
    soup = {u for _d, p in candidates if p['verb'] == 'cook' for u in p['order_uids']}
    juice = {u for _d, p in candidates if p['verb'] == 'mix' for u in p['order_uids']}
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


def run_trial(map_name, preset, seed, human_model, quality, skip_budget, recipes=None):
    env = make_env(map_name, seed, preset, recipes)
    ai = make_ai(skip_budget, partner_is_external=(human_model != 'follow_plan'))
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
    watcher = PlanWatcher(ai)
    human_name = env.sim_agents[human_idx].name
    human_events = {}
    started = time.time()
    steps = 0
    for steps in range(1, MAX_STEPS + 1):
        move, _ = ai(dcopy(state_for(env, 0)))
        actions = {a.name: (0, 0) for a in env.sim_agents}
        # AI は2体分の行動を返すが、使うのは自分(0番)の分だけ。
        # 1番は「相手はこう動くはず」という見込みにすぎないので捨てる。
        if isinstance(move, dict):
            own = move.get('ai_0')
            if own:
                actions[env.sim_agents[0].name] = own
        elif move:
            actions[env.sim_agents[0].name] = move

        watcher.observe(env)

        if human_model == 'follow_plan':
            # 相手が「AIの計画どおりに動く」条件。AI 自身が相手ぶんとして
            # 計算した行動をそのまま使うのが、その定義そのもの。
            h_action = move.get('ai_1') if isinstance(move, dict) else (0, 0)
            h_tid = human.planned_for_me()
            human.observe_plan_match(h_tid)
        else:
            h_action, h_tid = human.act(state_for(env, human_idx),
                                        env.sim_agents[0].location)
        actions[env.sim_agents[human_idx].name] = h_action or (0, 0)
        watcher.note_human_task(h_tid)
        human.record(state_for(env, human_idx), h_action or (0, 0))

        _s, _r, _done, info = env.step(actions, passed_time=0.1)
        # 「働いているのか、壁に向かって空振りしているのか」は行動だけでは
        # 区別できない(どちらも位置が変わらない)。世界に実際に起きた出来事を
        # 数えるのが確実なので、人間役が起こしたイベントを積算する。
        for ev in (info.get('events') or []):
            if ev.playerA != human_name:
                continue
            name = str(ev.event)
            if name in ('No-op', 'Move'):
                continue
            human_events[name.split('_')[0]] = human_events.get(name.split('_')[0], 0) + 1
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
        **human.time_breakdown(),
        # 実際に世界へ起こした出来事の数。多いほど「手を動かして仕事をした」。
        'human_useful_events': sum(human_events.values()),
        'human_event_kinds': '|'.join(f'{k}:{v}' for k, v in sorted(human_events.items())),
        'replans': watcher.replans,
        'preempted_by_human': len(watcher.preempted),
        'rework': watcher.rework,
        'wall_seconds': round(time.time() - started, 1),
        'status': 'ok',
    })
    return row


class PlanWatcher:
    """AI の計画が実行中にどれだけ揺さぶられたかを数える。

    「次の1タスクが当たったか」だけでは、次は当たっているのにその先で計画が
    破綻している状況を捉えられない。計画そのものへの影響を3つの数で見る。
      replans            : 再計画が走った回数
      preempted_by_human : AI が自分でやるつもりだったタスクを人間に取られた回数
      rework             : AI が向かっていたタスクが計画から消えた回数(手戻り)
    """

    def __init__(self, ai):
        self.ai = ai
        self.replans = 0
        self.preempted = set()
        self.rework = 0
        self._own_target = None
        # 再計画の回数は solve を数えれば分かる。ゲーム側には触らず、
        # このインスタンスのメソッドだけを包む。
        original = ai.solve_csp_scheduling

        def counted(*args, **kwargs):
            self.replans += 1
            return original(*args, **kwargs)

        ai.solve_csp_scheduling = counted

    def _own_plan_ids(self):
        return [t['id'] for t in (self.ai.schedule_per_agent or {}).get(0, [])]

    def observe(self, env):
        plan = self._own_plan_ids()
        idx = self.ai.current_task_idx
        idx = idx.get(0, 0) if isinstance(idx, dict) else 0
        target = plan[idx] if idx < len(plan) else None

        if self._own_target is not None and target != self._own_target:
            # 向かっていたタスクが計画から消えていたら手戻り。
            # (単に次へ進んだだけの場合は、前のタスクは完了しているので数えない)
            if self._own_target not in plan:
                self.rework += 1
        self._own_target = target

    def note_human_task(self, task_id):
        """人間が、AI が自分でやるつもりだったタスクに手を付けたか。"""
        if task_id is not None and task_id in self._own_plan_ids():
            self.preempted.add(task_id)


FIELDS = ['map', 'preset', 'seed', 'human_model', 'quality', 'skip_budget', 'status',
          'orders', 'target', 'target_verb', 'target_obj',
          'served', 'reward', 'orders_left', 'completed', 'timed_out',
          'makespan_actual_s', 'makespan_plan_baseline_s', 'makespan_plan_constrained_s',
          'loss_s', 'loss_status', 'prediction_match', 'prediction_samples',
          'replans', 'preempted_by_human', 'rework',
          'human_frames', 'human_idle_pct', 'human_move_pct', 'human_interact_pct',
          'human_task_switches', 'human_stuck_switches',
          'human_useful_events', 'human_event_kinds',
          'wall_seconds']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', default='experiment')
    ap.add_argument('--preset', default='experiment2')
    ap.add_argument('--seeds', type=int, default=30)
    ap.add_argument('--enumerate-orders', action='store_true',
                    help='注文を乱数で引かず、作りうる組み合わせを全部使う。'
                         '決定的に動く人間役では重複シードが同じ結果になるため、'
                         'こちらのほうが同じ試行数で標本数が多い。')
    ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--shard', default=None,
                    help='"i/n" 形式。試行を n 個に分けて i 番目だけ実行する(並列用)')
    ap.add_argument('--out', default=str(ROOT / 'results' / 'human_model_experiment.csv'))
    args = ap.parse_args()

    print(f'表示: なし(画面なしで実行します)  '
          f'[SDL_VIDEODRIVER={os.environ.get("SDL_VIDEODRIVER")}]')
    print('様子を目で見たいときは tools/watch_human_model.py を使ってください。')

    models = [m.strip() for m in args.models.split(',') if m.strip()]
    if args.enumerate_orders:
        order_sets = enumerate_order_recipes(args.preset)
        print(f'注文の組み合わせを全列挙: {len(order_sets)} 通り')
    else:
        order_sets = None
    num_cases = len(order_sets) if order_sets is not None else args.seeds
    combos = [(s, m, q, d)
              for s in range(num_cases)
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
        recipes = order_sets[seed] if order_sets is not None else None
        rows.append(run_trial(args.map, args.preset, seed, model, quality, budget, recipes))
        if True:   # 1件ごとに出す(黙っていると止まって見えるため)
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
