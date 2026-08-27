"""未完走の試行を全条件で洗い出し、止まった原因ごとに分類する。

個別のケースを1つずつ潰すのではなく、「原因が何種類あるのか」を先に
把握するための道具。試行ごとに、両者が動けなかった場面の内訳を記録する。

    python tools/diagnose_stalls.py --shard 0/8 --out results/stalls_0.csv
"""
import argparse
import collections
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'agent'))
sys.path.insert(0, str(ROOT / 'testbed-cooking'))
sys.path.insert(0, str(ROOT / 'tools'))

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

from gym_cooking.utils.order_preset import enumerate_order_recipes  # noqa: E402
import run_instruction_wait_experiment as W  # noqa: E402
import human_models  # noqa: E402
from agent.myagent.TaskAgent import TaskAgent  # noqa: E402

QUALITIES = ('good', 'bad', 'random')
SKIP_BUDGETS = (0, 2, 4)

FIELDS = ['seed', 'orders', 'quality', 'skip_budget', 'human_model',
          'served', 'completed', 'makespan_actual_s', 'unfinished_dish',
          'human_stall', 'human_stall_count', 'ai_stall', 'ai_stall_count',
          'progress_stall_events', 'progress_stall_detail']

# 人間役の実体を掴むために生成を見張る
_human_ref = []
_orig_init = human_models.HumanModel.__init__


def _init(self, *a, **k):
    _orig_init(self, *a, **k)
    _human_ref.append(self)


human_models.HumanModel.__init__ = _init

_progress_events = []
_human_stalls = collections.Counter()
_ai_stalls = collections.Counter()
_orig_call = TaskAgent.__call__


def _spy(self, env_, **kw):
    action, reason = _orig_call(self, env_, **kw)
    if action == (0, 0) or action is None:
        label = f'{self.task_name} :: {reason}'
        if _human_ref and self is _human_ref[-1].ta:
            _human_stalls[label] += 1
        elif getattr(env_, 'agent_idx', None) == 0:
            _ai_stalls[label] += 1
    return action, reason


TaskAgent.__call__ = _spy


def unfinished_dish(row):
    """作り切れなかった料理の系統。"""
    if row.get('completed'):
        return ''
    kinds = []
    for name in (row.get('orders') or '').split('|'):
        for suffix in ('Salad', 'Soup', 'Juice'):
            if name.endswith(suffix):
                kinds.append(suffix)
    served = int(row.get('served') or 0)
    # 提供順は分からないので、残った数だけを種類とともに記す
    return f'{len(kinds) - served}品未完({"/".join(sorted(set(kinds)))})'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--human-model', default='greedy')
    ap.add_argument('--shard', default=None, help='"i/n" 形式')
    ap.add_argument('--max-seconds', type=float, default=100.0)
    ap.add_argument('--out', default=str(ROOT / 'results' / 'stalls.csv'))
    args = ap.parse_args()

    W.MAX_STEPS = int(args.max_seconds / W.STEP_SECONDS)
    W.H.MAX_SECONDS_OVERRIDE = args.max_seconds

    sets = enumerate_order_recipes('experiment2')
    combos = [(c, q, d) for c in range(len(sets)) for q in QUALITIES for d in SKIP_BUDGETS]
    if args.shard:
        i, n = (int(x) for x in args.shard.split('/'))
        combos = [c for k, c in enumerate(combos) if k % n == i]

    from agent.myagent.CSPAgent import CSPAgent
    _orig_watch = CSPAgent._watch_progress

    def _watch(self, agent_idx, action, reason):
        before = dict(self.blocked_tasks[agent_idx])
        _orig_watch(self, agent_idx, action, reason)
        for tid in self.blocked_tasks[agent_idx]:
            if tid not in before:
                _progress_events.append(f'A{agent_idx}:{tid[0]}:{reason}')

    CSPAgent._watch_progress = _watch

    rows = []
    for k, (case, quality, budget) in enumerate(combos, 1):
        _progress_events.clear()
        _human_stalls.clear()
        _ai_stalls.clear()
        _human_ref.clear()
        r = W.run_trial(case, sets[case], quality, budget, args.human_model)
        h = _human_stalls.most_common(1)
        a = _ai_stalls.most_common(1)
        rows.append({
            'seed': case, 'orders': r.get('orders'), 'quality': quality,
            'skip_budget': budget, 'human_model': args.human_model,
            'served': r.get('served'), 'completed': r.get('completed'),
            'makespan_actual_s': r.get('makespan_actual_s'),
            'unfinished_dish': unfinished_dish(r),
            'human_stall': h[0][0] if h else '', 'human_stall_count': h[0][1] if h else 0,
            'progress_stall_events': len(_progress_events),
            'progress_stall_detail': '|'.join(_progress_events[:6]),
            'ai_stall': a[0][0] if a else '', 'ai_stall_count': a[0][1] if a else 0,
        })
        print(f'  {k}/{len(combos)} 件', flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f'{len(rows)} 試行 -> {out}')


if __name__ == '__main__':
    main()
