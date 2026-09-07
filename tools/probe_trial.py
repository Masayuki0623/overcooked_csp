"""1つの試行を、実験ハーネスと同じ条件で走らせて中身を見る。

止まった原因を追うための道具。実行側が何をしようとして、何回それを
繰り返したかを出す。指示ありの条件も再現する。

    python tools/probe_trial.py 1 bad 2 greedy
"""
import collections
import os
import random
import sys
from copy import deepcopy as dcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in ('.', 'agent', 'testbed-cooking', 'tools'):
    sys.path.insert(0, str(ROOT / _p))

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

from gym_cooking.utils.order_preset import enumerate_order_recipes  # noqa: E402
from gym_cooking.utils.replay import Replay  # noqa: E402
import run_human_model_experiment as H  # noqa: E402
from human_models import HumanModel  # noqa: E402
from agent.myagent.TaskAgent import TaskAgent  # noqa: E402


def main():
    case = int(sys.argv[1])
    quality = sys.argv[2] if len(sys.argv) > 2 else 'good'
    budget = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    model = sys.argv[4] if len(sys.argv) > 4 else 'greedy'

    sets = enumerate_order_recipes('experiment2')
    H.MAX_SECONDS_OVERRIDE = 100.0
    env = H.make_env('experiment', case, 'experiment2', sets[case])
    ai = H.make_ai(budget, partner_is_external=(model != 'follow_plan'))
    human = HumanModel(model, ai, 1, Replay(), seed=case * 31 + 7)

    state = H.state_for(env, 0)
    orders = ai._build_order_tasks(dcopy(state))
    picked = H.pick_instruction(ai, state, orders, quality,
                                random.Random('experiment-%d-%s' % (case, quality)))
    print('注文:', ' | '.join(sets[case]), '/ 指示:', picked[0] if picked else None)
    if picked is not None:
        pend = {'id': float(case), 'task': picked[1], 'target_idx': 0,
                'accepted_env_time': 0.0, 'status': 'pending',
                'skip_budget': budget, 'remaining_skip_budget': budget}
        shared = [pend]
        env._pending_instructions = shared
        ai._pending_instructions = shared

    seen = collections.Counter()
    first = {}
    orig = TaskAgent.__call__

    def spy(self, e, **kw):
        action, reason = orig(self, e, **kw)
        key = (getattr(e, 'agent_idx', None), self.task_name, reason)
        seen[key] += 1
        first.setdefault(key, round(env.current_time, 1))
        return action, reason

    TaskAgent.__call__ = spy
    for _step in range(1, 1001):
        move, _ = ai(dcopy(H.state_for(env, 0)))
        acts = {a.name: (0, 0) for a in env.sim_agents}
        own = move.get('ai_0') if isinstance(move, dict) else move
        if own:
            acts[env.sim_agents[0].name] = own
        if model == 'follow_plan':
            ha = move.get('ai_1') if isinstance(move, dict) else (0, 0)
        else:
            ha, _ = human.act(H.state_for(env, 1), env.sim_agents[0].location)
        acts[env.sim_agents[1].name] = ha or (0, 0)
        human.record(H.state_for(env, 1), ha or (0, 0))
        env.step(acts, passed_time=0.1)
        if not env.order_scheduler.current_orders:
            break

    sched = env.order_scheduler
    print('提供 %d / 残り %d / %.1f秒'
          % (sched.successful_orders, len(sched.current_orders), env.current_time))
    for (idx, name, reason), n in seen.most_common(8):
        print('  A%s %5d回 (初回t=%5.1f) %s :: %s'
              % (idx, n, first[(idx, name, reason)], name, reason))
    st = H.state_for(env, 0)
    print('  col6=', {p[1]: getattr(o, 'full_name', '')
                      for p, o in st.pos_obj.items() if p[0] == 6 and o})


if __name__ == '__main__':
    main()
