"""リプレイから、その回に何が起きたかを時刻つきで並べ直す。

リプレイには 5Hz の全行動(env.step)が残っているので、同じ地図・同じ注文で
作り直して行動を流し込めば、当時の持ち物・位置をそのまま再現できる。
「指示した作業に何秒で取りかかったか」を後から確かめるのに使う。

    python tools/replay_trace.py                      # 一番新しい回
    python tools/replay_trace.py <リプレイのパス>
    python tools/replay_trace.py --all                # 指示のあった回を一覧
"""
import argparse
import glob
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in ('.', 'agent', 'testbed-cooking'):
    sys.path.insert(0, str(ROOT / _p))
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting  # noqa: E402
from gym_cooking.play_test import MAP_SETTINGS  # noqa: E402
from gym_cooking.utils.replay import Replay  # noqa: E402
from agent.executor.low import EnvState  # noqa: E402
from agent.myagent.CSPAgent import CSPAgent  # noqa: E402

REPLAY_DIR = ROOT / 'agent' / 'agent' / 'replay'


def load(path):
    rep = Replay.from_file(path)
    his = list(rep)
    info = {'path': Path(path), 'his': his}
    for key in ('web_selection', 'order_result'):
        try:
            info[key] = rep[key]
        except Exception:
            info[key] = None
    info['ai_errors'] = []
    for h in his:
        if h['name'] == 'instruction_accepted':
            info['accepted'] = h['args']
        elif h['name'] == 'instruction_time_loss':
            info['loss'] = h['args']
        elif h['name'] == 'ai_error':
            info['ai_errors'].append(h['args'])
    return info


def rebuild(sel):
    """その回と同じ地図・同じ注文で環境を作り直す。"""
    name = sel['map']
    pool = list(sel.get('order_pool') or [])
    # フルーツを使わない回は、器具を外した版の地図で遊んでいる。
    # チュートリアルの地図は最初から必要な物しか置いていないので、そのまま。
    uses_fruit = any(f in r for f in ('Apple', 'Orange', 'Banana', 'Juice')
                     for r in (pool or sel.get('recipes', [])))
    if not name.startswith('tutorial_') and not uses_fruit:
        name = f"{name}_veg"
    kw = dict(MAP_SETTINGS[name])
    kw['order_recipes'] = tuple(sel['recipes'])
    if sel.get('endless'):
        # 補充はくじ引き。同じ種を入れないと注文の並びが変わり、
        # その回に何が起きたのかを追えない。
        kw.update({
            'endless_orders': True,
            'order_pool': tuple(pool),
            'order_seed': sel.get('order_seed'),
            'max_num_orders': sel.get('orders_active') or 3,
            'max_num_timesteps': sel.get('seconds') or 90,
        })
    env = OvercookedEnvironment(MapSetting(**kw))
    env.reset()
    return env


def make_probe(sel):
    """当時と同じ設定の AI を作る(何を考えていたかを見るため)。"""
    ai = CSPAgent(10, Replay(), sc_2agent=True,
                  skip_budget=sel.get('skip_budget'))
    ai.human_counterpart_mode = True
    ai.own_agent_idx = 0
    ai.priority_weights = {}
    ai.gui_text_input = ''
    ai.gui_constraint_input = ''
    ai.active_constraints = []
    ai.debug_counter_trace = False
    return ai


def state_of(env):
    i = env.get_ai_info()
    return EnvState(world=i['world'], agents=i['sim_agents'], agent_idx=0,
                    order=i['order_scheduler'], event_history=i['event_history'],
                    time=i['current_time'], chg_grid=i['chg_grid'])


def trace(info, upto=None):
    sel = info.get('web_selection')
    if not sel:
        print('  (どの地図で遊んだかが残っていないので再現できません)')
        return
    env = rebuild(sel)
    steps = [h for h in info['his'] if h['name'] == 'env.step']
    ai_name = env.sim_agents[0].name
    human_name = env.sim_agents[1].name if len(env.sim_agents) > 1 else None
    prev = (None, None)
    t = 0.0
    print('   時刻   AIの持ち物              人の持ち物')
    for h in steps[:upto or len(steps)]:
        acts = {a.name: tuple(h['args']['action_dict'].get(a.name) or (0, 0))
                for a in env.sim_agents}
        env.step(acts, passed_time=h['args'].get('passed_time', 0.2))
        t = env.current_time
        ai_hold = getattr(env.sim_agents[0].holding, 'full_name', None)
        hu_hold = (getattr(env.sim_agents[1].holding, 'full_name', None)
                   if human_name else None)
        now = (ai_hold, hu_hold)
        if now != prev:
            print('  %5.1fs %-24s %s' % (t, ai_hold or '-', hu_hold or '-'))
            prev = now
    sched = env.order_scheduler
    print('  -> 提供 %d 件 / 残り %d 件 / %.1f 秒'
          % (sched.successful_orders, len(sched.current_orders), t))


def why(info, upto=None):
    """当時の盤面を復元しながら、AI が何をしようとしていたかを並べる。

    盤面は記録どおりに再現し、その盤面を当時と同じ設定の AI に見せて
    「いま何の作業のつもりか」を言わせる。手が止まっていた理由を探すのに使う。
    """
    sel = info.get('web_selection')
    if not sel:
        print('  (どの地図で遊んだかが残っていないので再現できません)')
        return
    env = rebuild(sel)
    ai = make_probe(sel)
    steps = [h for h in info['his'] if h['name'] == 'env.step']
    prev = None
    print('   時刻   AIのつもり                        持ち物')
    for h in steps[:upto or len(steps)]:
        st = state_of(env)
        try:
            move, reason = ai(st)
        except Exception as e:
            reason = f'(判断できず: {type(e).__name__} {e})'
        sched = (ai.schedule_per_agent or {}).get(0) or []
        idx = (ai.current_task_idx or {}).get(0, 0)             if isinstance(ai.current_task_idx, dict) else 0
        cur = sched[idx]['id'] if idx < len(sched) else None
        hold = getattr(env.sim_agents[0].holding, 'full_name', None)
        line = (str(cur), str(reason)[:32], hold)
        if line != prev:
            print('  %5.1fs %-14s %-32s %s'
                  % (env.current_time, str(cur), str(reason)[:32], hold or '-'))
            prev = line
        acts = {a.name: tuple(h['args']['action_dict'].get(a.name) or (0, 0))
                for a in env.sim_agents}
        env.step(acts, passed_time=h['args'].get('passed_time', 0.2))


def summarize(info):
    sel = info.get('web_selection') or {}
    acc = info.get('accepted') or {}
    loss = info.get('loss') or {}
    res = info.get('order_result') or {}
    print(f"リプレイ: {info['path'].name}")
    print('  地図=%s / 注文=%s / skip_budget=%s'
          % (sel.get('map'), ', '.join(sel.get('recipes', [])), sel.get('skip_budget')))
    print('  指示=%s (%s) / 受け取った時刻=%ss'
          % (acc.get('task', '-'), '/'.join(acc.get('target_dish_kinds', []) or ['-']),
             acc.get('accepted_time_env')))
    print('  L=%s秒 (制約なし f=%s秒 -> 制約あり f\'=%s秒) %s'
          % (loss.get('loss_seconds', '-'), loss.get('baseline_seconds', '-'),
             loss.get('constrained_seconds', '-'), loss.get('status', '')))
    print('  結果: 提供%s件 / 失敗%s件 / スコア%s  (記録 %d ステップ)'
          % (res.get('success', '-'), res.get('fail', '-'), res.get('reward', '-'),
             sum(1 for h in info['his'] if h['name'] == 'env.step')))
    errs = info.get('ai_errors') or []
    if errs:
        kinds = {}
        for e in errs:
            kinds.setdefault(e.get('error', '?'), []).append(e.get('time'))
        print('  AI の判断が %d 回失敗しています:' % len(errs))
        for kind, times in kinds.items():
            print('    %s (最初は %.1f秒)' % (kind, min(t for t in times if t is not None)
                                              if any(t is not None for t in times) else -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path', nargs='?', default=None)
    ap.add_argument('--all', action='store_true', help='指示のあった回を一覧する')
    ap.add_argument('--steps', type=int, default=None, help='何ステップまで再現するか')
    ap.add_argument('--why', action='store_true',
                    help='AI が何をしようとしていたかも出す(遅い)')
    args = ap.parse_args()

    files = sorted(glob.glob(str(REPLAY_DIR / 'web*.rep')), key=os.path.getmtime)
    if not files:
        print('リプレイがありません')
        return 1

    if args.all:
        for f in files:
            info = load(f)
            if not info.get('loss'):
                continue
            loss = info['loss']
            sel = info.get('web_selection') or {}
            print('%-46s skip=%-3s %-12s L=%5s秒  提供%s件'
                  % (Path(f).name, sel.get('skip_budget'), loss.get('task'),
                     loss.get('loss_seconds'),
                     (info.get('order_result') or {}).get('success', '-')))
        return 0

    info = load(args.path or files[-1])
    summarize(info)
    print()
    if args.why:
        why(info, upto=args.steps)
    else:
        trace(info, upto=args.steps)
    return 0


if __name__ == '__main__':
    sys.exit(main())
