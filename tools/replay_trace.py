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
    for h in his:
        if h['name'] == 'instruction_accepted':
            info['accepted'] = h['args']
        elif h['name'] == 'instruction_time_loss':
            info['loss'] = h['args']
    return info


def rebuild(sel):
    """その回と同じ地図・同じ注文で環境を作り直す。"""
    name = sel['map']
    # フルーツを使わない回は、器具を外した版の地図で遊んでいる
    if not any(f in r for r in sel.get('recipes', [])
               for f in ('Apple', 'Orange', 'Banana', 'Juice')):
        name = f"{name}_veg"
    kw = dict(MAP_SETTINGS[name])
    kw['order_recipes'] = tuple(sel['recipes'])
    env = OvercookedEnvironment(MapSetting(**kw))
    env.reset()
    return env


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path', nargs='?', default=None)
    ap.add_argument('--all', action='store_true', help='指示のあった回を一覧する')
    ap.add_argument('--steps', type=int, default=None, help='何ステップまで再現するか')
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
    trace(info, upto=args.steps)
    return 0


if __name__ == '__main__':
    sys.exit(main())
