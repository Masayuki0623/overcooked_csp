"""CSP の AI と人間役モデルが遊んでいる様子を、そのまま画面で見る。

実験ハーネス(run_human_model_experiment / run_instruction_wait_experiment)と
まったく同じ試行を走らせ、描画だけを足したもの。ゲーム側は何も変えない。

    python tools/watch_human_model.py                       # 貪欲, d=4, 良い指示
    python tools/watch_human_model.py --model random --d 0
    python tools/watch_human_model.py --case 7 --speed 0.5  # ゆっくり
    python tools/watch_human_model.py --gif out.gif         # 録画して保存

操作: 何もしなくてよい。閉じるかウィンドウの×で終了。
"""
import argparse
import os
import random
import sys
import time
from copy import deepcopy as dcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'agent'))
sys.path.insert(0, str(ROOT / 'testbed-cooking'))
sys.path.insert(0, str(ROOT / 'tools'))

import pygame  # noqa: E402
from gym_cooking.misc.game.game import Game  # noqa: E402
from gym_cooking.utils.order_preset import enumerate_order_recipes  # noqa: E402
from gym_cooking.utils.replay import Replay  # noqa: E402

import run_human_model_experiment as H  # noqa: E402
from human_models import HumanModel  # noqa: E402

# 実験ハーネスは画面なしで回す前提なので、読み込むだけで
# SDL_VIDEODRIVER=dummy を設定する。こちらは見るための道具なので取り消す。
# pygame はウィンドウを作る瞬間にこの値を読むため、ここで消しておけば間に合う。
for _var in ('SDL_VIDEODRIVER', 'SDL_AUDIODRIVER'):
    if os.environ.get(_var) == 'dummy':
        os.environ.pop(_var)

MAX_STEPS = 1000


def task_label(ai, agent_idx, human=None):
    """いまその担当者が取り組んでいる作業の名前。"""
    if human is not None and getattr(human, 'current_id', None) is not None:
        tid = human.current_id
        return f'{tid[0]}:{tid[1]}'
    sched = (ai.schedule_per_agent or {}).get(agent_idx) or []
    idx = ai.current_task_idx
    idx = idx.get(agent_idx, 0) if isinstance(idx, dict) else (idx or 0)
    if 0 <= idx < len(sched):
        tid = sched[idx].get('id')
        if tid:
            return f'{tid[0]}:{tid[1]}'
    return '(手待ち)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', type=int, default=0, help='全列挙した注文構成の番号(0-17)')
    ap.add_argument('--model', default='greedy',
                    choices=['follow_plan', 'greedy', 'random'], help='人間役の方針')
    ap.add_argument('--d', type=int, default=4, help='skip_budget')
    ap.add_argument('--quality', default='good', choices=['good', 'bad', 'random'])
    ap.add_argument('--speed', type=float, default=1.0,
                    help='1.0 で実時間。0.5 で半分の速さ、2.0 で倍速')
    ap.add_argument('--gif', default=None, help='録画して保存する GIF のパス')
    ap.add_argument('--headless', action='store_true',
                    help='ウィンドウを開かない(画面のない環境や録画だけしたいとき)')
    args = ap.parse_args()

    sets = enumerate_order_recipes('experiment2')
    recipes = sets[args.case % len(sets)]

    env = H.make_env('experiment', args.case, 'experiment2', recipes)
    ai = H.make_ai(args.d, partner_is_external=(args.model != 'follow_plan'))
    human_idx = 1
    human = HumanModel(args.model, ai, human_idx, Replay(), seed=args.case * 31 + 7)

    state = H.state_for(env, 0)
    orders = ai._build_order_tasks(dcopy(state))
    rng = random.Random(f'experiment-{args.case}-{args.quality}')
    picked = H.pick_instruction(ai, state, orders, args.quality, rng)
    target_label = '(なし)'
    if picked is not None:
        target_label, payload = picked
        pending = {'id': float(args.case), 'task': payload, 'target_idx': 0,
                   'accepted_env_time': 0.0, 'status': 'pending',
                   'skip_budget': args.d, 'remaining_skip_budget': args.d}
        env._pending_instructions = [dcopy(pending)]
        ai._pending_instructions = [dcopy(pending)]

    print(f'注文: ' + ' | '.join(recipes))
    print(f'人間役: {args.model} / skip_budget: {args.d} / 指示: {target_label} ({args.quality})')
    print('左が AI(0番)、右が人間役(1番)。別ウィンドウが開きます(背面に出ることがあります)。')
    print('1秒ごとに進行状況をここに出します。終わったらウィンドウの×で閉じてください。')

    # play=True にすると Game が実ウィンドウを作る(play=False は隠しSurface)。
    # 中身の描画は同じで、見えるかどうかだけの違い。
    if args.headless:
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
    game = Game(env, play=not args.headless)
    pygame.display.quit()          # 初期化済みのドライバを捨てて開き直す
    try:
        pygame.display.init()
        game.on_init()
    except pygame.error as e:
        print(f'ウィンドウを開けませんでした: {e}')
        print('画面のない環境では --headless --gif out.gif で録画してください。')
        return
    pygame.display.set_caption(
        f'{args.model} / d={args.d} / {target_label}')
    driver = pygame.display.get_driver()
    visible = driver != 'dummy'
    print(f'表示: {"ウィンドウを開きます" if visible else "なし(画面には出ません)"}'
          f'  [描画ドライバ: {driver}]')
    clock = pygame.time.Clock()
    frames = []

    for step in range(1, MAX_STEPS + 1):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        move, _ = ai(dcopy(H.state_for(env, 0)))
        actions = {a.name: (0, 0) for a in env.sim_agents}
        own = move.get('ai_0') if isinstance(move, dict) else move
        if own:
            actions[env.sim_agents[0].name] = own

        if args.model == 'follow_plan':
            h_action = move.get('ai_1') if isinstance(move, dict) else (0, 0)
        else:
            h_action, _tid = human.act(H.state_for(env, human_idx),
                                       env.sim_agents[0].location)
        actions[env.sim_agents[human_idx].name] = h_action or (0, 0)
        human.record(H.state_for(env, human_idx), h_action or (0, 0))

        env.step(actions, passed_time=0.1)
        game.on_render()
        pygame.display.flip()

        # 画面だけだと動いているか分かりにくいので、1秒ごとに状況を出す。
        if step % 10 == 0:
            print(f'  t={env.current_time:5.1f}秒  提供={env.order_scheduler.successful_orders}'
                  f'  AI={task_label(ai, 0)}  人間役={task_label(ai, 1, human)}', flush=True)

        if args.gif:
            buf = pygame.surfarray.array3d(game.screen)
            frames.append(buf.swapaxes(0, 1).copy())

        # 環境は 10Hz。--speed 1.0 でそのままの速さ。
        clock.tick(max(1.0, 10 * args.speed))

        if not env.order_scheduler.current_orders:
            break

    sched = env.order_scheduler
    print(f'\n提供 {sched.successful_orders} / 残り {len(sched.current_orders)} '
          f'/ 経過 {env.current_time:.1f}秒')
    print(f'人間役: 停止 {human.time_breakdown().get("human_idle_pct")}% '
          f'/ タスク切替 {human.task_switches} 回')

    if args.gif and frames:
        try:
            import imageio.v2 as imageio
            imageio.mimsave(args.gif, frames, fps=10)
            print(f'録画を保存しました -> {args.gif}')
        except ImportError:
            print('GIF 保存には imageio が要ります: pip install imageio')

    if args.headless:
        pygame.quit()
        return

    # 最後の盤面を見られるように、閉じられるまで待つ。
    print('(ウィンドウの×を押すか、Ctrl+C で終了します)')
    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
            clock.tick(10)
    except KeyboardInterrupt:
        pass
    pygame.quit()


if __name__ == '__main__':
    main()
