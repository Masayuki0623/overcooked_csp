"""鍋の入口を塞いだまま「鍋が空くのを待つ」ことがないかの検証。

想定シナリオ:
    スープを2品頼むと、鍋は1つしかないので直列に2回使うことになる。
    1品目を煮ているあいだに、2品目の材料を切り終えた側が鍋の前へ行き、
    空くのを待つ。exp_ring の鍋 (0,7) に接する床は (1,7) の1マスだけ
    なので、そこに立って待つと、煮上がったスープを取り出しに来た相手が
    鍋へ近づけない。

    取り出さなければ鍋は空かない。空かないので待ち続ける。相手は経路が
    無いのでその場で停止する。実測では2人が24秒すくみ、その間にスープが
    焦げて、100秒で1品も出せなかった。

    「空くのを待つ」側が入口をどく、というのが直し方。

実行方法:
    python tests/repro_pot_access_deadlock.py
"""
import contextlib
import io
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'agent'))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from gym_cooking.play_test import MAP_SETTINGS
from gym_cooking.utils.interact import resolve_action
from gym_cooking.utils.replay import Replay
from agent.executor.low import EnvState
from agent.myagent.CSPAgent import CSPAgent

MAP = 'exp_ring'
RECIPES = ['TomatoLettuceSoup', 'OnionLettuceSoup', 'BananaOrangeJuice']
COOK_SECONDS = 25
INSTRUCT_AT = 10.0
DEADLINE = 50.0          # ここまでに鍋から取り出せていれば合格


def set_cook_time(sec):
    import gym_cooking.utils.config as cfg
    cfg.COOKING_TIME_SECONDS = sec
    for mod in ('gym_cooking.recipe_planner.utils',
                'gym_cooking.envs.overcooked_environment',
                'gym_cooking.misc.game.game',
                'agent.myagent.CSPAgent'):
        try:
            m = __import__(mod, fromlist=['_'])
        except Exception:
            continue
        if hasattr(m, 'COOKING_TIME_SECONDS'):
            setattr(m, 'COOKING_TIME_SECONDS', sec)


def state_for(env, idx=0):
    i = env.get_ai_info()
    return EnvState(world=i['world'], agents=i['sim_agents'], agent_idx=idx,
                    order=i['order_scheduler'], event_history=i['event_history'],
                    time=i['current_time'], chg_grid=i['chg_grid'])


def make_agent(counterpart, budget, limit):
    a = CSPAgent(10, Replay(), sc_2agent=True, skip_budget=budget)
    a.human_counterpart_mode = counterpart
    a.own_agent_idx = 0
    a.priority_weights = {}
    a.gui_text_input = ''
    a.gui_constraint_input = ''
    a.active_constraints = []
    a.debug_counter_trace = False
    a.solve_deterministic_limit = limit
    return a


def pot_contents(env):
    names = []
    for objs in env.world.objects.values():
        for o in objs:
            fn = getattr(o, 'full_name', None)
            if fn and getattr(o, 'location', None) == (0, 7):
                names.append(fn)
    return '/'.join(names)


def main():
    set_cook_time(COOK_SECONDS)
    kw = dict(MAP_SETTINGS[MAP])
    kw['order_recipes'] = tuple(RECIPES)
    kw['max_num_orders'] = len(RECIPES)
    env = OvercookedEnvironment(MapSetting(**kw))
    env.reset()
    driver = make_agent(False, None, 0.5)
    names = [a.name for a in env.sim_agents]

    def step():
        with contextlib.redirect_stdout(io.StringIO()):
            move, _ = driver(state_for(env, 0))
        acts = {}
        for n, key in ((names[0], 'ai_0'), (names[1], 'ai_1')):
            agent = next(a for a in env.sim_agents if a.name == n)
            acts[n] = tuple(resolve_action(agent, env.world,
                                           tuple(move.get(key) or (0, 0))))
        env.step(acts, passed_time=0.2)

    while env.current_time < INSTRUCT_AT:
        step()

    # 煮込み中のスープを「出して」と、猶予なしで指示する。
    # 指示の選択画面には出さないようにしたので(いま実行できないため)、
    # ここでは工程の一覧から直接そのタスクを組み立てる。
    # 確かめたいのは、そういう指示を受けたときに実行側が固まらないこと。
    probe = make_agent(True, 0, None)
    with contextlib.redirect_stdout(io.StringIO()):
        st = state_for(env, 0)
        probe(st)
        orders = probe._build_order_tasks(st)
    cands = [(str(t['id'][0]), str(t['id'][1]))
             for o in orders for t in (o.get('tasks') or [])
             if str(t['id'][0]) == 'serve']
    if not cands:
        print('前提が崩れています: スープを出す工程が見つかりません')
        return 1
    verb, obj = cands[0]
    pending = {'id': time.time(), 'task': {'verb': verb, 'obj': obj},
               'target_idx': 0, 'status': 'pending',
               'skip_budget': 0, 'remaining_skip_budget': 0,
               'tasks_before_target_log': [], 'execution_logged': False,
               'deadline_constraint_applied': False}
    driver.skip_budget = 0
    driver._pending_instructions = [pending]
    driver._mark_reschedule_needed('instruction_accepted')
    print(f'指示: {verb} {obj} (猶予0) を {INSTRUCT_AT:.0f}秒時点で出した')

    cooked_seen_at = None
    blocked_frames = 0
    while env.current_time < DEADLINE:
        step()
        pot = pot_contents(env)
        if cooked_seen_at is None and 'Cooked' in pot:
            cooked_seen_at = env.current_time
            print(f'  {env.current_time:5.1f}秒 煮上がった: {pot}')
        if 'Charred' in pot:
            print(f'  {env.current_time:5.1f}秒 焦げた: {pot}')
            break
        if cooked_seen_at is not None:
            waiter = env.sim_agents[1].location
            if waiter == (1, 7):
                blocked_frames += 1
        if not pot:
            break

    pot = pot_contents(env)
    print(f'  最終 {env.current_time:5.1f}秒 鍋={pot or "空"} '
          f'提供{env.order_scheduler.successful_orders}件')
    if cooked_seen_at is not None:
        print(f'  煮上がってから入口(1,7)を塞がれていたフレーム数: {blocked_frames}')

    # ここで見るのは「待つ側が入口を塞がないこと」。取り出しそのものが
    # 進むかは実行側の別の問題(材料をまな板へ置きに行く経路が出ない)が
    # 絡むため、この検証では分けて扱う。
    if cooked_seen_at is None:
        print('前提が崩れています: 検証時間内にスープが煮上がりませんでした')
        return 1
    if 'Charred' in pot:
        print('NG: 取り出せないまま焦げた。入口を塞がれて近づけていない')
        return 1
    if 'Cook' in pot:
        print(f'NG: 煮上がって {DEADLINE - cooked_seen_at:.0f} 秒たっても '
              f'鍋から出せていない (入口が {blocked_frames} フレーム塞がれていた)')
        return 1
    print('OK: 煮上がったスープを鍋から取り出せている '
          f'(入口を通れずにいたのは {blocked_frames} フレームだけ)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
