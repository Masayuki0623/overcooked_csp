"""指示した作業を、実際にやったのが誰かを記録することの検証。

なぜ要るか:
    指示は AI 宛てだが、待ちきれずに参加者が自分でやってしまうことがある。
    そのとき「指示した作業は済んだ」ように見えるので、記録だけでは
    「AI が指示に従った回」と区別できない。追従性を測る実験なので、
    ここを取り違えると結果の意味が変わる。

ここで確かめること:
    1. 参加者がその作業をやり遂げたら、時刻が残る
    2. 触っただけ(拾う・置く)は「やった」にしない
    3. AI がやったときは、人の欄は空のまま
    4. どちらが先だったかが残る
    5. どちらも手を付けなければ「両方なし」
    6. 記録の列に入っている(定量ファイルと既存ファイルの両方)

実行方法:
    python tests/repro_who_did_instruction.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'agent'))

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from gym_cooking.envs.overcooked_environment import (
    OvercookedEnvironment, MapSetting, _INSTRUCTION_EVENT_BY_VERB)
from gym_cooking.utils.core import Object, Onion, FoodState
from gym_cooking.utils.interact import resolve_action

import server as srv

results = []


def check(label, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}{(' -> ' + detail) if detail else ''}")


# 6. 列があること
for name in ('人がやったか', '人がやった時刻_秒', '人が触った時刻_秒', '先にやったのは'):
    check(f'定量ファイルに「{name}」がある', name in srv.QUANT_FIELDS)
check('既存ファイルにも同じ列がある',
      all(n in srv.INSTRUCTION_FIELDS
          for n in ('人がやったか', '人がやった時刻_秒', '先にやったのは')))

# 動詞と出来事の対応
check('切る作業は Chop_ で「やった」と数える',
      _INSTRUCTION_EVENT_BY_VERB.get('chop') == 'Chop_')
check('提供は Deliver_ で数える',
      _INSTRUCTION_EVENT_BY_VERB.get('serve_salad') == 'Deliver_')

# 置いてある物を切り終えたときにも記録が出ること。出ないと、まな板の前で
# 押し続けて切り上げた分は誰がやったのか分からない。
import inspect
from gym_cooking.utils import interact as _it
check('置いてある物を切り終えたときにも記録が出る',
      "event=f'Chop_{obj.full_name}'" in inspect.getsource(_it.interact)
      and 'not obj.needs_chopped()' in inspect.getsource(_it.interact))


def make(verb='chop', obj='onion'):
    env = OvercookedEnvironment(MapSetting(level='exp_ring',
                                           order_recipes=('OnionLettuceSalad',),
                                           max_num_timesteps=200))
    env.reset()
    pend = {'id': 1.0, 'target_idx': 0, 'status': 'pending',
            'task': ('x', {'verb': verb, 'obj': obj})}
    env._pending_instructions = [pend]
    return env, pend


def step(env, human_action=(0, 0), ai_action=(0, 0)):
    acts = {env.sim_agents[0].name: ai_action,
            env.sim_agents[1].name: human_action}
    for a in env.sim_agents:
        acts[a.name] = resolve_action(a, env.world, acts[a.name])
    env.step(acts, passed_time=0.2)


# 1 + 2. 人が玉ねぎを切る
env, pend = make()
human = env.sim_agents[1]
board = min([tuple(o.location) for o in env.world.objects.get('Cutboard', [])],
            key=lambda b: abs(b[0] - human.location[0]) + abs(b[1] - human.location[1]))
# 切りかけの玉ねぎをまな板に置き、人に最後の一押しをさせる
o = Onion()
o.set_state(FoodState.FRESH)
ob = Object(location=board, contents=[o])
env.world.insert(ob)
env.world.get_gridsquare_at(board).acquire(ob)
# 人をまな板の隣へ運び、向かせて手を出す
# まな板の隣に立たせ、そちらへ向かって押し続ける
face = None
for cand in ((board[0], board[1] + 1), (board[0], board[1] - 1),
             (board[0] + 1, board[1]), (board[0] - 1, board[1])):
    gs = env.world.get_gridsquare_at(cand)
    if gs is not None and type(gs).__name__ == 'Floor':
        human.location = cand
        face = (board[0] - cand[0], board[1] - cand[1])
        break
for _ in range(20):
    step(env, human_action=face)
    if pend.get('human_done_env_time') is not None:
        break
check('人が切り終えたら、その時刻が残る',
      pend.get('human_done_env_time') is not None,
      f"人がやった={pend.get('human_done_env_time')} 触った={pend.get('human_touch_env_time')}")
check('人がやったので「人がやったか」は 1 になる',
      int(pend.get('human_done_env_time') is not None) == 1)

# 3 + 5. 誰も触らなければ空のまま
env2, pend2 = make()
for _ in range(10):
    step(env2)
check('誰も触らなければ、人の欄は空のまま',
      pend2.get('human_done_env_time') is None
      and pend2.get('human_touch_env_time') is None,
      str(pend2))

# 4. 先にやったのは、の決め方
def first_of(started, human_done):
    if started is not None and human_done is not None:
        return 'AI' if float(started) <= float(human_done) else '人'
    if started is not None:
        return 'AI'
    if human_done is not None:
        return '人'
    return '両方なし'


check('AI が先なら AI', first_of(3.0, 8.0) == 'AI')
check('人が先なら 人', first_of(9.0, 4.0) == '人')
check('AI だけなら AI', first_of(3.0, None) == 'AI')
check('人だけなら 人', first_of(None, 4.0) == '人')
check('どちらも手を付けなければ 両方なし', first_of(None, None) == '両方なし')

print()
ok = sum(1 for r in results if r)
if ok == len(results):
    print(f'[SUCCESS] {ok}/{len(results)} 件すべて期待どおり')
else:
    print(f'[FAILURE] {ok}/{len(results)} 件のみ期待どおり')
    sys.exit(1)
