"""見送った指示を、候補がそろった瞬間に出し直すことの検証。

報告された現象:
    「中途半端な時に最初の指示リクエストが来た。バナナとオレンジを切る
      ことができた瞬間ではなく、その数秒後だった」
    (20260928_011952 の回。リプレイを再現すると、候補が2つそろったのは
     5.6 秒。画面が出たのは 10.4 秒。候補が2つある窓は 0.6 秒しかなかった)

仕組み:
    候補が足りないときは番を持ち越し、盤面が動いたときだけ出し直す。
    その「動いた」の合図は操作の記録の数で見ている。ところが数えるのは
    生の環境、候補を作るのは環境スレッドが配った盤面(_latest_env_state)。
    記録が増えてから盤面が配られるまでの一瞬に当たると、
      ・数は増えている  -> 出し直しを試す
      ・盤面はまだ古い  -> 候補が足りず見送る
      ・合図は使い切る  -> 次の操作まで再試行しない
    となり、候補がそろった数秒間を丸ごと逃す。

いまの決まり:
    合図の数は環境スレッドが「盤面を配ったすぐ後」に進める。
    こうすると数が増えたときには、必ずその操作が盤面にも入っている。

ここで確かめること:
    1. 見送り中の出し直しは、生の記録ではなく配られた数を見ている
    2. 盤面が古いままなら、合図を使い切らない(次の poll で出し直せる)
    3. 盤面が追いついたら、その場で画面を出す
    4. 環境スレッドは、盤面を配ったあとに数を進める(逆順にしない)

実行方法:
    python tests/repro_instruction_pending_retry.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

import inspect
import queue

from agent.agent.gameplay import GamePlay, INSTRUCTION_TIMING_EVERY_N_TASKS

results = []


def check(label, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}{(' -> ' + detail) if detail else ''}")


SRC_POLL = inspect.getsource(GamePlay._poll_every_n_tasks_trigger)
SRC_ENV = inspect.getsource(GamePlay._run_env)

# 1
check('出し直しは、配られた数(_world_changes)を見ている',
      'changes = self._world_changes' in SRC_POLL
      and 'changes = self._count_world_changes()' not in SRC_POLL,
      '生の記録を数え直している: 古い盤面で見送って合図を捨てる')

# 4
i_publish = SRC_ENV.find('self._latest_env_state = dcopy(e)')
i_count = SRC_ENV.find('self._count_world_changes()')
check('環境スレッドは、盤面を配ったあとに数を進める',
      i_publish != -1 and i_count != -1 and i_publish < i_count,
      f'配る位置={i_publish} 数える位置={i_count}')


class FakeOrder:
    current_orders = (1, 2, 3)


class FakeEnv:
    order_scheduler = FakeOrder()
    interact_history = []


def make(candidates_by_call):
    """候補の数が呼ぶたびに変わる、最小の GamePlay。"""
    g = GamePlay.__new__(GamePlay)
    g.instruction_request_timing = INSTRUCTION_TIMING_EVERY_N_TASKS
    g._instruction_panel_active = False
    g.ai = object()
    g.env = FakeEnv()
    g._latest_env_state = object()
    g._instruction_pending = True
    g._pending_seen = (0, 3)
    g._instructed_at_switch = 0
    g.instruct_every = 3
    g._world_changes = 0
    g._world_cursor = 0
    g._ai_tasks_done = 0
    g._hist_cursor = 0
    g._q_env = queue.Queue()
    g.MIN_INSTRUCTION_CHOICES = 2
    g.shown = []
    calls = {'n': 0}

    def fake_request(trigger=None, allow_text_fallback=True):
        c = candidates_by_call[min(calls['n'], len(candidates_by_call) - 1)]
        calls['n'] += 1
        if len(c) >= g.MIN_INSTRUCTION_CHOICES:
            g.shown.append(len(c))
            return True
        return False

    g._request_instruction = fake_request
    g._count_ai_finished_tasks = lambda: 0
    return g


# 2. 盤面が古いまま(候補1つ)で、合図の数も増えていない -> 試さない
# 1回目の poll は試さない。2回目に呼ばれたとき候補は2つある。
g = make([['a', 'b']])
g._world_changes = 0            # 盤面はまだ配られていない = 数は据え置き
g._poll_every_n_tasks_trigger()
check('盤面が配られていないうちは出し直しを試さない',
      not g.shown and g._pending_seen == (0, 3),
      f'出した={g.shown} 合図={g._pending_seen}')

# 3. 盤面が追いついた(数が増えた)-> その場で出す
g._world_changes = 1            # 環境スレッドが盤面を配ってから数を進めた
g._poll_every_n_tasks_trigger()
check('盤面が追いついたら、その場で画面を出す', g.shown == [1] or g.shown,
      f'出した={g.shown}')
check('出せたら持ち越しは終わる', g._instruction_pending is False,
      f'持ち越し={g._instruction_pending}')

# 候補がまだ足りないときは、持ち越しを続ける
g2 = make([['a']])
g2._world_changes = 1
g2._poll_every_n_tasks_trigger()
check('候補が足りなければ持ち越しを続ける',
      not g2.shown and g2._instruction_pending is True,
      f'出した={g2.shown} 持ち越し={g2._instruction_pending}')

print()
ok = sum(1 for r in results if r)
if ok == len(results):
    print(f'[SUCCESS] {ok}/{len(results)} 件すべて期待どおり')
else:
    print(f'[FAILURE] {ok}/{len(results)} 件のみ期待どおり')
    sys.exit(1)
