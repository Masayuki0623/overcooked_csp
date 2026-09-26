"""選べる作業が1つしかないときは指示画面を出さないことの検証。

背景(利用者の要望):
    指示できる作業が1つだけの画面は「選ぶ」ことにならない。押すしか
    ないうえ、良い指示と悪い指示の差も測れないので、次の機会へ回す。
    これまでは「0個のときだけ」見送っていた。

ここで確かめること:
    1. 見送りの下限は 2 個
    2. 候補が 0 個・1 個なら画面を出さず、見送ったことを返す
    3. 候補が 2 個以上なら出す
    4. Web 側の入口も同じ下限を使っている

実行方法:
    python tests/repro_instruction_needs_choices.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

import inspect
import queue

from agent.agent.gameplay import GamePlay, MIN_INSTRUCTION_CHOICES

results = []


def check(label, ok, detail=""):
    results.append(ok)
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}{(' -> ' + detail) if detail else ''}")


check('見送りの下限は 2 個', MIN_INSTRUCTION_CHOICES == 2,
      str(MIN_INSTRUCTION_CHOICES))


def run_with(n_candidates):
    """候補が n 個のときに、指示画面を出すかどうか。"""
    g = GamePlay.__new__(GamePlay)
    g._instruction_panel_active = False
    g._q_env = queue.Queue()
    g.instruction_chooser = None
    shown = []

    cands = [(f'chop_{i}', {'verb': 'chop', 'obj': f'x{i}'})
             for i in range(n_candidates)]
    g._get_unexecuted_task_candidates = lambda: cands

    def fake_panel(c):
        shown.append(len(c))
        return None            # 選ばずに閉じた扱い
    g._show_instruction_panel = fake_panel
    out = g._request_instruction(trigger='every_n_tasks',
                                 allow_text_fallback=False)
    return out, shown


for n, expect_shown in ((0, False), (1, False), (2, True), (3, True)):
    out, shown = run_with(n)
    check(f'候補 {n} 個 -> 画面を{"出す" if expect_shown else "出さない"}',
          bool(shown) == expect_shown, f'出した={shown}')
    if not expect_shown:
        check(f'候補 {n} 個 -> 見送ったことを返す', out is False, f'戻り値={out}')

# Web 側の入口も同じ下限を使っているか
import server as srv
src = inspect.getsource(srv.WebGamePlay.prepare)
check('Web 側も同じ下限を使っている',
      'MIN_INSTRUCTION_CHOICES' in src)

ok = sum(1 for r in results if r)
print()
if ok == len(results):
    print(f'[SUCCESS] {ok}/{len(results)} 件が期待どおり')
else:
    print(f'[FAILURE] {ok}/{len(results)} 件のみ期待どおり')
    sys.exit(1)
