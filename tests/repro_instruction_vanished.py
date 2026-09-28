"""注文が消えて指示の作業が無くなった回を、done と区別することの検証。

なぜ要るか:
    注文には期限がある(サラダ30秒 / スープ・ジュース60秒)。エンドレスでは
    期限切れの注文が消えて新しい注文に入れ替わる。指示した作業がその注文の
    ものだった場合、作業は計画から消える。

    これまでは、計画から消えた指示をまとめて done にしていた。誰も手を
    付けていないのに「やり終えた」と記録されるので、
      ・AI が指示どおりやり終えた回
      ・注文が消えて、うやむやに終わった回
    が区別できなかった。追従性を測る実験なので、ここを混ぜると結果の
    意味が変わる。

いまの決まり:
    計画から消えたとき、AI も人も一度も手を付けていなければ vanished。
    どちらかが手を付けていれば done。

ここで確かめること:
    1. 誰も手を付けていなければ vanished
    2. AI が取りかかっていれば done
    3. 人がやってしまっていれば done
    4. vanished も「済んだ指示」として扱う(以後は縛らない)
    5. L はこの影響を受けない(指示を受け取った瞬間に1度だけ測るため)
    6. 記録の説明に vanished が載っている

実行方法:
    python tests/repro_instruction_vanished.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'agent'))

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

import inspect

from agent.myagent.CSPAgent import CSPAgent
import server as srv

results = []


def check(label, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}{(' -> ' + detail) if detail else ''}")


gone = CSPAgent._instruction_outcome_when_gone

# 1
check('誰も手を付けていなければ vanished',
      gone({}) == 'vanished', gone({}))
# 2
check('AI が取りかかっていれば done',
      gone({'execution_logged': True}) == 'done',
      gone({'execution_logged': True}))
check('AI の着手時刻があれば done',
      gone({'started_env_time': 3.4}) == 'done',
      gone({'started_env_time': 3.4}))
# 3
check('人がやってしまっていれば done',
      gone({'human_done_env_time': 5.0}) == 'done',
      gone({'human_done_env_time': 5.0}))
# 人が触っただけでは done にしない
check('人が触っただけなら vanished のまま',
      gone({'human_touch_env_time': 5.0}) == 'vanished',
      gone({'human_touch_env_time': 5.0}))

# 4
SRC = inspect.getsource(CSPAgent)
check('vanished も「済んだ指示」として扱う',
      "{'done', 'canceled', 'vanished'}" in SRC,
      'done/canceled だけを飛ばしている')
check('計画から消えたときに、この判定を使っている',
      SRC.count('_instruction_outcome_when_gone(pending)') >= 2,
      f"{SRC.count('_instruction_outcome_when_gone(pending)')} 箇所")
check('無条件の done が残っていない',
      "pending['status'] = 'done'\n                            continue" not in SRC)

# 5. L は受け取った瞬間に1度だけ測る
from agent.gameplay import GamePlay
GSRC = inspect.getsource(GamePlay)
check('L は指示を受け取った直後に1度だけ測る',
      'estimate_instruction_time_loss(env_state, pending_entry)' in GSRC
      and GSRC.count('estimate_instruction_time_loss') == 2,
      f"{GSRC.count('estimate_instruction_time_loss')} 箇所")

# 6
for name, notes in (('定量', srv.QUANT_NOTES), ('既存', srv.INSTRUCTION_NOTES)):
    check(f'{name}ファイルの説明に vanished がある',
          'vanished' in notes.get('指示の結末', ''),
          notes.get('指示の結末', '')[:40])

print()
ok = sum(1 for r in results if r)
if ok == len(results):
    print(f'[SUCCESS] {ok}/{len(results)} 件すべて期待どおり')
else:
    print(f'[FAILURE] {ok}/{len(results)} 件のみ期待どおり')
    sys.exit(1)
