"""別の注文の鍋が煮えている間に、AI が皿を持ったまま固まらないことの検証。

報告された現象:
    「また最後の方に動かなくなった」
    (20260926_144508 の回。ログには
     「[AI] 3 秒以上動いていません: ターゲットの完成品が入った鍋が
      見つかりません」が残り、そのまま試合が終わっていた)

仕組み:
    AI は「同じ作業を抱えたまま進んでいない」状態を見張って、一定時間で
    その作業を諦め、別の割り当てに移る。ただし鍋が煮えるのを待っている
    だけなら諦めてはいけないので、_waiting_for_pot が True の間は見張りの
    時計をリセットする。

    ところがこの判定が「どれか1つでも煮えている鍋があるか」だった。
    別の注文の鍋が煮えている間は、材料がまだどの鍋にも入っていない配膳
    作業まで「待ちが正しい」と見なされ、時計が毎回リセットされる。
    実行側は「ターゲットの完成品が入った鍋が見つかりません」と言って
    (0,0) を返し続けるので、皿を持ったまま試合が終わる。

ここで確かめること:
    1. 自分の料理が煮えている鍋があるときは「待ち」(諦めない)
    2. 別の料理しか煮えていないときは「待ち」ではない(見張りが働く)
    3. どの鍋も煮えていないときは「待ち」ではない
    4. 煮上がった(Cooked)鍋は「待ち」ではない -- 取りに行く段階
    5. 「必要なものが盤面に無い」という報告のときは、短い時間で諦める

実行方法:
    python tests/repro_wait_for_other_pot.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)

from agent.agent.myagent.CSPAgent import CSPAgent

results = []


def check(label, ok, detail=""):
    results.append(ok)
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}{(' -> ' + detail) if detail else ''}")


class FakeObj:
    def __init__(self, full_name):
        self.full_name = full_name


class FakeEnv:
    """鍋の中身だけを持つ、判定に必要な最小限の盤面。"""

    def __init__(self, pots):
        self.pos_obj = {}
        self._pots = []
        for i, name in enumerate(pots):
            loc = (i, 0)
            self._pots.append(loc)
            self.pos_obj[loc] = FakeObj(name) if name else None
        self.time = 0.0

    def get_pos_by_obj_gs(self, gs=None, obj=None):
        return list(self._pots) if gs == 'Pot' else []


def waiting(pots, dish='lettuce-tomato soup'):
    a = CSPAgent.__new__(CSPAgent)
    a._last_env = FakeEnv(pots)
    return a._waiting_for_pot(('serve', dish, 1))


# 1. 自分の料理が煮えている -> 待ち
check('自分の料理が煮えている鍋があれば、待つ',
      waiting(['CookingLettuce-CookingTomato']) is True)

# 2. 別の料理しか煮えていない -> 待ちではない(これが今回の不具合)
check('別の料理の鍋しか煮えていなければ、待ちにしない',
      waiting(['CookingLettuce-CookingOnion']) is False)

# 2'. 別の鍋が煮えていても、自分の鍋があれば待ち
check('別の鍋と自分の鍋が両方あれば、待つ',
      waiting(['CookingLettuce-CookingOnion',
               'CookingTomato-CookingLettuce']) is True)

# 3. どの鍋も空 -> 待ちではない
check('鍋が空なら、待ちにしない', waiting([None, None]) is False)

# 4. 煮上がっている -> 待ちではない(取りに行く段階)
check('煮上がった鍋は、待ちにしない',
      waiting(['CookedLettuce-CookedTomato']) is False)

# 5. サラダの配膳は鍋を使わない -> 待ちではない
check('サラダの配膳は、鍋の煮込み待ちにしない',
      waiting(['CookingLettuce-CookingTomato'],
              dish='lettuce-tomato salad') is False)

# 6. 「見つかりません」の報告は短い時間で諦める
check('「見つかりません」は短い見張り時間にする',
      CSPAgent._reason_means_missing('ターゲットの完成品が入った鍋が見つかりません') is True)
check('普通の報告は短くしない',
      CSPAgent._reason_means_missing('調理完了待ち') is False)

# 7. 「鍋が空くまで待機中」も短い時間で諦める
#    (20260926_153422 の回。最後の7秒、刻んだ玉ねぎを持ったまま
#     鍋が空くのを待ち続けて試合が終わった。空けるのは相手の仕事なので、
#     待っても自分では何も変えられない)
check('「空くまで待機中」も短い見張り時間にする',
      CSPAgent._reason_means_missing('鍋が空くまで待機中') is True)
check('ミキサー待ちも同じ扱いにする',
      CSPAgent._reason_means_missing('ミキサーが空くまで待機中') is True)
check('煮込み待ちは短くしない',
      CSPAgent._reason_means_missing('調理完了 (Done)') is False)

# 8. 相手が材料を持ってくるのを待っているだけの状態も、短い時間で諦める
#    (20260926_161533 の回。手ぶらのまま 5.2 秒
#     「不足分がそろうのを待機中」で止まっていた)
for reason in ('不足分がそろうのを待機中', 'マージ対象の食材を待機中',
               'マージ対象がそろうのを待機中', '必要な食材 (Chopped) を待機中',
               '指定テーブルが使用中のため待機中', '共有置き場ID未割当のため待機中'):
    check(f'「{reason}」は短い見張り時間にする',
          CSPAgent._reason_means_missing(reason) is True)

# 放っておけば進む待ちは、これまでどおり長く待つ
for reason in ('調理完了待ち', '受け渡し待ち'):
    check(f'「{reason}」は短くしない',
          CSPAgent._reason_means_missing(reason) is False)

ok = sum(1 for r in results if r)
print()
if ok == len(results):
    print(f'[SUCCESS] {ok}/{len(results)} 件が期待どおり')
else:
    print(f'[FAILURE] {ok}/{len(results)} 件のみ期待どおり')
    sys.exit(1)
