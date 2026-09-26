"""作れない料理が指示の候補に出ないことの検証。

報告された現象(20260926_162748):
    「今できない指示が表示されていた気がする。
      全部載せサラダを今すぐ作れそうじゃなかったのに指示できていた」

当時の盤面(リプレイから再現):
    台に ChoppedLettuce-ChoppedOnion と ChoppedOnion-ChoppedTomato の
    2つの山だけ。両者とも手ぶら。

    全部載せサラダ(レタス・玉ねぎ・トマト)の材料は3つとも「盤面にある」。
    ところが、この環境では台に置いた材料どうしが1つの山になり、同じ材料が
    2つ入る山は作れない。上の2つは玉ねぎが重なるので合流できず、
    全部載せサラダは作れない。それでも候補に出ていた。

    材料を1つずつ「どこかにあるか」で数えていたのが原因。

ここで確かめること:
    1. 重なる山しか無いときは作れないと判断する(報告の場面そのもの)
    2. 重ならない山どうしは合流できる(トマト単体 + レタス玉ねぎ = 全部載せ)
       一方で、山は分解できない。はみ出す山を割って一部だけ使うことはできない
    3. バラバラに置かれていても作れる
    4. 余計な材料が混ざった山は使わない
    5. 皿に乗っているぶんは数える

実行方法:
    python tests/repro_unassemblable_salad_offered.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'testbed-cooking'))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from agent.agent.myagent.CSPAgent import CSPAgent

results = []


def check(label, ok, detail=""):
    results.append(ok)
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}{(' -> ' + detail) if detail else ''}")


FULL = ['Lettuce', 'Onion', 'Tomato']

# 1. 報告の場面。玉ねぎが重なるので合流できない。
check('重なる山しか無いなら作れない',
      CSPAgent._assemblable(
          ['ChoppedLettuce-ChoppedOnion', 'ChoppedOnion-ChoppedTomato'],
          FULL) is False)

# 2. 重ならない山なら作れる(トマト単体 + レタス玉ねぎ = 全部載せ)
check('トマト単体とレタス玉ねぎの山なら合流できる',
      CSPAgent._assemblable(
          ['ChoppedLettuce-ChoppedOnion', 'ChoppedTomato'], FULL) is True)

# 2'. 山は分解できない。はみ出す山を割って一部だけ使うことはできない。
check('はみ出す山を割って使うことはできない',
      CSPAgent._assemblable(
          ['ChoppedLettuce-ChoppedOnion-ChoppedTomato'],
          ['Lettuce', 'Onion']) is False)

# 3. バラバラでも作れる
check('バラバラに置かれていても作れる',
      CSPAgent._assemblable(
          ['ChoppedLettuce', 'ChoppedOnion', 'ChoppedTomato'], FULL) is True)

# 3'. 1つ足りなければ作れない
check('1つ足りなければ作れない',
      CSPAgent._assemblable(['ChoppedLettuce', 'ChoppedOnion'], FULL) is False)

# 4. 余計な材料が混ざった山は使えない
check('余計な材料が混ざった山は使わない',
      CSPAgent._assemblable(
          ['ChoppedLettuce-ChoppedApple', 'ChoppedOnion', 'ChoppedTomato'],
          FULL) is False)

# 5. 皿に乗っているぶんは数える
check('皿に乗っているぶんは数える',
      CSPAgent._assemblable(
          ['ChoppedLettuce-ChoppedOnion-Plate', 'ChoppedTomato'],
          FULL) is True)

# 6. 切っていない材料は数えない
check('切っていない材料は数えない',
      CSPAgent._assemblable(
          ['ChoppedLettuce', 'ChoppedOnion', 'FreshTomato'], FULL) is False)

# 7. 2品ぶんの材料があるときも、ちょうど1皿ぶん取れれば作れる
check('余分にあっても、1皿ぶん取れれば作れる',
      CSPAgent._assemblable(
          ['ChoppedLettuce-ChoppedOnion', 'ChoppedOnion-ChoppedTomato',
           'ChoppedLettuce', 'ChoppedTomato', 'ChoppedOnion'],
          FULL) is True)

# 8. 2つの材料の料理でも同じように効く
check('2つの材料でも、重なる山だけなら作れない',
      CSPAgent._assemblable(
          ['ChoppedOnion-ChoppedTomato'], ['Lettuce', 'Onion']) is False)

ok = sum(1 for r in results if r)
print()
if ok == len(results):
    print(f'[SUCCESS] {ok}/{len(results)} 件が期待どおり')
else:
    print(f'[FAILURE] {ok}/{len(results)} 件のみ期待どおり')
    sys.exit(1)
