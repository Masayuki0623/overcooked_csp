"""実験の順序統制(カウンターバランス)。

8ゲームでも学習効果が出るので、順序を完全にくじ引きにすると、
ある条件だけたまたま後半に偏る参加者が出る。少人数では打ち消されない。
そこで3つを統制する。

    1. 地図をひとまとまりにする(同じ地図を4回続けてから、もう一方へ)
    2. 地図の順序を参加者間で半々にする(ring 先と partition 先)
    3. かたまりの中の割り込み許容数の順序を 4x4 のラテン方格で回す
       (Williams 方格。直前の条件の組も釣り合う)

    G1..G4 : ring -> partition、行 A..D
    G5..G8 : partition -> ring、行 A..D

参加者は連番の順に G1, G2, ... G8, G1, ... と循環して割り当てる。
8人で1周するので、人数は8の倍数にする。

注文の構成(case)は、かたまりの中の「何番目か」だけで決める。
参加者をまたいで同じになるので、条件の比較に注文の違いが混ざらない。
しかも skip_budget の並びはラテン方格で回っているため、4行そろえば
(skip_budget x 注文構成) の組み合わせがちょうど1回ずつ現れる。
"""
import re

# 設計を変えたら上げる。古い割り当てを引きずらないための印。
DESIGN_VERSION = 'cb3'

MAPS = ('exp_ring', 'exp_partition')
# 割り込み許容数は4水準(2026-09-30〜)。指示が「工程の鎖」になり、
# d は鎖の最後が終わるまでに挟んでよい工程の数になった。鎖は長いので
# 2 も意味のある水準になる(以前の単一の指示では 2 はほとんど縛らなかった)。
BUDGETS = (0, 1, 2, 'inf')

# 4x4 のラテン方格(Williams 方格)。どの列(何番目か)にも、どの条件も
# ちょうど1回ずつ入り、「直前がどの条件か」の組も1回ずつになる。
LATIN_SQUARE = {
    'A': (0, 1, 'inf', 2),
    'B': (1, 2, 0, 'inf'),
    'C': (2, 'inf', 1, 0),
    'D': ('inf', 0, 2, 1),
}
ROWS = ('A', 'B', 'C', 'D')

# G1..G6。前半の地図と、使うラテン方格の行。
GROUPS = tuple(
    {'name': f'G{i + 1}',
     'first_map': MAPS[0] if i < len(ROWS) else MAPS[1],
     'row': ROWS[i % len(ROWS)]}
    for i in range(2 * len(ROWS)))

BLOCK = len(BUDGETS)            # 1つの地図で何セッションか
TOTAL = BLOCK * len(MAPS)       # 本番のセッション数(8)


def group_index_of(participant, group=None):
    """その参加者が G いくつか(0 始まり)。

    group を渡せばそれを使う(1..8 でも 'G3' でもよい)。渡さなければ
    参加者IDの末尾の数字から決める。p01 -> G1、p09 -> G1。
    数字が無いIDは、文字列から安定した値を作って割り当てる。連番で
    運用しないときも同じIDなら必ず同じグループになる。
    """
    if group is not None:
        s = str(group).strip().upper()
        if s.startswith('G'):
            s = s[1:]
        try:
            n = int(s)
        except ValueError:
            raise ValueError(f'グループの指定が読めません: {group!r}')
        if not 1 <= n <= len(GROUPS):
            raise ValueError(f'グループは1〜{len(GROUPS)}で指定してください: {group!r}')
        return n - 1
    pid = str(participant or '').strip()
    m = re.search(r'(\d+)\s*$', pid)
    if m:
        return (int(m.group(1)) - 1) % len(GROUPS)
    # 数字が無いID。文字列から決める(くじ引きではないので、毎回同じ)。
    h = 0
    for ch in pid:
        h = (h * 131 + ord(ch)) % 1000003
    return h % len(GROUPS)


def group_of(participant, group=None):
    return GROUPS[group_index_of(participant, group)]


def plan_for(participant, group=None):
    """その参加者の本番8ゲームぶんの条件。

    戻り値は session 1 から順に並んだ一覧。
        {'session', 'map', 'skip_budget', 'block', 'position', 'group', 'row'}
        block    : 前半=1 / 後半=2
        position : そのかたまりの中で何番目か(1..4)。注文構成はこれで決まる
    """
    g = group_of(participant, group)
    order_of_maps = ((g['first_map'], MAPS[1] if g['first_map'] == MAPS[0] else MAPS[0]))
    budgets = LATIN_SQUARE[g['row']]
    out = []
    for block, map_name in enumerate(order_of_maps, 1):
        for position, budget in enumerate(budgets, 1):
            out.append({
                'session': len(out) + 1,
                'map': map_name,
                'skip_budget': budget,
                'block': block,
                'position': position,
                'group': g['name'],
                'row': g['row'],
            })
    return out


def condition_for(participant, session, group=None):
    """そのセッション(1 始まり)の条件。範囲外なら None。"""
    plan = plan_for(participant, group)
    if not 1 <= int(session) <= len(plan):
        return None
    return plan[int(session) - 1]


def case_for(cases, position):
    """注文の構成。かたまりの中の「何番目か」だけで決める。

    参加者をまたいで同じになるので、条件の比較に注文の違いが混ざらない。
    候補が4つより少ない地図では、足りないぶんは先頭から繰り返す。
    """
    cases = list(cases)
    if not cases:
        return None
    return cases[(int(position) - 1) % len(cases)]


# 練習の設定。本番8回の前に1回だけ行う。
#
# 本番と同じ条件を使うと、その条件だけ余分に経験することになる。とはいえ
# 全員に同じものを与えるなら、増えるのは全員ぶんの下駄であって、条件間の
# 差には効かない。そこで skip_budget は 0(指示をいちばん素直に聞く)に
# 固定し、地図は本番の1つ目と同じにする。先に触る地図で練習したほうが
# 配置を覚えられるうえ、地図の順序は参加者間で半々なので偏らない。
#
# 注文の構成だけは本番で使わないものを充てる(position=0)。同じ並びを
# 2回遊ばせない。
PRACTICE_BUDGET = 0
PRACTICE_POSITION = 0


def practice_case(all_count, used):
    """練習で使う注文の構成。本番で使うものは避ける。

    本番の候補は「良い指示が一意に定まる」もので絞ってある。練習は
    分析しないので、その絞りから外れた構成をあてればよい。全部が本番で
    使われている地図では、やむなく本番のものを1つ使い回す。
    """
    used = set(used)
    for i in range(int(all_count)):
        if i not in used:
            return i
    return sorted(used)[-1] if used else 0


def practice_condition(participant, group=None):
    g = group_of(participant, group)
    return {'map': g['first_map'], 'skip_budget': PRACTICE_BUDGET,
            'position': PRACTICE_POSITION, 'group': g['name'], 'row': g['row']}
