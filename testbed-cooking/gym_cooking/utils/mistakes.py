"""プレイ中のミスを、盤面で起きた出来事から拾う。

ミスは3種類。どれも「その瞬間に残っている注文」と突き合わせて決める
(あとで注文が片づいて余った物は、作った時点ではミスではない)。

    misserve    注文外の提供
                注文に無い物を提供口へ出した。
    unusable    使えない組み合わせ
                残っている注文のどれにも合わない組み合わせを作った。
                台・皿の上で重ねたときは「どの注文の材料にも収まらない」こと、
                鍋・ミキサーに入れたときは「どの注文ともぴったり一致しない」こと
                (入れたあとは足せないので、一部だけ合っていても使えない)。
    extra_chop  余分に切った
                その材料を、残っている注文に要る数より多く切った。
                数えるのは切り始めた時点。使えない組み合わせに入ってしまった
                ぶんは「ある」うちに数えない(切り直しはミスではない)。

記録は env.mistake_log に足す。誰がやったか(by)も残すので、人のぶんだけ
数えるかどうかは読む側で決める。

    {'type': 'unusable', 'time': 12.4, 'by': 'agent-2',
     'detail': 'ChoppedOnion-ChoppedTomato'}
"""

FOODS = ('Lettuce', 'Onion', 'Tomato', 'Apple', 'Orange', 'Banana')
TYPES = ('misserve', 'unusable', 'extra_chop')
# 入れたら足せない器具。出来事の名前 -> その器具で作る料理の種類
_STATION_KIND = {'Cook': 'soup', 'Mix': 'juice'}


def foods_in(name):
    """名前に含まれる材料の集合('ChoppedOnion-ChoppedTomato-Plate' -> {Onion, Tomato})。"""
    name = str(name)
    return frozenset(f for f in FOODS if f in name)


def pending_orders(env):
    """いま残っている注文を (種類, 材料の集合) の一覧にする。"""
    out = []
    sched = getattr(env, 'order_scheduler', None)
    for o in (getattr(sched, 'current_orders', None) or []):
        name = str(getattr(o[0], 'full_name', '') or '')
        if not name:
            continue
        kind = 'soup' if 'Cooked' in name else 'juice' if 'Mixed' in name else 'salad'
        out.append((kind, foods_in(name)))
    return out


def combo_usable(foods, orders, station_kind=None):
    """その材料の組み合わせが、残っている注文に使えるか。

    station_kind を渡すと(鍋なら 'soup')、その種類の注文とぴったり一致する
    ときだけ使えると見る。渡さなければ、どれかの注文の材料に収まればよい。
    """
    if not foods:
        return True
    if station_kind:
        return any(k == station_kind and f == foods for k, f in orders)
    return any(foods <= f for _k, f in orders)


def _objects(env):
    from gym_cooking.utils.core import Object
    return [o for o in env.world.get_object_list() if isinstance(o, Object)]


def _processed_foods(obj):
    """その物に入っている、生でない材料の名前(切りかけ・切った・煮ている等)。"""
    out = []
    for c in getattr(obj, 'contents', []) or []:
        name = getattr(c, 'name', None)
        if name in FOODS and not str(getattr(c, 'full_name', '')).startswith('Fresh'):
            out.append(name)
    return out


def _supply(env, ing, orders):
    """その材料のうち、切り始めてあって、まだ注文に使えるものの数。"""
    n = 0
    for obj in _objects(env):
        foods = _processed_foods(obj)
        if ing not in foods:
            continue
        if not combo_usable(frozenset(foods), orders):
            continue                      # 使えない山に入ったぶんは数えない
        n += foods.count(ing)
    return n


def _add(env, kind, by, detail):
    env.mistake_log.append({
        'type': kind,
        'time': round(float(getattr(env, 'current_time', 0.0) or 0.0), 1),
        'by': by,
        'detail': str(detail),
    })


def note_misserve(env, by, dish):
    _add(env, 'misserve', by, dish)


def _count_stacks(env, foods):
    return sum(1 for o in _objects(env) if frozenset(_processed_foods(o)) == foods)


def note_events(env, events):
    """この1手で起きた出来事から、使えない組み合わせと余分に切ったを拾う。

    注文の消し込みより前に呼ぶこと(同じ手で出された料理の注文が、まだ
    残っているうちに突き合わせる)。
    """
    _scan(env, events)
    # 使えない山が盤面から無くなっていたら(提供口へ出した等)、覚えている
    # 数も減らす。減らさないと、同じ組み合わせをもう一度作っても数えない。
    for foods, seen in list(env._unusable_seen.items()):
        now = _count_stacks(env, foods)
        if now < seen:
            env._unusable_seen[foods] = now


def _scan(env, events):
    orders = None
    for ev in events:
        name = str(getattr(ev, 'event', '') or '')
        kind, _, body = name.partition('_')
        if kind not in ('Assemble', 'Cook', 'Mix', 'Chop') or not body:
            continue
        if orders is None:
            orders = pending_orders(env)
        by = getattr(ev, 'playerA', None)

        if kind in _STATION_KIND:
            foods = foods_in(body)
            if not combo_usable(foods, orders, _STATION_KIND[kind]):
                _add(env, 'unusable', by, body)
            continue

        if kind == 'Assemble':
            # 切った材料を2つ以上重ねたときだけ見る(皿に1つ乗せただけ、
            # 鍋から皿に取った、は組み合わせを作っていない)。
            chopped = [p for p in body.split('-') if p.startswith('Chopped')]
            if len(chopped) < 2:
                continue
            foods = foods_in('-'.join(chopped))
            if combo_usable(foods, orders):
                continue
            # 同じ山に皿を足しただけのときも Assemble が出る。その山の数が
            # 増えたときだけ数える。
            now = _count_stacks(env, foods)
            seen = env._unusable_seen.get(foods, 0)
            if now > seen:
                env._unusable_seen[foods] = now
                _add(env, 'unusable', by, '-'.join(sorted(chopped)))
            continue

        # Chop: 持っている物をまな板に置いた瞬間と、切り終えた瞬間の両方で
        # 出る。同じ材料を2回数えないよう、物に印を付ける。
        ing = body[len('Fresh'):] if body.startswith('Fresh') else body
        if ing not in FOODS:
            continue
        try:
            obj = env.world.get_object_at(ev.location, None, find_held_objects=False)
        except Exception:
            obj = None
        if obj is None or getattr(obj, '_chop_noted', False):
            continue
        obj._chop_noted = True
        need = sum(1 for _k, f in orders if ing in f)
        if _supply(env, ing, orders) > need:
            _add(env, 'extra_chop', by, ing)


def summarize(log, by=None):
    """種類ごとの回数と、内訳の文字列('12.4:unusable:...' を | で区切る)。

    by を渡すと、その人のぶんだけ数える。
    """
    rows = [m for m in (log or []) if by is None or m.get('by') == by]
    counts = {t: sum(1 for m in rows if m.get('type') == t) for t in TYPES}
    detail = '|'.join('%s:%s:%s' % (m.get('time'), m.get('type'), m.get('detail'))
                      for m in rows)
    return {'total': len(rows), 'counts': counts, 'detail': detail}
