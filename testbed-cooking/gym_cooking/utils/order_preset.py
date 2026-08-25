"""実験用の注文セット(プリセット)を生成する。

`--orders experiment1` のように、注文ファイルの代わりにプリセット名を指定できる。
プリセットが固定するのは「どの系統の料理を何品出すか」という枠組みだけで、
各料理が必要とする材料の組み合わせはレシピ一覧からランダムに選ぶ。

生成結果は具体的なレシピ名のリストとして呼び出し側に返し、そのまま
MapSetting.order_recipes に載せる。こうすることで、リプレイには
「プリセット名」ではなく「実際に出た注文」が記録され、再実行しても
別の注文に化けることがない。
"""
import random

import gym_cooking.recipe_planner.recipe as RECIPE

# 実験で使うのは「材料を2つ以上必要とするレシピ」のみ。
# 単品(SimpleTomato / TomatoSoup など)は工程が短く、
# 分担や段取りの差が出ないため除外する。
MIN_INGREDIENTS = 2

SALAD = 'salad'
SOUP = 'soup'

# プリセット名 -> ((系統, 品数), ...)
ORDER_PRESETS = {
    # サラダ2品 + スープ1品。材料の組み合わせはそれぞれランダム。
    'experiment1': ((SALAD, 2), (SOUP, 1)),
}


def is_order_preset(name):
    return str(name) in ORDER_PRESETS


def preset_names():
    return sorted(ORDER_PRESETS)


JUICE = 'juice'


def _category_of(recipe):
    """レシピの系統(サラダ/スープ/ジュース)を判定する。

    材料の登録状態で確実に区別できる。サラダは Chopped、スープは Cooked、
    ジュースは Mixed。名前に 'cooked' を含むかだけで見ると、ジュース
    (MixedApple-MixedOrange)がサラダに分類されてしまうため、
    'mixed' を先に判定する。
    """
    name = recipe.full_name.lower()
    if 'mixed' in name:
        return JUICE
    if 'cooked' in name:
        return SOUP
    return SALAD


def recipe_pool(category, min_ingredients=MIN_INGREDIENTS):
    """指定した系統で、材料が min_ingredients 個以上のレシピ名を返す。

    レシピ一覧をそのまま走査するので、recipe.py にレシピを足せば
    プリセット側を触らなくても候補に入る。
    """
    pool = []
    for name in sorted(dir(RECIPE)):
        cls = getattr(RECIPE, name)
        if not isinstance(cls, type) or not issubclass(cls, RECIPE.Recipe):
            continue
        if cls is RECIPE.Recipe:
            continue
        recipe = cls()
        if len(recipe.contents) < min_ingredients:
            continue
        if _category_of(recipe) == category:
            pool.append(name)
    return pool


def _ingredient_names(recipe_name):
    """レシピ名から材料名の集合を返す(例: TomatoLettuceSalad -> {Tomato, Lettuce})。"""
    recipe = getattr(RECIPE, recipe_name)()
    return {c.name for c in recipe.contents}


def has_exclusive_salad_ingredient(recipe_names):
    """サラダにしか使わない具材が1つ以上あるか。

    スープが具材を全種類使ってしまうと、どの下ごしらえもスープに寄与する
    ことになり、「サラダ専用の下ごしらえ」が存在しなくなる。指示の質を
    good/bad で分ける実験では、この状態だと bad が定義できない。
    """
    salad, soup = set(), set()
    for name in recipe_names:
        cat = _category_of(getattr(RECIPE, name)())
        target = soup if cat == SOUP else salad if cat == SALAD else None
        if target is not None:
            target |= _ingredient_names(name)
    return bool(salad - soup)


def generate_order_recipes(preset_name, rng=None, require_exclusive_salad_ingredient=True,
                           max_attempts=200):
    """プリセット名から、実際に出す注文のレシピ名リストを生成する。

    require_exclusive_salad_ingredient=True のとき、サラダにしか使わない
    具材が必ず1つ以上ある組み合わせになるまで引き直す。実験で「悪い指示
    (サラダ専用の下ごしらえを優先させる)」が必ず成立するようにするため。
    """
    if not is_order_preset(preset_name):
        raise ValueError(
            f"Unknown order preset: {preset_name} (available: {', '.join(preset_names())})")

    rng = rng or random

    def draw():
        recipes = []
        for category, count in ORDER_PRESETS[preset_name]:
            pool = recipe_pool(category)
            if not pool:
                raise ValueError(
                    f"No recipe with {MIN_INGREDIENTS}+ ingredients for category: {category}")
            # 材料の組み合わせは注文ごとに独立に選ぶ(同じ組み合わせが並ぶこともある)
            recipes.extend(rng.choice(pool) for _ in range(count))
        return recipes

    for _ in range(max_attempts):
        recipes = draw()
        if not require_exclusive_salad_ingredient or has_exclusive_salad_ingredient(recipes):
            return recipes
    raise ValueError(
        f"サラダ専用の具材を持つ組み合わせを {max_attempts} 回引いても作れませんでした: "
        f"{preset_name}")
