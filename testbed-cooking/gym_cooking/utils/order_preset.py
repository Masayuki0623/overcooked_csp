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


def _is_salad_recipe(recipe):
    """サラダかスープかを判定する。

    レシピ定義上サラダは材料が Chopped(state_index=2)、スープは
    Cooked(state_index=4) で登録されるため、完成品名に 'cooked' を
    含むかどうかで確実に区別できる。
    """
    return 'cooked' not in recipe.full_name.lower()


def recipe_pool(category, min_ingredients=MIN_INGREDIENTS):
    """指定した系統で、材料が min_ingredients 個以上のレシピ名を返す。

    レシピ一覧をそのまま走査するので、recipe.py にレシピを足せば
    プリセット側を触らなくても候補に入る。
    """
    want_salad = (category == SALAD)
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
        if _is_salad_recipe(recipe) == want_salad:
            pool.append(name)
    return pool


def generate_order_recipes(preset_name, rng=None):
    """プリセット名から、実際に出す注文のレシピ名リストを生成する。"""
    if not is_order_preset(preset_name):
        raise ValueError(
            f"Unknown order preset: {preset_name} (available: {', '.join(preset_names())})")

    rng = rng or random
    recipes = []
    for category, count in ORDER_PRESETS[preset_name]:
        pool = recipe_pool(category)
        if not pool:
            raise ValueError(
                f"No recipe with {MIN_INGREDIENTS}+ ingredients for category: {category}")
        # 材料の組み合わせは注文ごとに独立に選ぶ(同じ組み合わせが並ぶこともある)
        recipes.extend(rng.choice(pool) for _ in range(count))
    return recipes
