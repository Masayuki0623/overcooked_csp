import itertools

from gym_cooking.utils.core import GRIDSQUARES, PUTTABLE_GRIDSQUARES, FRESH_FOOD, CHOPPED_FOOD, CHOPPING_FOOD, \
    COOKING_FOOD, COOKED_FOOD, ASSEMBLE_CHOPPED_FOOD, ASSEMBLE_CHOPPED_PLATE_FOOD, ASSEMBLE_COOKING_FOOD, \
    ASSEMBLE_COOKING_PLATE_FOOD, ASSEMBLE_COOKED_FOOD, ASSEMBLE_COOKED_PLATE_FOOD, FOOD_TILE, \
    ASSEMBLE_CHARRED_FOOD, ASSEMBLE_CHARRED_PLATE_FOOD,     ASSEMBLE_CHOPPED_FRUIT, ASSEMBLE_MIXED_FOOD, ASSEMBLE_MIXED_CUP_FOOD


class Event:
    def __init__(self, playerA, event, location, time, playerB=None):
        self.playerA = playerA
        self.event = event
        self.location = location
        self.time = time
        self.playerB = playerB


def get_all_events(recipes):
    no_op = ['No-op']

    move = ['Move']

    chop = [f'Chop_{f}' for f in FRESH_FOOD]

    cook = [f'Cook_{f}' for f in ASSEMBLE_CHOPPED_FOOD]

    # ミキサーに入れる。鍋(Cook)と同じく、入れる時点では材料はまだ Chopped。
    mix = [f'Mix_{f}' for f in ASSEMBLE_CHOPPED_FRUIT]

    assemble = [f'Assemble_{f}' for f in ASSEMBLE_CHOPPED_FOOD + ASSEMBLE_CHOPPED_PLATE_FOOD
                + ASSEMBLE_MIXED_CUP_FOOD]

    # 持ち運べるもの。コップと、コップ入りのジュースも含める。
    CARRIABLE = FRESH_FOOD + ASSEMBLE_CHOPPED_FOOD + ASSEMBLE_CHOPPED_PLATE_FOOD \
        + ASSEMBLE_COOKED_PLATE_FOOD + ASSEMBLE_CHARRED_PLATE_FOOD \
        + ASSEMBLE_MIXED_CUP_FOOD + ['FireExtinguisher', 'Plate', 'Cup']

    put = [f'Put_{f}_on_{gs}' for f, gs in
           itertools.product(CARRIABLE, PUTTABLE_GRIDSQUARES)] \
          + ['Put_Plate_on_PlateTile', 'Put_Cup_on_CupTile']

    pickup = [f'Pickup_{f}_from_{gs}' for f, gs in
              itertools.product(CARRIABLE, PUTTABLE_GRIDSQUARES)] \
             + [f'Pickup_{f}_from_{gs}' for f, gs in itertools.product(ASSEMBLE_COOKED_FOOD + ASSEMBLE_CHARRED_FOOD, ["Pot"])] \
             + [f'Pickup_{f}_from_Blender' for f in ASSEMBLE_MIXED_FOOD]
    pickup += [f'Pickup_{f}_from_{gs}' for f, gs in zip(FRESH_FOOD, FOOD_TILE)] \
              + ['Pickup_Plate_from_PlateTile', 'Pickup_Cup_from_CupTile']

    deliver = [f'Deliver_{f}' for f in ASSEMBLE_CHOPPED_PLATE_FOOD + ASSEMBLE_COOKED_PLATE_FOOD
               + ASSEMBLE_MIXED_CUP_FOOD]

    drop = [f'Drop_{f}' for f in FRESH_FOOD + ASSEMBLE_CHOPPED_FOOD + ASSEMBLE_COOKED_FOOD + ASSEMBLE_CHARRED_FOOD
            + ASSEMBLE_MIXED_FOOD]

    putout = ['Putout_Fire']

    return no_op + move + chop + cook + mix + assemble + put + pickup + deliver + drop + putout

# chop
# putout
# cook
# deliver
# drop
# put on gs
# assemble x (on gs)
# pickup x (on gs)
