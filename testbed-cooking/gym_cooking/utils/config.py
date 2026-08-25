# -----------------------------------------------------------
# Parameter configuration
# -----------------------------------------------------------

COOKING_TIME_SECONDS = 15 # time required to cook sth
COOKED_BEFORE_FIRE_TIME_SECONDS = 25 # time before a cooked soup turning into fire
FIRE_PUTOUT_TIME_SECONDS = 5 # time required to put out the fire
FIRE_RECOVER_GAP_TIME_SECONDS = 1 # time gap before the fire starts to grow again
CHOPPING_NUM_STEPS = 8 # steps required to chop some ingredient, e.g. tomato/lettuce
# ミキサーでフルーツを混ぜるのに必要なインタラクト回数。
# 仕様により、まな板で刻む回数(CHOPPING_NUM_STEPS)と同じにする。
# 定数を分けてあるのは、実験条件として独立に動かせるようにするため。
BLENDING_NUM_STEPS = CHOPPING_NUM_STEPS
MAX_ORDER_LENGTH_SECONDS = 75
ORDER_EXPIRE_PUNISH = 5
