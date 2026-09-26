# -----------------------------------------------------------
# Parameter configuration
# -----------------------------------------------------------

COOKING_TIME_SECONDS = 15 # time required to cook sth
COOKED_BEFORE_FIRE_TIME_SECONDS = 25 # time before a cooked soup turning into fire
# 煮込みすぎたものが焦げて火事になるかどうか。False だと、煮上がった料理は
# そのまま待っていてくれる(取り出しそこねても作り直しにならない)。
ENABLE_OVERCOOK_FIRE = False
FIRE_PUTOUT_TIME_SECONDS = 5 # time required to put out the fire
FIRE_RECOVER_GAP_TIME_SECONDS = 1 # time gap before the fire starts to grow again
CHOPPING_NUM_STEPS = 8 # steps required to chop some ingredient, e.g. tomato/lettuce
# 1秒あたりに行動できる回数(入力の速さ)。10 だと 1秒に 10 マス進める。
# ゲームの1手は 1/INPUT_HZ 秒で、環境もこの速さで進む。
BASE_INPUT_HZ = 10
INPUT_HZ = 5        # 既定。--input-hz で変える


def set_input_hz(hz):
    """入力の速さを変える。ゲームを組み立てる前に呼ぶこと。"""
    global INPUT_HZ
    hz = int(hz)
    if not (1 <= hz <= BASE_INPUT_HZ):
        raise ValueError('input_hz は 1〜%d' % BASE_INPUT_HZ)
    INPUT_HZ = hz


def seconds_per_step():
    return 1.0 / INPUT_HZ


def chopping_steps():
    """まな板で刻むのに要るインタラクト回数。

    行動の回数は減るが、かかる時間(秒)は変えない。10Hz で 8回(0.8秒)なら、
    5Hz では 4回(0.8秒)。器具の時間(煮る・焦げる)は秒で決まっているので
    そのまま。
    """
    return max(1, round(CHOPPING_NUM_STEPS * INPUT_HZ / BASE_INPUT_HZ))


def blending_steps():
    return max(1, round(BLENDING_NUM_STEPS * INPUT_HZ / BASE_INPUT_HZ))
# ミキサーでフルーツを混ぜるのに必要なインタラクト回数。
# 仕様により、まな板で刻む回数(CHOPPING_NUM_STEPS)と同じにする。
# 定数を分けてあるのは、実験条件として独立に動かせるようにするため。
BLENDING_NUM_STEPS = CHOPPING_NUM_STEPS
MAX_ORDER_LENGTH_SECONDS = 75
ORDER_EXPIRE_PUNISH = 5
