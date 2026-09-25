from gym_cooking.misc.game.gameplay import GamePlay
from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting

import argparse


def parse_arguments():
    parser = argparse.ArgumentParser("Overcooked argument parser")
    parser.add_argument(
        "--map", type=str,
        choices=['ring', 'bottleneck', 'partition', 'quick'], default='ring'
    )

    return parser.parse_args()


MAP_SETTINGS = dict(
    ring=dict(level="new1",),
    bottleneck=dict(level="new3",),
    partition=dict(level="new2"),
    quick=dict(level="new5", max_num_orders=4,),
    # ジュース工程の動作確認用。仕切りは無く、AI側/人間側の区別もしていない。
    # 実験用の非対称マップは別途追加する。
    juice=dict(level="juice_test", max_num_orders=1,),
    # 実験用。仕切りで左右に分断し、AI側にスープ用野菜・フルーツ・鍋、
    # 人間側にサラダ用野菜・ミキサー・提供口を置いた非対称マップ。
    experiment=dict(level="experiment", max_num_orders=3,),
    # 仕事量をそろえた実験用の地図(tools/layout_search.py で探した配置)。
    # 器具と材料の置き場所は3つとも同じで、左右のつながり方だけが違う。
    # 左が AI 側(鍋・ミキサー・提供口・玉ねぎ・リンゴ)、右が人間側
    # (レタス・トマト・オレンジ・バナナ)。まな板・皿・コップは両側にある。
    #   exp_partition  : 仕切りで完全に分かれていて、行き来できない
    #   exp_bottleneck : 仕切りの真ん中に1マスだけ通れる穴がある
    #   exp_ring       : 真ん中が島になっていて、ぐるっと回れる
    exp_partition=dict(level="exp_partition", max_num_orders=3,),
    exp_bottleneck=dict(level="exp_bottleneck", max_num_orders=3,),
    exp_ring=dict(level="exp_ring", max_num_orders=3,),
    #   exp_dualpot    : 鍋を2つ、中央の島に置いた版。どちらの鍋にも
    #                    両側から触れるので、待つ側が入口を塞がない。
    #                    煮込みの開始をずらすと「残り時間の違う鍋」が
    #                    同時に存在し、指示の重さに段ができる。
    exp_dualpot=dict(level="exp_dualpot", max_num_orders=3,),
    # 上の3つから、フルーツ・ミキサー・コップを取り除いた版(野菜だけの注文用)
    exp_partition_veg=dict(level="exp_partition_veg", max_num_orders=3,),
    exp_bottleneck_veg=dict(level="exp_bottleneck_veg", max_num_orders=3,),
    exp_ring_veg=dict(level="exp_ring_veg", max_num_orders=3,),
    exp_dualpot_veg=dict(level="exp_dualpot_veg", max_num_orders=3,),
    # チュートリアル用。1人で遊ぶ小さい台所で、その回に使う材料と道具だけを
    # 置いてある。時間制限は付けない(max_num_timesteps=0 で無制限)。
    #   tutorial_salad : 切って皿に乗せて出す
    #   tutorial_soup  : 切って鍋で煮て、皿に取って出す
    #   tutorial_juice : 切ってミキサーで混ぜ、コップに注いで出す
    tutorial_salad=dict(level="tutorial_salad", max_num_orders=2,
                        max_num_timesteps=0, num_agents=1,),
    tutorial_soup=dict(level="tutorial_soup", max_num_orders=2,
                       max_num_timesteps=0, num_agents=1,),
    tutorial_juice=dict(level="tutorial_juice", max_num_orders=2,
                        max_num_timesteps=0, num_agents=1,),
)

if __name__ == '__main__':
    arglist = parse_arguments()

    map_set = MapSetting(**MAP_SETTINGS[arglist.map])
    env = OvercookedEnvironment(map_set)
    env.reset()

    game = GamePlay(env)

    ok = game.on_execute()
    print(ok)
