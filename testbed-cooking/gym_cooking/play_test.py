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
)

if __name__ == '__main__':
    arglist = parse_arguments()

    map_set = MapSetting(**MAP_SETTINGS[arglist.map])
    env = OvercookedEnvironment(map_set)
    env.reset()

    game = GamePlay(env)

    ok = game.on_execute()
    print(ok)
