def print_player_positions(env, idx_human=1):
    """
    EnvStateのenvから人間プレイヤーの位置を取得して表示する
    idx_human: 人間プレイヤーのインデックス（デフォルト1）
    """
    position = env.agents[idx_human].location
    #print(f"Human player position: {position}")
    return position