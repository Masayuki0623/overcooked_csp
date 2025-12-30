# HumanPredictor

`HumanPredictor` クラスは、Overcooked 環境における人間の行動（タスク）を予測し、各タスクを実行するためのコスト（距離）を計算するためのモジュールです。

## 概要

このクラスは、現在の環境状態 (`env`) とエージェントの状態 (`agent`) を分析し、人間プレイヤーが次に行う可能性が高い行動を予測します。予測は主に以下の2つのステップで行われます。

1.  **状態に基づく即時判定**: エージェントが特定のオブジェクトを持っている場合や、特定の場所にいる場合に、直感的な次の行動を決定します（例：生の食材を持っているなら「切る」、皿を持っているなら「配膳する」）。
2.  **コスト最小化に基づく予測**: 現在の注文 (`orders`) から導き出される全ての可能なタスク（Chop, Cook, Serve）について、実行にかかるコスト（移動距離）を計算し、最もコストが低いタスクを予測行動とします。

## 主なメソッド

### `predict(env, agent_idx)`

人間エージェントの次の行動を予測します。

*   **引数**:
    *   `env`: 現在の環境オブジェクト (`OvercookedEnvironment`)
    *   `agent_idx`: 対象とするエージェントのインデックス
*   **戻り値**:
    *   `best_task_str`: 最も可能性が高いと予測されたタスク名（文字列）
    *   `min_cost`: そのタスクの推定コスト
    *   `all_costs`: 検討された全てのタスクとそのコストのリスト

### `_calc_chop_cost_new(env, agent, ingredient)`

食材を「切る」タスクのコストを計算します。

*   **ロジック**:
    1.  対象の生の食材 (`Fresh{Ingredient}`) がある場所を探します（Tile または Counter）。
    2.  利用可能なまな板 (`Cutboard`) を探します。
    3.  `エージェント -> 食材 -> まな板` の移動距離を計算します。

### `_calc_cook_cost_new(env, agent, soup_name, order_idx)`

スープを「調理する（鍋に入れる/調理開始する）」タスクのコストを計算します。

*   **ロジック**:
    1.  対象のスープを作るのに適した鍋を探します。
    2.  鍋に必要な食材が全て揃っているか確認します。
        *   **不足している食材がある場合**: その食材を取りに行き、鍋に入れるまでの距離を計算します。
        *   **全て揃っている場合**: 鍋の場所へ移動して調理を開始する距離を計算します。
    3.  **食材の探索 (`_get_ingredient_sources`)**:
        *   単体の食材だけでなく、**結合されたオブジェクト（例: `ChoppedLettuce-ChoppedOnion`）の中身も再帰的にチェック**し、必要な `Chopped` 食材が含まれているかを確認します。
        *   `Plate` オブジェクトが含まれている場合（完成品など）は、食材ソースとしては除外します。
    4.  必要な食材が環境内に存在しない、またはエージェントが持っていない場合は、このタスクは実行不可能とみなされます。

### `_calc_serve_cost_new(env, agent, soup_name, order_idx)`

完成したスープを「配膳する」タスクのコストを計算します。

*   **ロジック**:
    1.  調理済み (`is_cooked`) で、かつ中身が注文と一致する鍋を探します。
    2.  エージェントが皿を持っているか確認します。
        *   **皿を持っている場合**: 鍋への距離を計算します。
        *   **皿を持っていない場合**: `エージェント -> 皿 -> 鍋` の移動距離を計算します。

## 依存関係

*   `agent.TSP.TSPSolverAgent`: マップ上の2点間の最短距離を計算するために使用されます。
*   `gym_cooking.utils.core`: 環境内のオブジェクト（Food, Plate, GridSquareなど）の定義を参照します。

## 特記事項

*   **結合オブジェクトの扱い**: カウンター上で複数の食材が重なっている場合（Merged Object）、その中身 (`contents`) を走査して、特定の食材（例: `ChoppedOnion`）が含まれているかを正しく判定します。
*   **Plateのエラー回避**: `contents` 内に `Plate` オブジェクトが含まれている場合、`get_state()` メソッドを持たないため、属性チェックを行ってエラーを回避しています。

---

# CSP Solver Implementation

`agent/agent/myagent/csp/` および `CSPAgent.py` に実装されている制約充足問題（CSP）ソルバーについての説明です。Google OR-Tools を使用して実装されています。

## ファイル構成

*   `agent/agent/myagent/csp/model.py`: `CSPModel` クラス。OR-Tools の `cp_model` のラッパーで、変数や制約の追加を簡易化します。
*   `agent/agent/myagent/csp/solver.py`: `solve` 関数。`CSPModel` を受け取り、`cp_model.CpSolver` を実行して結果を返します。
*   `agent/agent/myagent/CSPAgent.py`: `CSPAgent` クラス。環境情報からタスクを生成し、CSPモデルを構築して解くロジックを含みます。

## 現在の実装: 0-1 選択問題 (Knapsack with Precedence)

`CSPAgent.solve_csp_knapsack_with_ortools` メソッドにて、タスク選択の最適化問題が実装されています。

### 問題設定
限られた時間（予算）内で、タスク間の依存関係を守りつつ、重要度の高いタスクをできるだけ多く実行する計画を立てます。

### 変数
*   `x_{verb}_{obj}_{order}_{idx}` (Bool): 各タスクを実行するかどうか（1: 実行, 0: しない）。

### 制約
1.  **予算制約 (Budget Constraint)**:
    *   選択されたタスクの所要時間の合計が、設定された予算 (`budget_frames`) 以下であること。
    *   $\sum (duration_t \times x_t) \le Budget$

2.  **前後関係制約 (Precedence Constraint)**:
    *   **Cookタスクの条件**: Cookタスクを実行するには、その料理に必要な全てのChopタスクが実行されている必要がある。
        *   $x_{cook} \le x_{chop\_i}$ (全ての $i$ について)
    *   **Serveタスクの条件**: Serveタスクを実行するには、対応するCookタスクが実行されている必要がある。
        *   $x_{serve} \le x_{cook}$

### 目的関数
*   **重み付き利益の最大化**:
    *   各タスクの「重み (`weight`) $\times$ 所要時間 (`duration`)」の合計を最大化します。
    *   Maximize $\sum (weight_t \times duration_t \times x_t)$
    *   重み設定（デフォルト）: Serve(5) > Cook(2) > Chop(1)

## 今後の拡張予定
現在は「どのタスクを行うか」の選択のみを行っていますが、今後は「いつ、誰が、どの順番で行うか」というスケジューリング問題（RCPSPなど）へと拡張していく基盤となります。
