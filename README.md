# Overcooked CSP エージェント

本プロジェクトは、Overcookedゲームのための様々なエージェントを実装したものです。特に制約充足問題（CSP）と巡回セールスマン問題（TSP）に基づくアプローチに焦点を当てています。

本プロジェクトは、[LLM-Powered Hierarchical Language Agent](https://arxiv.org/abs/2312.15224) プロジェクトのテストベッドを基に構築されています。

## エージェント

本リポジトリには以下のカスタムエージェントが含まれています：

- **CSP Agent** (`CSP`): 行動計画に制約充足問題（CSP）ソルバーを使用します。
- **TSP Solver Agent** (`TSPSolver`): 調理タスクをTSPとしてモデル化し、効率的な経路を見つけます。
- **Greedy Agent** (`Greedy`): 注文を完了するために単純な貪欲法を使用します。
- **Random Agent** (`Random`): ランダムな行動をとります。

また、オリジナルのエージェントもサポートしています：
- **HLA**: Hierarchical Language Agent（LLMのセットアップが必要です）。

## インストール

### 前提条件

- Python 3.10
- Conda (推奨)

### セットアップ

1.  Conda環境を作成します：
    ```bash
    conda create -n overcooked-csp python=3.10
    conda activate overcooked-csp
    ```

2.  環境用の依存関係をインストールします：
    ```bash
    cd testbed-cooking
    pip install -e .
    cd ..
    ```

3.  エージェント用の依存関係をインストールします：
    ```bash
    cd agent
    pip install -e .
    cd ..
    ```

## 使用方法

特定のエージェントでゲームを実行するには、`agent/agent/play_main.py` スクリプトを使用します。

```bash
python agent/agent/play_main.py --map <MAP_NAME> --agent <AGENT_NAME>
```

### 引数

- `--map`: プレイするマップ。
    - 選択肢: `ring`, `bottleneck`, `partition`, `quick`
    - デフォルト: `ring`
- `--agent`: 使用するエージェント。
    - 選択肢: `CSP`, `TSPSolver`, `Greedy`, `Random`, `HLA` など
    - デフォルト: `TSPSolver`

### 実行例

**CSP Agent** で ring マップを実行する場合：
```bash
python agent/agent/play_main.py --map ring --agent CSP
```

**TSP Solver Agent** で partition マップを実行する場合：
```bash
python agent/agent/play_main.py --map partition --agent TSPSolver
```

**Greedy Agent** で quick マップを実行する場合：
```bash
python agent/agent/play_main.py --map quick --agent Greedy
```

## プロジェクト構成

- `agent/`: エージェントの実装が含まれています。
    - `agent/myagent/`: `CSPAgent` と `GreedyAgent` が含まれています。
    - `agent/TSP/`: `TSPSolverAgent` が含まれています。
- `testbed-cooking/`: Overcookedのゲーム環境（`gym-cooking`）が含まれています。
