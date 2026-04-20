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
- `--sc_2agent`: 2つのAIエージェントで協調してタスクを実行するマルチエージェントモードを有効にします（CSPエージェントのみ対応）。

### 実行例

**CSP Agent** で ring マップを実行する場合：
```bash
python agent/agent/play_main.py --map ring --agent CSP
```

### AIによる優先度自動生成 (CSP Agent)

CSPエージェントでは、ゲーム開始前に表示されるGUIからAI（OpenAI API）を使用して、日本語の指示からタスクの優先度を自動生成することができます。この機能を使用するには、OpenAIのAPIキーの設定が必要です。

#### APIキーの設定方法

環境変数 `OPENAI_API_KEY` にAPIキーを設定してください。

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY = "your-api-key-here"
```

**Windows (コマンドプロンプト):**
```cmd
set OPENAI_API_KEY=your-api-key-here
```

**Linux / macOS:**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

設定後、`play_main.py` を実行し、GUIの「AI Assistant」セクションから指示を入力して「Generate Weights with AI」ボタンを押してください。

**TSP Solver Agent** で partition マップを実行する場合：
```bash
python agent/agent/play_main.py --map partition --agent TSPSolver
```

**Greedy Agent** で quick マップを実行する場合：
```bash
python agent/agent/play_main.py --map quick --agent Greedy
```

**CSP Agent** で2エージェントを用いた協調動作（マルチエージェント機能）を実行する場合：
```bash
python agent/agent/play_main.py --map ring --agent CSP --sc_2agent
```

### 2エージェント行動計画スケジューリング (`--sc_2agent`) の実装状況

`--sc_2agent` オプションを付与することで、人間とAIではなく、両方の操作キャラクターをAI（CSPエージェント）に任せ、2エージェント向けの行動計画スケジューリングを行います。現在の実装の仕組みは以下の通りです：

- **ゲームプレイ制御 (`gameplay.py`)**: 
  - コマンドライン引数から `sc_2agent=True` を受け取ると、人間のキーボード入力を無効化し（人間エージェントのインデックスを `None` に設定）、両方のエージェントに対するAIのアクション（辞書型）を受け付けて環境を進行させます。
- **タスク割り当てと協調 (`CSPAgent.py`)**:
  - **初期化**: 2つのエージェントそれぞれのインデックスを持ち、各エージェントに対応する `TaskAgent` を生成してタスク進捗をそれぞれ管理します。
  - **ターン制による排他制御**: 現状では2エージェントが「完全同期的」に動作するのではなく、衝突やデッドロックを防ぐために「交互に動く」ターン制のアプローチが実装されています（`self.turn` 変数によって管理）。
  - **衝突回避**: 動こうとしているエージェントは、片方のエージェントが現在いる位置を完全に「壁（動的障害物 / `dynamic_obstacles`）」として扱い、経路探索（A*）を行います。
  - **スケジューリング**: OR-ToolsのCP-SATでタスク全体（切る、煮る、配膳など）の最適な順序や移動コストを計算・スケジュール化し、それを元にエージェント間でタスクの担当と実行順序が割り当てられます。

## プロジェクト構成

- `agent/`: エージェントの実装が含まれています。
    - `agent/myagent/`: `CSPAgent` と `GreedyAgent` が含まれています。
    - `agent/TSP/`: `TSPSolverAgent` が含まれています。
- `testbed-cooking/`: Overcookedのゲーム環境（`gym-cooking`）が含まれています。
