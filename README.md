# Overcooked CSP エージェント

本プロジェクトは、Overcookedゲームのための様々なエージェントを実装したものです。特に制約充足問題（CSP）と巡回セールスマン問題（TSP）に基づくアプローチに焦点を当てています。

本プロジェクトは、[LLM-Powered Hierarchical Language Agent](https://arxiv.org/abs/2312.15224) プロジェクトのテストベッドを基に構築されています。

## エージェント

本リポジトリには以下のカスタムエージェントが含まれています：

- **CSP Agent** (`CSP`): 行動計画に制約充足問題（CSP）ソルバーを使用します。人間との協調プレイと、AI2体による協調プレイの両方に対応しています。
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

`agent/agent/play_main.py` スクリプトでゲームを実行します。

```bash
python agent/agent/play_main.py --agent0 human --agent1 CSP --sc_2agent
```

`python -m agent.play_main ...` でも同じように起動できます。

### 引数

| 引数 | 既定値 | 説明 |
| --- | --- | --- |
| `--map` | `ring` | プレイするマップ。`ring` / `bottleneck` / `partition` / `quick` |
| `--agent0` | （`--agent` にフォールバック） | プレイヤー0のエージェント。`human` または各エージェント名 |
| `--agent1` | `human` | プレイヤー1のエージェント。`human` または各エージェント名 |
| `--agent` | `TSPSolver` | 旧形式の指定。`--agent0` を省略したときのフォールバック |
| `--sc_2agent` | 無効 | CSPエージェントの2エージェント向けスケジューリングを有効にする |
| `--orders` | `sample.txt` | 注文の指定。プリセット名または注文ファイル |
| `--order-seed` | なし | `--orders` にプリセットを指定したときの抽選シード |
| `--instruction_request_timing` | `free` | 人間がAIに指示を出せるタイミング |
| `--deadline` | なし | 指示したタスクを実行するまでに、AIが割り込んでよい他タスクの上限個数 |
| `--debug` | 無効 | デバッグ表示と詳細ログ（`agent/agent/logs/` に出力）を有効にする |
| `--no_reschedule` | 無効 | CSPエージェントの再スケジューリングを無効にする |
| `--task` | なし | `--agent Task` のときに実行するタスク名（例: `chop_tomato`） |

`--deadline` は名前こそ秒数のようですが、現在の実装では**タスク数**として扱われます（`--deadline 2` なら、指示したタスクの前に実行してよい他タスクは2件まで）。

### プレイ構成の例

人間（プレイヤー0）とCSPエージェント（プレイヤー1）の協調プレイ：
```bash
python agent/agent/play_main.py --map ring --agent0 human --agent1 CSP --sc_2agent
```

AI2体（両方CSP）による協調プレイ：
```bash
python agent/agent/play_main.py --map ring --agent0 CSP --agent1 CSP --sc_2agent
```

**TSP Solver Agent** で partition マップを実行する場合：
```bash
python agent/agent/play_main.py --map partition --agent TSPSolver
```

**Greedy Agent** で quick マップを実行する場合：
```bash
python agent/agent/play_main.py --map quick --agent Greedy
```

## 注文の指定 (`--orders`)

`--orders` には2種類の指定方法があります。既定は `sample.txt` です。

### 1. プリセット名 — 実験用のランダム生成

```bash
python agent/agent/play_main.py --agent0 human --agent1 CSP --sc_2agent --orders experiment1
```

生成ルールは `testbed-cooking/gym_cooking/utils/order_preset.py` に定義しています。

- **`experiment1`**: サラダ2品 + スープ1品。いずれも**材料を2つ以上必要とするレシピ**のみを候補とし（単品は工程が短く差が出ないため除外）、どの材料の組み合わせになるかを注文ごとに独立してランダムに選びます。

候補はレシピ一覧を走査して求めているため、`recipe_planner/recipe.py` にレシピを追加すればプリセット側を変更しなくても候補に入ります。

起動時に、実際に選ばれた注文が出力されます：

```
[Orders] preset 'experiment1' (seed=1): OnionLettuceSalad, FullSalad, OnionTomatoSoup
```

`--order-seed` を付けると抽選が再現可能になります（実験を反復するとき用）。省略した場合は毎回ランダムです。

```bash
python agent/agent/play_main.py --agent0 human --agent1 CSP --sc_2agent --orders experiment1 --order-seed 42
```

ランダム化はゲーム開始前に解決され、確定したレシピ名がリプレイに記録されます。そのため、リプレイを再生しても注文が別のものに変わることはありません。

### 2. 注文ファイル — 直接指定

`testbed-cooking/gym_cooking/utils/order/` 配下のファイル名か、任意のパスを指定します。拡張子は省略できます。

```bash
python agent/agent/play_main.py --agent0 human --agent1 CSP --sc_2agent --orders salad_test.txt
```

ファイル形式は、1行目が注文数、以降がレシピ名です。

```
3
TomatoSoup
OnionTomatoSoup
FullSoup
```

同梱のファイル：

- `sample.txt` — スープ3品（既定）
- `salad_test.txt` — サラダ2品 + スープ1品

### 指定できるレシピ

| 種類 | 1材料 | 2材料 | 3材料 |
| --- | --- | --- | --- |
| サラダ（刻んで皿に盛る） | `SimpleTomato` / `SimpleLettuce` / `SimpleOnion` | `TomatoLettuceSalad` / `OnionTomatoSalad` / `OnionLettuceSalad` | `FullSalad` |
| スープ（刻んで鍋で調理する） | `TomatoSoup` / `LettuceSoup` / `OnionSoup` | `TomatoLettuceSoup` / `OnionTomatoSoup` / `OnionLettuceSoup` | `FullSoup` |

サラダとスープでは工程が異なります。サラダは**鍋を使わず**、刻んだ材料を皿に盛って提供します（`chop` → `serve_salad`）。スープは刻んだ材料を鍋で調理してから皿に移して提供します（`chop` → `cook` → `serve`）。

## AIへの指示 (`--instruction_request_timing`)

ゲーム中に指示画面を開くと、AIに次に実行してほしいタスクをカードから選べます。`--instruction_request_timing` は、その指示画面を開けるタイミングを制御します。実験条件として指示のタイミングを揃えるための引数です。

| 値 | 挙動 |
| --- | --- |
| `free`（既定） | 従来どおり、**Spaceキー**を押すといつでも指示できる |
| `enable_cook` | 調理タスクに**今すぐ着手できる状態になった瞬間**に、自動で指示画面が開く。タイミングを固定するため、Spaceキーによる任意の呼び出しは無効 |
| `no_instruction` | 指示を出せない |

```bash
python agent/agent/play_main.py --agent0 human --agent1 CSP --sc_2agent --orders experiment1 --instruction_request_timing enable_cook
```

### `enable_cook` の判定条件

「今すぐ着手できる」は、AI自身が調理を開始できるか判断するのと同じ条件です。

- 材料が刻み終わって世界に存在する
- 実際に投入できる鍋がある（空の鍋か、既にそのレシピが入っている鍋）

どちらか欠けた状態で選ばせても、AIはその場で待つことしかできないため、両方を条件にしています。判定はCSPエージェントが通常の判断サイクルで書き出し（`CSPAgent.ready_cook_actions`）、ゲーム側はそれを読むだけにしています。

発火するのは「着手できなかった状態から着手できる状態に変わった」瞬間だけです。着手可能なまま留まっている間に繰り返し開くことはありません。

なお、材料の有無は世界全体で判定するため、サラダ用に刻んだ材料が置かれた時点でスープの調理が着手可能と判定されることがあります。

指示のトリガ種別（`space` / `enable_cook`）はログとリプレイに記録されるため、解析時に人間が自発的に出した指示と自動的に出た指示を区別できます。

## 2エージェント行動計画スケジューリング (`--sc_2agent`)

`--sc_2agent` を付けると、CSPエージェントが2人分のタスク割り当てと実行順序をまとめてスケジューリングします。相手が人間かAIかは `--agent0` / `--agent1` で決まります。

- **人間 + CSP**（`--agent0 human --agent1 CSP --sc_2agent`）
  - CSPは2人分の計画を立てますが、実際に動かすのは自分の担当キャラクターだけです（`human_counterpart_mode`）。
  - もう一方の計画は「人間がこう動くだろう」という推測に過ぎないため、人間の持ち物や所要時間の見積もりから推測が外れたことを検知すると、その場で再スケジューリングします。
  - 人間側スロットに割り当てたタスクは誰も実行しない可能性があるので、AIが手待ちになった場合や前提タスクが揃わない場合は、AIがそのタスクを引き受けます。
- **CSP + CSP**（`--agent0 CSP --agent1 CSP --sc_2agent`）
  - 両方のキャラクターをAIが操作します。

実装の要点：

- **スケジューリング**: OR-ToolsのCP-SATで、切る・調理する・提供するといったタスク全体の担当と順序を、移動コスト込みで最適化します。
- **実行**: 両エージェントとも毎フレーム行動を決定します（交互に動くターン制ではありません）。
- **衝突回避**: 経路探索（A*）では、相手が現在いるマスを動的障害物として扱います。同じマスで待ち合う状態が続いた場合は退避します。
- **共有置き場**: 複数の材料を1か所に集めてから運ぶため、注文ごとに置き場となるカウンターを割り当てます。

## リプレイ

プレイ終了時に、リプレイが `agent/agent/replay/` へ自動保存されます。

```bash
python agent/agent/replay_main.py --replay <ファイル名>
```

## LLMによる優先度・制約の生成

タスクの優先度重みと制約を、日本語の指示からLLMで生成する機能が `agent/agent/myagent/gui.py` に実装されています。ただし現在の `play_main.py` はCSPエージェントの起動時にこの設定GUIを開かず、既定値（重みなし・制約なし）で始まります。

利用する場合はAPIキーを環境変数に設定してください。モデル名が `gemini` で始まる場合は `GOOGLE_API_KEY`、それ以外（OpenAI）は `OPENAI_API_KEY` を参照します。リポジトリ直下の `google_api_key.txt` / `openai_api_key.txt` からも読み込みます。

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY = "your-api-key-here"
```

**Linux / macOS:**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## 既知の問題

- **`ring` 以外のマップでCSPエージェントが起動直後に落ちます。** `bottleneck` / `partition` / `quick` のマップには消火器が配置されており、食材として扱えないオブジェクトの状態を参照して `AttributeError: 'FireExtinguisher' object has no attribute 'get_state'` が発生します。現状、CSPエージェントでの実験は `ring` マップで行ってください。
- **スープの注文が途中で停止することがあります。** 鍋が全て埋まった状態で、鍋を空けるはずの提供タスクが後回しになると、双方が進まなくなります。サラダのみ、またはサラダ中心の注文構成では発生しません。
- **AI2体（`--agent0 CSP --agent1 CSP`）で相互に進路を塞ぎ合うことがあります。** 両者が同じマスへ進もうとして停止します。

## プロジェクト構成

- `agent/`: エージェントの実装が含まれています。
    - `agent/play_main.py`: ゲームの起動スクリプト
    - `agent/replay_main.py`: リプレイの再生スクリプト
    - `agent/gameplay.py`: ゲームループと指示画面の制御
    - `agent/instruction_panel.py`: 指示カードの描画
    - `agent/myagent/`: `CSPAgent`、`TaskAgent`、`GreedyAgent` など
    - `agent/TSP/`: `TSPSolverAgent`
- `testbed-cooking/`: Overcookedのゲーム環境（`gym-cooking`）が含まれています。
    - `gym_cooking/utils/levels/`: マップ定義
    - `gym_cooking/utils/order/`: 注文ファイル
    - `gym_cooking/utils/order_preset.py`: 注文プリセットの生成ルール
    - `gym_cooking/recipe_planner/recipe.py`: レシピ定義
