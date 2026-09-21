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

## ブラウザからのプレイ (`server.py`)

`server.py` を起動すると、ブラウザからプレイできます。pygame をウィンドウなし（`SDL_VIDEODRIVER=dummy`）で動かし、pygame が描くときに使った命令（どの絵をどこに描くか）だけをWebSocketで送ってブラウザで描き直し、ブラウザのキー入力をpygameのイベントに戻す方式です。ゲームの描画・指示パネル・イベント処理はローカル版と同じコードがそのまま動き、見た目も同じになります。送る量は1コマ約200バイトです（画像で送ると約9KB）。指示パネルを開いている間だけは画像で送ります。URLに `?mode=png` を付けると、常に画像で送る従来の方式になります。

```bash
pip install fastapi uvicorn
python server.py
```

起動後、表示されたURL（既定は `http://localhost:8000/`）を開きます。操作は矢印キーでの移動（長押しで歩き続けます）、`Space` で向いている先に手を出す（拾う・置く・刻む・入れる）、`Enter` で指示パネル（パネル内はカードをクリック）です。

### 操作と時間の刻み

- **「使う」を押しっぱなしにしても、置く・取るは1回だけです。** 切る・混ぜるは押している間ずっと続きます。ただし**まな板に置いたときだけは例外**で、そのまま「置く→切る→取る」まで1回の長押しでできます。もう一度できるようになるのは、**手を離したとき**か、**向きを変えたとき**です（押しっぱなしのまま別の台を向けば、そのまま置けます）。
- **キャラの絵は向きに合わせて変わります**（前・後ろ・横の3種類。横は左右で反転）。自分のキャラは、押した瞬間に向きも絵も変わります（通信を待ちません）。絵は `python tools/gen_agent_sprites.py` で作り直せます。向いている先のマスには青い枠も出ます。
- **移動と手を出すのは別の操作です。** 台や器具の方向へ移動を押しても手は出ません（その向きを向くだけ）。向いている先は青い枠で示されます。手を出すのは `Space`（スマホは十字キーの真ん中の「使う」ボタン）です。
- **1秒あたりに行動できる回数は `--input-hz`（既定 5）で決めます。** ゲームの1手は 1/n 秒で、人も AI も同じ速さです。n=5 なら1秒に5マス進みます。
- まな板で刻む・ミキサーで混ぜる回数は n に合わせて減らすので、**かかる秒数は n を変えても同じ**です（10Hz: 8回=0.8秒 / 5Hz: 4回=0.8秒）。鍋で煮る時間や焦げるまでの時間は秒で決まっているので変わりません。
- CSP の見積もりも n と「向きを変える1手」を織り込みます（1回の手出し＝向く＋出すの2手。続けて手を出すときは1手ずつ）。
- ブラウザ側では、マス目の仕組みはそのままに、**1手ぶんの時間をかけて隣のマスへ滑らせて見せます**（見た目だけの連続移動）。自分のキャラは押した瞬間に動き、AI は届いた位置の間を滑ります。


画面の下に矢印ボタンと「使う」ボタンがあり、キーボードの無いスマホでも同じ操作ができます（指示パネルのカードはタップで選べます）。十字キーはスティックのように使えます。押したまま指を滑らせると、指の位置に応じて上下左右に切り替わります（斜めには進みません）。スマホではページ全体を画面の高さに収め（スクロールしない）、ボタンは画面下に固定します。ボタンの大きさ・画面下からの高さ・十字キーの左右は「⚙ 設定」で変えられ、端末ごとに保存されます。ページを開いただけではゲームは始まりません。最初に「ステージ」を選び、「スタート」を押すと始まります。

| 選ぶもの | 選択肢 |
| --- | --- |
| 地図 | **仕切り**（`exp_partition`、左右が行き来できない）/ **ボトルネック**（`exp_bottleneck`、仕切りの真ん中に1マスだけ穴）/ **リング**（`exp_ring`、真ん中の島のまわりを回れる） |
| レシピ | **野菜のみ**（`experiment1`、サラダ2品＋スープ1品、27通り）/ **野菜＋フルーツ**（`experiment2`、サラダ＋スープ＋ジュース、18通り） |
| 注文の組み合わせ | そのレシピの組み合わせから1つ、または「おまかせ」（毎回ランダム） |

3つの地図は、器具と材料の置き場所が同じで、左右のつながり方だけが違います。左が AI 側（鍋・ミキサー・提供口・玉ねぎ・リンゴ）、右が人間側（レタス・トマト・オレンジ・バナナ）で、まな板・皿・コップは両側にあります。この配置は `tools/layout_search.py` で、二人の仕事量がおおむね半々になるものを探して選びました。選んだ内容は端末に覚えられ、リプレイにも記録されます。

自分のキャラは、矢印を押した瞬間に端末側で1マス動かして見せます（先読み）。サーバーから正しい位置が届いたら、まだ届いていない操作ぶんを当てはめ直して合わせます。これにより、**押してから自分が動くまでの時間は通信の速さに左右されません**（遅れて見えるのは AI の動きだけになります）。そのため、最初の1回だけ通れるマスの地図と1マスの大きさを渡し、毎コマ「自分の位置・相手の位置・サーバーが処理した入力の数」を一緒に送ります。押しすぎた分の扱い（1秒に10回まで、溜めるのは3つまで）は端末とサーバーでそろえてあります。

盤面は、端末が「描いた」と返事をするまで送りすぎません。返事待ちで送れる数は往復時間に合わせて 3〜8 コマの間で決めます（往復1秒の回線でも毎秒10コマ届きます。窓が固定だと毎秒3コマまで落ちていました）。

「スタート」を押すとサーバーがゲームを組み立て、最初の盤面を送ります。このときゲームの時間は止まっています。端末で盤面が描き終わると 3・2・1 のカウントダウンが出て、終わった時点で時間が進み始めます。ゲームで使う絵はページを開いた時点で全部読み込み、読み終わるまで「スタート」は押せません（スマホで絵が描けていないうちにゲームが始まらないようにするため）。ブラウザに配る絵は、サーバーの起動時に 80px に縮めた版を `.cache/web_graphics/` に作って使います（元の絵は合計約 13MB、縮めると約 600KB。画面では 1マス 40px で描くので見た目は変わりません）。読み込み画面には合計の容量を表示します。2回目以降はブラウザに保存された絵を使うので、ほとんど落としません。カウントダウン中の操作は送られません。ボタンは押した瞬間に1回だけ入力され、押しっぱなしでは連続しません（キーボード側もキーリピートを無視しているのとそろえています）。

操作できるのは同時に1人だけです。遊んでいる人がいる間に別の人が開くと、「別の人がプレイ中です」と残り時間が表示され、終わりしだい自動でその人の番になります。1ゲーム終わるたびにサーバーが次のゲームを用意し直すので、再起動は要りません。遊んでいる人が途中でページを閉じる・再読み込みすると枠が空き、次に開いた人が続きから操作します。

外部から遊べるようにするには、Tailscale Funnel で公開します（URL は PC の名前で決まり、固定です）。

```bash
tailscale funnel --bg 8000
```

止めるときは `tailscale funnel --https=443 off` です。

ただし Funnel は中継が遠く、同じ Wi-Fi からでも往復1秒近くかかることがありました（この PC から東京の中継までは 16ms なので、遅いのは中継から端末までの経路です）。そこで **固定 URL は受付だけにして、遊ぶ画面は Cloudflare のトンネル（往復 30ms 前後）へ送ります**。

```bash
winget install --id Cloudflare.cloudflared -e   # 最初の1回だけ
python server.py --host 0.0.0.0                 # ゲームのサーバー
python tools/serve_public.py                    # 別の窓で。トンネルを立てる
```

`tools/serve_public.py` は、立てたトンネルの URL を `.cache/public_url.txt` に書きます。固定 URL（`https://desktop-1.tail9a3ca5.ts.net/`）を開いた人は、そこへ自動で移動します。URL は起動のたびに変わりますが、参加者に伝えるのは固定 URL のままで構いません。止めるとファイルを消すので、固定 URL はそのまま遊べる画面に戻ります。家の中から遊ぶときは `http://192.168.3.7:8000/`（LAN 直）がいちばん速いです。

### 実験モード（参加者IDを入れて遊ぶ）

最初の画面で「参加者ID」を入れると、実験のセッションになります。地図とレシピは選べなくなり、その参加者に割り当てた条件どおりに始まります。IDを空にすれば、これまでどおり自由に遊べます。

- 条件は **地図3種 × AI が指示を後回しにできる量（skip_budget 0 / 2 / 4）= 9通り**。1人が9セッション全部を遊び、**順番だけ人ごとにランダム**です。並びは `results/assignments.json` に残るので、途中でサーバーを止めても続きから遊べます。
- 注文は `experiment2`（サラダ＋スープ＋ジュース）の中から、指示の良し悪しが判別できる構成をランダムに選びます。指示は開始直後に1回だけ（見送り不可）です。
- 途中で接続が切れた回は記録に残しますが、セッション数には数えません（同じ条件をやり直します）。

セッションが終わるとアンケートが出ます（全14項目・1分程度）。パートAは協調感（CCR尺度 Coordination 因子、6項目・5段階）、パートBはAIパートナーへの信頼感（MDMT v1 Capacity Trust、8項目・0〜7の8段階＋「あてはまらない」）です。全項目必須で、「あてはまらない」は欠損として平均から除きます（0点ではありません）。

| ファイル | 内容 |
| --- | --- |
| `results/survey.csv` | 1行 = 1セッションの回答。`coord_1`〜`coord_6` と平均 `coord_mean`、`trust_1`〜`trust_8` と平均 `trust_mean`、「あてはまらない」の数 `trust_dnf_count`、条件（`map` / `skip_budget` / `case`）と結果（`served` / `makespan_s`） |
| `results/web_sessions.csv` | そのセッションの客観データ（提供数・所要時間・打ち切りかどうかなど） |
| `results/assignments.json` | 参加者ごとの条件の並びと、終わったセッション数 |

アンケートと客観データは `participant_id` と `session` で突き合わせます。

`play_main.py` と同じオプション名を受け付けるため、実験条件は同じ書き方で指定できます。既定値は実験用の構成になっています。

| 引数 | 既定値 |
| --- | --- |
| `--host` / `--port` | `127.0.0.1` / `8000` |
| `--agent0` / `--agent1` | `CSP` / `human` |
| `--map` | `ring` |
| `--sc_2agent` | 有効（無効にするなら `--no-sc_2agent`） |
| `--orders` | `experiment1` |
| `--instruction_request_timing` | `free` |
| `--deadline` | `0` |

LANの別端末から接続する場合は `--host 0.0.0.0` を指定します。

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
- `server.py`: ブラウザからプレイするための実験用サーバー
- `web/index.html`: ブラウザ側のクライアント
- `testbed-cooking/`: Overcookedのゲーム環境（`gym-cooking`）が含まれています。
    - `gym_cooking/utils/levels/`: マップ定義
    - `gym_cooking/utils/order/`: 注文ファイル
    - `gym_cooking/utils/order_preset.py`: 注文プリセットの生成ルール
    - `gym_cooking/recipe_planner/recipe.py`: レシピ定義
