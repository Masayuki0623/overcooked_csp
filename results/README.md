# 指示の質と skip_budget の効果（ソルバー水準の自動計測）

`tools/run_instruction_experiment.py` で生成したデータと図。

条件は2つ。どちらも 3指示質 × skip_budget d∈{0,2,4} × 30シード = 270試行。

- `instruction_experiment.csv` — ring / experiment1（サラダ2 + スープ1、仕切りなし）
  悪い指示 = サラダ専用の下ごしらえ
- `instruction_experiment_juice.csv` — experiment / experiment2
  （サラダ1 + スープ1 + ジュース1、仕切りあり）
  悪い指示 = ジュース（フルーツ）の下ごしらえ。ミキサーは人間側にあるため、
  「自分の作業を早く始めたい」という動機のある悪い指示になる
- `figures/fig1_natural_rank.png` — 指示の質ごとの自然順位の分布
- `figures/fig2_loss_vs_excess.png` — 効率損失 L と超過分の関係（仮説H2の核心）
- `figures/fig3_loss_by_budget.png` — skip_budget 別・指示の質別の平均 L
- `figures/fig4_inserted_tasks.png` — 前に挟まった他タスクの個数と合計時間
- `figures/fig5_infeasible_rate.png` — INFEASIBLE の発生率

再現手順:

```bash
python tools/run_instruction_experiment.py --map ring --seeds 30
python tools/run_instruction_experiment.py --map experiment --preset experiment2 --seeds 30     --out results/instruction_experiment_juice.csv
python tools/plot_instruction_experiment.py     --csv results/instruction_experiment.csv,results/instruction_experiment_juice.csv
```

注意点:

- 対象タスクは `(動詞, 対象)` 単位でまとめられており、複数の注文にまたがる
  ことがある。自然順位はそのうち最も早いものを採用している。
- skip_budget の水準は {0,2,4}。第1回の {0,1,2} では d=1 と d=2 でLが変わる
  試行が9%しかなく、水準として機能していなかった。
- 注文生成側に「サラダにしか使わない具材が必ず1つ以上ある」制約を入れたため
  (order_preset.has_exclusive_salad_ingredient)、bad が定義できないシードは
  発生しない。
- `inserted_count` は依存タスクも含めた「対象の前にある同エージェントのタスク数」。
  skip_budget の勘定は依存タスクを除外するため、両者は一致しないことがある。

## 主な結果

仮説H2（自然順位が skip_budget を超えたときにだけ効率損失が出る）は
両条件で完全に成立した（超過分=0 の 322試行すべてで L=0.00、最大も0.00）。

指示の質による差は、仕切りありのジュース構成の方がはるかに大きい。

| 条件 | d=0 で悪い指示に損失が出た割合 | 悪い指示 − 良い指示 の平均差(d=0) |
| --- | --- | --- |
| ring / experiment1 | 67% | +1.2 秒 |
| experiment / experiment2 | **100%** | **+4.1 秒** |

skip_budget の水準は d=0 と d=2 で明確に差が出る（57%の試行でLが変化）が、
d=2 と d=4 の差は小さい（17%）。d=4 では悪い指示でも損失がゼロになる。
