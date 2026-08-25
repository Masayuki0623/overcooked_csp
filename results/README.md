# 指示の質と skip_budget の効果（段階A: ring 単独・ソルバー水準）

`tools/run_instruction_experiment.py` で生成したデータと図。

- `instruction_experiment.csv` — 1試行1行。270試行（ring × 3指示質 × skip_budget d∈{0,2,4} × 30シード）
- `figures/fig1_natural_rank.png` — 指示の質ごとの自然順位の分布
- `figures/fig2_loss_vs_excess.png` — 効率損失 L と超過分の関係（仮説H2の核心）
- `figures/fig3_loss_by_budget.png` — skip_budget 別・指示の質別の平均 L
- `figures/fig4_inserted_tasks.png` — 前に挟まった他タスクの個数と合計時間
- `figures/fig5_infeasible_rate.png` — INFEASIBLE の発生率

再現手順:

```bash
python tools/run_instruction_experiment.py --map ring --seeds 30
python tools/plot_instruction_experiment.py
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
