"""
協調スキル推定器（Cooperative Skill Estimator）

CSPエージェントが常時計算するJoint最適スケジュールを利用し、
プレイヤーの「相手（AI）の意図を先読みし、その行動を阻害しないように立ち回る能力」
を定量化する。

定義:
  V_coop(t→t+1) = Distance(A_virtual(t+1), A_plan(t+1))
    A_virtual: 人間が何もしなかった場合の仮想計画
    A_plan:    実際のAI計画
  S_coop = max(0, 1.0 - V_coop / Max_Volatility)
  S_ema  = α * S_coop + (1-α) * S_ema   (EMAで平滑化)

参考: Gutwin & Greenberg (2002) の協調意識モデル
"""


class SkillEstimator:
    """協調スキル推定器"""

    def __init__(self, alpha=0.3):
        """
        Args:
            alpha: EMA平滑化係数 (0.0〜1.0)。大きいほど直近の値に敏感。
        """
        self.alpha = alpha
        self.s_ema = 1.0              # 初期スキル値（最大=協調的と仮定）
        self.history = []             # (time, V_coop, S_coop, S_ema) の時系列履歴
        self.prev_ai_plan = None      # A_plan(t): 前回のAIタスク配列（タスクIDのリスト）

    @staticmethod
    def _log(message):
        print(f"[SkillEstimator] {message}")

    def levenshtein_distance(self, seq1, seq2):
        """
        2つのタスクID配列間のレーベンシュタイン距離（編集距離）を計算する。

        Args:
            seq1: タスクIDのリスト e.g. [('chop','tomato',0), ('cook','tomato soup',0), ...]
            seq2: タスクIDのリスト（同上）

        Returns:
            int: 編集距離（挿入・削除・置換のコスト = 各1）
        """
        n = len(seq1)
        m = len(seq2)

        self._log(f"Levenshtein開始: len(A_virtual)={n}, len(A_plan)={m}")
        self._log(f"  A_virtual = {seq1}")
        self._log(f"  A_plan     = {seq2}")

        # DPテーブル
        dp = [[0] * (m + 1) for _ in range(n + 1)]

        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j

        self._log(f"DP初期化 row0 = {dp[0]}")

        for i in range(1, n + 1):
            self._log(f"  行{i}計算開始: {seq1[i - 1]}")
            for j in range(1, m + 1):
                cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
                delete_cost = dp[i - 1][j] + 1
                insert_cost = dp[i][j - 1] + 1
                replace_cost = dp[i - 1][j - 1] + cost
                dp[i][j] = min(
                    delete_cost,       # 削除
                    insert_cost,        # 挿入
                    replace_cost        # 置換
                )

                if cost == 0:
                    op = "一致"
                elif dp[i][j] == delete_cost:
                    op = "削除"
                elif dp[i][j] == insert_cost:
                    op = "挿入"
                else:
                    op = "置換"

                self._log(
                    f"    cell[{i}][{j}] compare {seq1[i - 1]} vs {seq2[j - 1]}: "
                    f"del={delete_cost}, ins={insert_cost}, rep={replace_cost} -> {dp[i][j]} ({op})"
                )

            self._log(f"  行{i}完了 = {dp[i]}")

        self._log(f"Levenshtein終了: V_coop={dp[n][m]}")
        return dp[n][m]

    @staticmethod
    def extract_plan_ids(schedule_per_agent, agent_idx=1):
        """
        スケジュール結果からAI側のタスクID配列を抽出する。

        Args:
            schedule_per_agent: {0: [...], 1: [...]} 形式のスケジュール
            agent_idx: AI側のエージェント番号（デフォルト=1）

        Returns:
            list: タスクIDのリスト（開始時刻順） e.g. [('chop','tomato',0), ...]
        """
        if schedule_per_agent is None:
            return []
        agent_schedule = schedule_per_agent.get(agent_idx, [])
        # 開始時刻順にソート済みのはずだが念のため
        sorted_schedule = sorted(agent_schedule, key=lambda x: x.get('start', 0))
        return [task['id'] for task in sorted_schedule]

    def update(self, time, virtual_plan_ids, actual_plan_ids):
        """
        スキル推定値を更新する。

        タスク変化が発生したタイミングで呼ばれ、仮想計画と実際の計画の差異から
        協調スキルを計算する。

        Args:
            time: 現在のゲーム内時刻
            virtual_plan_ids: 仮想計画のAIタスクID配列（人間が静止した場合）
            actual_plan_ids:  実際の計画のAIタスクID配列
        """
        self._log(f"時刻 {time:.1f} のスキル推定を開始")
        self._log(f"仮想計画 A_virtual = {virtual_plan_ids}")
        self._log(f"実際計画 A_plan     = {actual_plan_ids}")

        # V_coop: 仮想計画と実際の計画の編集距離
        v_coop = self.levenshtein_distance(virtual_plan_ids, actual_plan_ids)
        self._log(f"V_coop = levenshtein_distance(A_virtual, A_plan) = {v_coop}")

        # Max_Volatility: 完全に入れ替わった場合の最大距離
        max_vol = max(len(virtual_plan_ids), len(actual_plan_ids), 1)
        self._log(f"Max_Volatility = max(len(A_virtual), len(A_plan), 1) = {max_vol}")

        # S_coop: 正規化スキル値 (0.0〜1.0)
        s_coop = max(0.0, 1.0 - v_coop / max_vol)
        self._log(f"S_coop = max(0, 1 - V_coop / Max_Volatility) = {s_coop:.3f}")

        prev_s_ema = self.s_ema

        # EMA平滑化
        self.s_ema = self.alpha * s_coop + (1.0 - self.alpha) * self.s_ema
        self._log(
            f"S_ema = alpha * S_coop + (1 - alpha) * prev_S_ema = {self.alpha:.3f} * {s_coop:.3f} + {1.0 - self.alpha:.3f} * {prev_s_ema:.3f} = {self.s_ema:.3f}"
        )

        # 履歴に記録
        entry = {
            'time': time,
            'v_coop': v_coop,
            's_coop': s_coop,
            's_ema': self.s_ema,
            'virtual_plan': list(virtual_plan_ids),
            'actual_plan': list(actual_plan_ids),
        }
        self.history.append(entry)

        # スキルレベルの解釈
        if s_coop >= 0.9:
            level = "★★★ 高スキル（先読み○・不阻害）"
        elif s_coop >= 0.5:
            level = "★★☆ 中スキル（部分的な干渉あり）"
        else:
            level = "★☆☆ 低スキル（干渉・阻害あり）"

        self._log(f"判定 = {level}")

        return entry

    def get_current_score(self):
        """現在のEMA平滑化済みスキル値を返す"""
        return self.s_ema

    def get_history(self):
        """スキル推定の全履歴を返す（リプレイ保存用）"""
        return self.history

    def get_summary(self):
        """ゲーム終了時のサマリーを返す"""
        if not self.history:
            return {
                'final_score': self.s_ema,
                'num_measurements': 0,
                'avg_v_coop': 0,
                'avg_s_coop': 1.0,
            }

        avg_v_coop = sum(h['v_coop'] for h in self.history) / len(self.history)
        avg_s_coop = sum(h['s_coop'] for h in self.history) / len(self.history)

        return {
            'final_score': self.s_ema,
            'num_measurements': len(self.history),
            'avg_v_coop': avg_v_coop,
            'avg_s_coop': avg_s_coop,
        }

    def print_final_report(self):
        """ゲーム終了時の最終レポートを出力する"""
        summary = self.get_summary()
        self._log("最終レポート")
        self._log(f"計測回数 = {summary['num_measurements']}")
        self._log(f"平均 V_coop = {summary['avg_v_coop']:.3f}")
        self._log(f"平均 S_coop = {summary['avg_s_coop']:.3f}")
        self._log(f"最終 S_ema = {summary['final_score']:.3f}")

        score = summary['final_score']
        if score >= 0.9:
            grade = "A（優秀な協調能力）"
        elif score >= 0.7:
            grade = "B（良好な協調能力）"
        elif score >= 0.5:
            grade = "C（平均的な協調能力）"
        elif score >= 0.3:
            grade = "D（改善の余地あり）"
        else:
            grade = "E（協調が困難な状態）"
        self._log(f"総合評価 = {grade}")
