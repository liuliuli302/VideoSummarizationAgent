from __future__ import annotations

from typing import Dict, Optional

from src.datasets.schemas import WindowScore


DEFAULT_SCORE_WEIGHTS = {
    "mainline": {"强": 3.0, "中": 1.5, "弱": 0.0},
    "novelty": {"高": 2.0, "中": 1.0, "低": 0.0},
    "event": {"高": 2.0, "中": 1.0, "低": 0.0},
    "temporal": {"高": 1.0, "中": 0.5, "低": 0.0},
    "domain": {"高": 1.0, "中": 0.5, "低": 0.0},
    "critic": {"高": 1.5, "中": 0.5, "低": -0.5},
}

DEFAULT_DECISION_THRESHOLDS = {
    "suggest_keep": 4.5,
    "optional": 2.0,
}


class AggregationAgent:
    """Aggregate baseline expert text outputs into a unified window decision."""

    def __init__(
        self,
        score_weights: Optional[Dict[str, Dict[str, float]]] = None,
        decision_thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        self.score_weights = self._merge_nested(DEFAULT_SCORE_WEIGHTS, score_weights or {})
        self.decision_thresholds = {**DEFAULT_DECISION_THRESHOLDS, **(decision_thresholds or {})}

    def run(
        self,
        win_id: str,
        expert_outputs: Dict[str, str],
        planner_plan: str = "",
        critic_output: str = "",
    ) -> WindowScore:
        score = 0.0
        reasons = []

        mainline_value = self._extract_value(expert_outputs.get("mainline", ""), "Mainline Judgment")
        novelty_value = self._extract_value(expert_outputs.get("novelty", ""), "Novelty")
        event_value = self._extract_value(expert_outputs.get("event", ""), "Event Importance")
        temporal_value = self._extract_value(expert_outputs.get("temporal", ""), "Temporal Value")
        domain_value = self._extract_value(expert_outputs.get("domain", ""), "Domain Match")
        critic_contribution = self._extract_value(critic_output, "Marginal Contribution")
        critic_recommendation = self._extract_value(critic_output, "Recommendation")

        score += self.score_weights["mainline"].get(mainline_value, 0.0)
        score += self.score_weights["novelty"].get(novelty_value, 0.0)
        score += self.score_weights["event"].get(event_value, 0.0)
        score += self.score_weights["temporal"].get(temporal_value, 0.0)
        score += self.score_weights["domain"].get(domain_value, 0.0)
        score += self.score_weights["critic"].get(critic_contribution, 0.0)

        if mainline_value == "强":
            reasons.append("主线推进明确")
        if novelty_value == "高":
            reasons.append("新颖性高")
        if event_value == "高":
            reasons.append("事件价值高")
        if temporal_value == "高":
            reasons.append("时序衔接强")
        if domain_value == "高":
            reasons.append("领域匹配高")
        if critic_contribution == "高":
            reasons.append("反事实边际贡献高")
        elif critic_recommendation == "可省略":
            reasons.append("反事实评论认为删除损失有限")

        if mainline_value == "强" and (event_value == "高" or novelty_value == "高"):
            decision = "必须保留"
            conflict = "主线推进与关键信息信号一致，直接提升为必须保留。"
        elif score >= self.decision_thresholds["suggest_keep"]:
            decision = "建议保留"
            conflict = "多数专家支持保留，未发现明显冲突。"
        elif score >= self.decision_thresholds["optional"]:
            decision = "可选"
            conflict = "部分专家支持保留，但整体证据强度中等。"
        else:
            decision = "建议省略"
            conflict = "主线、新颖性和事件信号均较弱，优先省略。"

        if critic_recommendation == "保留" and decision in {"建议保留", "可选"}:
            decision = "必须保留" if decision == "建议保留" else "建议保留"
            conflict = "反事实评论指出删除损失较大，因此上调窗口重要性。"
        elif critic_recommendation == "可省略" and decision in {"建议保留", "可选"}:
            decision = "可选" if decision == "建议保留" else "建议省略"
            conflict = "反事实评论认为删除损失有限，因此下调窗口重要性。"

        if planner_plan:
            conflict = f"{conflict} 规划参考：{planner_plan.splitlines()[0]}"

        reason_text = "；".join(reasons) if reasons else "专家信号整体偏弱。"
        base_decision = self._format_decision_text(decision, reason_text, conflict)

        return WindowScore(
            win_id=win_id,
            expert_opinions=expert_outputs,
            base_decision=base_decision,
            cf_comment=critic_output or "未启用 CMCC",
            final_importance=decision,
        )

    def _extract_value(self, text: str, prefix: str) -> str:
        for line in text.splitlines():
            if line.startswith(f"{prefix}:"):
                return line.split(":", maxsplit=1)[1].strip()
        return ""

    def _format_decision_text(self, decision: str, reason: str, conflict: str) -> str:
        return f"Final Importance: {decision}\nReason: {reason}\nConflict Handling: {conflict}"

    def _merge_nested(
        self,
        base: Dict[str, Dict[str, float]],
        override: Dict[str, Dict[str, float]],
    ) -> Dict[str, Dict[str, float]]:
        merged = {key: value.copy() for key, value in base.items()}
        for key, value in override.items():
            if key not in merged or not isinstance(value, dict):
                continue
            merged[key].update(value)
        return merged