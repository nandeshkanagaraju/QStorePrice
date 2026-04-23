"""Scores the quality of a valid Operating Brief independently of reward.

The quality score is the independent research metric — tracks whether
reasoning quality correlates with WRR over training.

Rule-based scoring only. No LLM calls. No randomness.
"""

from __future__ import annotations

import re

from freshprice_env.entities import SimulatedMarketState
from freshprice_env.enums import (
    BatchStatus,
    BriefEngineType,
    ExpiryUrgency,
    FarmerOfferStatus,
    TrendAction,
)


class BriefQualityScorer:
    """Scores brief quality 0.0-1.0 as a proxy metric for reasoning quality."""

    @staticmethod
    def score(
        brief: dict,
        state: SimulatedMarketState,
    ) -> float:
        """Return a quality score 0.0-1.0.

        Composed of three sub-scores:
          situation_score  × 0.33
          reasoning_score  × 0.34
          directive_score  × 0.33

        All scoring is keyword/pattern based — simple string contains checks.
        This is a proxy metric, not ground truth.
        """
        engine_type = brief["engine_type"]

        situation_score = BriefQualityScorer._score_situation(brief, state)
        reasoning_score = BriefQualityScorer._score_reasoning(brief, state, engine_type)
        directive_score = BriefQualityScorer._score_directive(brief, state, engine_type)

        total = situation_score * 0.33 + reasoning_score * 0.34 + directive_score * 0.33
        return max(0.0, min(1.0, total))

    # ------------------------------------------------------------------
    # Situation sub-score
    # ------------------------------------------------------------------

    @staticmethod
    def _score_situation(brief: dict, state: SimulatedMarketState) -> float:
        situation = brief.get("situation", "")
        confidence = brief.get("confidence")
        score = 0.0

        # 0.2 — situation text is not empty
        if situation and len(situation.strip()) > 10:
            score += 0.2

        # 0.2 — mentions at least one specific batch_id or category name
        categories = {b.category for b in state.batches}
        batch_ids = {b.batch_id for b in state.batches}
        situation_lower = situation.lower()
        if any(cat.lower() in situation_lower for cat in categories):
            score += 0.2
        elif any(bid in situation for bid in batch_ids):
            score += 0.2

        # 0.2 — mentions urgency level
        urgency_keywords = {"urgent", "critical", "watch", "fresh", "expir", "at-risk", "at risk"}
        if any(kw in situation_lower for kw in urgency_keywords):
            score += 0.2

        # 0.2 — mentions a specific number
        if re.search(r"\d+", situation):
            score += 0.2

        # 0.2 — confidence is consistent with state
        score += BriefQualityScorer._score_confidence_consistency(confidence, state)

        return score

    @staticmethod
    def _score_confidence_consistency(confidence, state: SimulatedMarketState) -> float:
        """Check if confidence level is consistent with state."""
        if confidence is None:
            return 0.0

        confidence_val = confidence.value if hasattr(confidence, "value") else str(confidence).upper()

        critical_count = sum(
            1 for b in state.batches
            if b.status == BatchStatus.ACTIVE and b.urgency == ExpiryUrgency.CRITICAL
        )
        low_buffer = state.risk_buffer_balance < 2000.0

        # HIGH confidence with multiple CRITICALs and low buffer = inconsistent
        if confidence_val == "HIGH" and critical_count >= 2 and low_buffer:
            return 0.0

        # LOW confidence when no CRITICALs = slightly inconsistent but acceptable
        if confidence_val == "LOW" and critical_count == 0 and not low_buffer:
            return 0.1

        # Otherwise consistent
        return 0.2

    # ------------------------------------------------------------------
    # Reasoning sub-score
    # ------------------------------------------------------------------

    @staticmethod
    def _score_reasoning(
        brief: dict, state: SimulatedMarketState, engine_type: BriefEngineType,
    ) -> float:
        if engine_type == BriefEngineType.FARMER:
            return BriefQualityScorer._score_farmer_reasoning(brief, state)
        if engine_type == BriefEngineType.TREND:
            return BriefQualityScorer._score_trend_reasoning(brief, state)
        return BriefQualityScorer._score_pricing_reasoning(brief, state)

    @staticmethod
    def _score_pricing_reasoning(brief: dict, state: SimulatedMarketState) -> float:
        """1.0 — recommendation mentions at least one specific batch and price decision."""
        recommendation = brief.get("recommendation", "")
        rec_lower = recommendation.lower()

        # Check for batch reference (batch_id or category name)
        batch_ids = {b.batch_id for b in state.batches if b.status == BatchStatus.ACTIVE}
        categories = {b.category for b in state.batches if b.status == BatchStatus.ACTIVE}

        has_batch_ref = (
            any(bid in recommendation for bid in batch_ids)
            or any(cat.lower() in rec_lower for cat in categories)
        )

        # Check for price-related language
        price_keywords = {"discount", "price", "reduce", "flash", "hold", "maintain", "multiplier"}
        has_price_ref = any(kw in rec_lower for kw in price_keywords)

        if has_batch_ref and has_price_ref:
            return 1.0
        if has_batch_ref or has_price_ref:
            return 0.5
        return 0.0

    @staticmethod
    def _score_farmer_reasoning(brief: dict, state: SimulatedMarketState) -> float:
        score = 0.0
        viability_check = brief.get("viability_check")
        recommendation = brief.get("recommendation", "")

        # 0.5 — viability_check contains PASS/FLAG/FAIL
        if viability_check and isinstance(viability_check, dict):
            outcomes_found = 0
            for key, val in viability_check.items():
                if isinstance(val, dict) and "outcome" in val:
                    outcomes_found += 1
                elif isinstance(val, str) and val.upper() in ("PASS", "FLAG", "FAIL"):
                    outcomes_found += 1
            if outcomes_found >= 3:
                score += 0.5
            elif outcomes_found >= 1:
                score += 0.25

        # 0.5 — recommendation mentions a specific price or viability score
        rec_lower = recommendation.lower()
        has_number = bool(re.search(r"\d+\.?\d*", recommendation))
        has_viability_ref = any(
            kw in rec_lower for kw in ("viability", "score", "viable", "risk")
        )
        if has_number and has_viability_ref:
            score += 0.5
        elif has_number or has_viability_ref:
            score += 0.25

        return score

    @staticmethod
    def _score_trend_reasoning(brief: dict, state: SimulatedMarketState) -> float:
        score = 0.0
        signal_analysis = brief.get("signal_analysis") or ""
        recommendation = brief.get("recommendation", "")

        # 0.5 — signal_analysis mentions the composite_score value
        has_score_mention = bool(re.search(r"\d{2,3}", signal_analysis))
        score_keywords = {"composite", "score", "signal", "trend"}
        has_score_context = any(kw in signal_analysis.lower() for kw in score_keywords)
        if has_score_mention and has_score_context:
            score += 0.5
        elif has_score_mention:
            score += 0.25

        # 0.5 — recommendation references conversion or demand
        rec_lower = recommendation.lower()
        demand_keywords = {"conversion", "demand", "velocity", "historical", "uplift", "sales"}
        if any(kw in rec_lower for kw in demand_keywords):
            score += 0.5
        elif any(kw in rec_lower for kw in ("approve", "decline", "order")):
            score += 0.25

        return score

    # ------------------------------------------------------------------
    # Directive sub-score
    # ------------------------------------------------------------------

    @staticmethod
    def _score_directive(
        brief: dict, state: SimulatedMarketState, engine_type: BriefEngineType,
    ) -> float:
        directive = brief.get("directive", {})
        actions = directive.get("actions", [])
        score = 0.0

        # 0.4 — directive JSON is valid (always true if we got this far)
        score += 0.4

        # 0.3 — action count matches expected
        score += BriefQualityScorer._score_action_coverage(
            actions, state, engine_type,
        )

        # 0.3 — at least one non-trivial action
        score += BriefQualityScorer._score_action_nontrivial(
            actions, state, engine_type,
        )

        return score

    @staticmethod
    def _score_action_coverage(
        actions: list, state: SimulatedMarketState, engine_type: BriefEngineType,
    ) -> float:
        """0.3 if action count matches expected targets."""
        if engine_type == BriefEngineType.PRICING:
            urgent_critical = sum(
                1 for b in state.batches
                if b.status == BatchStatus.ACTIVE
                and b.urgency in (ExpiryUrgency.URGENT, ExpiryUrgency.CRITICAL)
            )
            if urgent_critical == 0:
                return 0.3 if len(actions) > 0 else 0.15
            if len(actions) >= urgent_critical:
                return 0.3
            return 0.15

        if engine_type == BriefEngineType.FARMER:
            pending = sum(
                1 for o in state.pending_offers
                if o.status == FarmerOfferStatus.PENDING
            )
            if pending == 0:
                return 0.3
            if len(actions) == pending:
                return 0.3
            if len(actions) > 0:
                return 0.15
            return 0.0

        if engine_type == BriefEngineType.TREND:
            actionable = sum(
                1 for sig in state.trend_signals.values()
                if sig.action_taken == TrendAction.PENDING
                and sig.is_actionable(state.tick)
            )
            if actionable == 0:
                return 0.3
            if len(actions) == actionable:
                return 0.3
            if len(actions) > 0:
                return 0.15
            return 0.0

        return 0.0

    @staticmethod
    def _score_action_nontrivial(
        actions: list, state: SimulatedMarketState, engine_type: BriefEngineType,
    ) -> float:
        """0.3 if at least one action is non-trivial."""
        if not actions:
            return 0.0

        if engine_type == BriefEngineType.PRICING:
            # At least one action that isn't just holding at 1.0
            has_nontrivial = any(
                a.get("price_multiplier", 1.0) != 1.0
                or a.get("flash_sale", False)
                for a in actions
            )
            return 0.3 if has_nontrivial else 0.0

        if engine_type in (BriefEngineType.FARMER, BriefEngineType.TREND):
            # At least one non-DECLINE action when signals are strong
            has_positive = any(
                a.get("decision", "").upper() in ("ACCEPT", "COUNTER", "APPROVE")
                for a in actions
            )
            return 0.3 if has_positive else 0.15

        return 0.0
