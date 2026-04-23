"""CounterfactualEngine — shadow expert policy for regret scoring and DPO pair synthesis.

Runs alongside the agent, identifies high-regret decisions, and generates
synthetic rejected briefs when real pairs cannot be found.
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass

from freshprice_env.constants import (
    PRICE_MULTIPLIER_MIN,
    R2_MISSED_OPPORTUNITY_PENALTY,
    R2_RECKLESS_ACCEPT_PENALTY,
    R3_OVERTRADE_PENALTY,
    R3_PERFECT_TIMING_BONUS,
    RISK_BUFFER_INITIAL_SEED_RS,
)
from freshprice_env.entities import SimulatedMarketState
from freshprice_env.enums import (
    BatchStatus,
    BriefEngineType,
    ExpiryUrgency,
    FarmerOfferStatus,
    TrendAction,
)

logger = logging.getLogger(__name__)

# Max regret reference values per engine type (for normalisation)
_MAX_REGRET_PRICING: float = 2.0   # approximate: multiple CRITICAL batches expiring
_MAX_REGRET_FARMER: float = R2_MISSED_OPPORTUNITY_PENALTY + R2_RECKLESS_ACCEPT_PENALTY
_MAX_REGRET_TREND: float = R3_PERFECT_TIMING_BONUS + R3_OVERTRADE_PENALTY

# Probability of using LLM-based synthetic rejection (Mode 2)
_LLM_REJECTION_PROBABILITY: float = 0.20


@dataclass
class ExpertDecision:
    """The decision a rules-based expert policy would have made."""

    tick: int
    engine_type: BriefEngineType
    recommended_action: dict        # What the expert would put in DIRECTIVE
    reasoning: str                  # Plain language explanation
    expected_reward_delta: float    # Expert's expected reward from this action


class CounterfactualEngine:
    """Shadow expert policy for regret scoring and synthetic DPO pair generation."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    # ------------------------------------------------------------------
    # Expert decision computation
    # ------------------------------------------------------------------

    def compute_expert_decision(
        self,
        state: SimulatedMarketState,
        engine_type: BriefEngineType,
        tick: int,
    ) -> ExpertDecision:
        """Compute the rules-based expert policy decision for the current state.

        The expert is a deterministic heuristic with full state visibility.
        It is NOT the LLM — it always makes the locally optimal decision.
        """
        if engine_type == BriefEngineType.PRICING:
            return self._expert_pricing(state, tick)
        if engine_type == BriefEngineType.FARMER:
            return self._expert_farmer(state, tick)
        return self._expert_trend(state, tick)

    def _expert_pricing(self, state: SimulatedMarketState, tick: int) -> ExpertDecision:
        """Expert pricing: aggressive on CRITICAL, proportional on URGENT, hold on FRESH/WATCH."""
        actions: list[dict] = []
        reasoning_parts: list[str] = []
        expected_delta = 0.0

        for batch in state.batches:
            if batch.status != BatchStatus.ACTIVE:
                continue

            if batch.urgency == ExpiryUrgency.CRITICAL:
                if batch.quantity_remaining > 5:
                    multiplier = max(PRICE_MULTIPLIER_MIN, 0.40)
                    has_credits = state.notification_credits.get(batch.category, 0) > 0
                    actions.append({
                        "batch_id": batch.batch_id,
                        "price_multiplier": multiplier,
                        "flash_sale": has_credits,
                        "bundle_with": None,
                    })
                    reasoning_parts.append(
                        f"CRITICAL {batch.batch_id}: deep discount at {multiplier}, "
                        f"flash_sale={'yes' if has_credits else 'no'}"
                    )
                else:
                    actions.append({
                        "batch_id": batch.batch_id,
                        "price_multiplier": PRICE_MULTIPLIER_MIN,
                        "flash_sale": False,
                        "bundle_with": None,
                    })
                    reasoning_parts.append(
                        f"CRITICAL {batch.batch_id}: clearance at min multiplier (low stock)"
                    )
                expected_delta += 0.15  # near-expiry clearance bonus expected

            elif batch.urgency == ExpiryUrgency.URGENT:
                velocity = state.sales_velocity.get(batch.batch_id, 0.0)
                if batch.hours_to_expiry > 0:
                    velocity_needed = batch.quantity_remaining / batch.hours_to_expiry
                else:
                    velocity_needed = 999.0

                if velocity < velocity_needed * 0.7:
                    ratio = velocity / velocity_needed if velocity_needed > 0 else 0
                    multiplier = round(0.55 + ratio * 0.35, 2)
                    multiplier = max(PRICE_MULTIPLIER_MIN, min(1.0, multiplier))
                    actions.append({
                        "batch_id": batch.batch_id,
                        "price_multiplier": multiplier,
                        "flash_sale": False,
                        "bundle_with": None,
                    })
                    reasoning_parts.append(
                        f"URGENT {batch.batch_id}: velocity gap, discount to {multiplier}"
                    )
                else:
                    actions.append({
                        "batch_id": batch.batch_id,
                        "price_multiplier": 0.85,
                        "flash_sale": False,
                        "bundle_with": None,
                    })
                    reasoning_parts.append(
                        f"URGENT {batch.batch_id}: velocity adequate, nudge to 0.85"
                    )
                expected_delta += 0.05

            else:
                # WATCH and FRESH: hold full price
                actions.append({
                    "batch_id": batch.batch_id,
                    "price_multiplier": 1.0,
                    "flash_sale": False,
                    "bundle_with": None,
                })

        directive = {"engine": "PRICING", "actions": actions}
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "All batches FRESH/WATCH — hold prices"

        return ExpertDecision(
            tick=tick,
            engine_type=BriefEngineType.PRICING,
            recommended_action=directive,
            reasoning=reasoning,
            expected_reward_delta=expected_delta,
        )

    def _expert_farmer(self, state: SimulatedMarketState, tick: int) -> ExpertDecision:
        """Expert farmer: accept high-viability, counter medium, decline low."""
        actions: list[dict] = []
        reasoning_parts: list[str] = []
        expected_delta = 0.0
        buffer_healthy = state.risk_buffer_balance > RISK_BUFFER_INITIAL_SEED_RS * 0.5

        for offer in state.pending_offers:
            if offer.status != FarmerOfferStatus.PENDING:
                continue

            score = offer.viability_score
            if score is None:
                score = 0.0

            if score >= 0.75:
                actions.append({
                    "offer_id": offer.offer_id,
                    "decision": "ACCEPT",
                    "counter_price": None,
                })
                reasoning_parts.append(
                    f"ACCEPT {offer.offer_id}: viability {score:.2f} >= 0.75"
                )
                expected_delta += 0.20 * score

            elif score >= 0.55 and buffer_healthy:
                counter = round(offer.offered_price_per_kg * 1.10, 2)
                actions.append({
                    "offer_id": offer.offer_id,
                    "decision": "COUNTER",
                    "counter_price": counter,
                })
                reasoning_parts.append(
                    f"COUNTER {offer.offer_id}: viability {score:.2f}, "
                    f"buffer healthy, counter at Rs {counter}/kg"
                )
                expected_delta += 0.10 * score

            else:
                actions.append({
                    "offer_id": offer.offer_id,
                    "decision": "DECLINE",
                    "counter_price": None,
                })
                reason = "low viability" if score < 0.55 else "low buffer"
                reasoning_parts.append(
                    f"DECLINE {offer.offer_id}: {reason} (viability {score:.2f})"
                )

        directive = {"engine": "FARMER", "actions": actions}
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "No pending offers"

        return ExpertDecision(
            tick=tick,
            engine_type=BriefEngineType.FARMER,
            recommended_action=directive,
            reasoning=reasoning,
            expected_reward_delta=expected_delta,
        )

    def _expert_trend(self, state: SimulatedMarketState, tick: int) -> ExpertDecision:
        """Expert trend: approve high-confidence, conservative on medium, decline weak."""
        actions: list[dict] = []
        reasoning_parts: list[str] = []
        expected_delta = 0.0

        for category, signal in state.trend_signals.items():
            if signal.action_taken != TrendAction.PENDING:
                continue
            if not signal.is_actionable(tick):
                continue

            if signal.composite_score >= 80.0:
                actions.append({
                    "category": category,
                    "decision": "APPROVE",
                    "order_quantity_kg": signal.suggested_order_kg,
                })
                reasoning_parts.append(
                    f"APPROVE {category}: score {signal.composite_score:.0f} >= 80 "
                    f"at {signal.suggested_order_kg:.0f}kg"
                )
                expected_delta += 0.25

            elif signal.composite_score >= 65.0 and signal.historical_conversion >= 0.6:
                conservative_qty = round(signal.suggested_order_kg * 0.75, 1)
                actions.append({
                    "category": category,
                    "decision": "APPROVE",
                    "order_quantity_kg": conservative_qty,
                })
                reasoning_parts.append(
                    f"APPROVE {category}: score {signal.composite_score:.0f}, "
                    f"hist_conv {signal.historical_conversion:.2f}, "
                    f"conservative order {conservative_qty:.0f}kg"
                )
                expected_delta += 0.10

            else:
                actions.append({
                    "category": category,
                    "decision": "DECLINE",
                    "order_quantity_kg": None,
                })
                reasoning_parts.append(
                    f"DECLINE {category}: score {signal.composite_score:.0f} too low "
                    f"or hist_conv {signal.historical_conversion:.2f} < 0.6"
                )

        directive = {"engine": "TREND", "actions": actions}
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "No actionable signals"

        return ExpertDecision(
            tick=tick,
            engine_type=BriefEngineType.TREND,
            recommended_action=directive,
            reasoning=reasoning,
            expected_reward_delta=expected_delta,
        )

    # ------------------------------------------------------------------
    # Regret computation
    # ------------------------------------------------------------------

    def compute_regret(
        self,
        agent_brief: dict,
        expert_decision: ExpertDecision,
        actual_reward_delta: float,
    ) -> float:
        """Compute how much the agent's decision cost vs the expert decision.

        regret = (expert.expected_reward_delta - actual_reward_delta) / max_regret
        Clamped to [0.0, 1.0]. Returns 0.0 if decisions are equivalent.
        """
        # Check if decisions are equivalent
        if self._decisions_equivalent(agent_brief, expert_decision):
            return 0.0

        raw_regret = expert_decision.expected_reward_delta - actual_reward_delta

        max_regret = {
            BriefEngineType.PRICING: _MAX_REGRET_PRICING,
            BriefEngineType.FARMER: _MAX_REGRET_FARMER,
            BriefEngineType.TREND: _MAX_REGRET_TREND,
        }.get(expert_decision.engine_type, 1.0)

        if max_regret <= 0:
            return 0.0

        normalised = raw_regret / max_regret
        return max(0.0, min(1.0, normalised))

    def _decisions_equivalent(
        self, agent_brief: dict, expert: ExpertDecision,
    ) -> bool:
        """Check if agent and expert made the same decision."""
        agent_directive = agent_brief.get("directive", {})
        expert_directive = expert.recommended_action

        agent_actions = agent_directive.get("actions", [])
        expert_actions = expert_directive.get("actions", [])

        if len(agent_actions) != len(expert_actions):
            return False

        if expert.engine_type == BriefEngineType.PRICING:
            # Same batch_ids with similar multipliers (within 0.10)
            agent_map = {a.get("batch_id"): a.get("price_multiplier", 1.0) for a in agent_actions}
            for ea in expert_actions:
                bid = ea.get("batch_id")
                if bid not in agent_map:
                    return False
                if abs(agent_map[bid] - ea.get("price_multiplier", 1.0)) > 0.10:
                    return False
            return True

        if expert.engine_type == BriefEngineType.FARMER:
            agent_map = {a.get("offer_id"): a.get("decision", "").upper() for a in agent_actions}
            for ea in expert_actions:
                oid = ea.get("offer_id")
                if oid not in agent_map:
                    return False
                if agent_map[oid] != ea.get("decision", "").upper():
                    return False
            return True

        if expert.engine_type == BriefEngineType.TREND:
            agent_map = {a.get("category"): a.get("decision", "").upper() for a in agent_actions}
            for ea in expert_actions:
                cat = ea.get("category")
                if cat not in agent_map:
                    return False
                if agent_map[cat] != ea.get("decision", "").upper():
                    return False
            return True

        return False

    # ------------------------------------------------------------------
    # Synthetic rejected brief generation
    # ------------------------------------------------------------------

    def generate_synthetic_rejected(
        self,
        chosen_brief: dict,
        prompt: str,
        expert_decision: ExpertDecision | None = None,
        llm_client=None,
    ) -> str:
        """Generate a synthetic lower-quality brief for DPO rejection.

        Mode 1 (template-based, default): Degrades the chosen brief's decisions.
        Mode 2 (LLM-based, 20% chance if llm_client provided): Asks LLM for a
        naive conservative brief. Falls back to Mode 1 on failure.
        """
        # Mode 2: probabilistic LLM-based generation
        if llm_client is not None and self.rng.random() < _LLM_REJECTION_PROBABILITY:
            try:
                result = self._llm_generate_rejected(prompt, llm_client)
                if result:
                    return result
            except Exception:
                logger.debug("LLM synthetic rejection failed, falling back to template", exc_info=True)

        # Mode 1: template-based degradation
        return self._template_generate_rejected(chosen_brief)

    def _template_generate_rejected(self, chosen_brief: dict) -> str:
        """Degrade the chosen brief into a poor-quality rejected brief."""
        engine_type = chosen_brief.get("engine_type")
        if hasattr(engine_type, "value"):
            engine_val = engine_type.value
        else:
            engine_val = str(engine_type).upper()

        situation = chosen_brief.get("situation", "Current state assessed.")

        if engine_val == "PRICING":
            directive = self._degrade_pricing_directive(chosen_brief.get("directive", {}))
        elif engine_val == "FARMER":
            directive = self._degrade_farmer_directive(chosen_brief.get("directive", {}))
        elif engine_val == "TREND":
            directive = self._degrade_trend_directive(chosen_brief.get("directive", {}))
        else:
            directive = chosen_brief.get("directive", {"engine": engine_val, "actions": []})

        directive_json = json.dumps(directive, indent=2)

        return (
            f"SITUATION: {situation}\n"
            f"SIGNAL ANALYSIS: N/A\n"
            f"VIABILITY CHECK: N/A\n"
            f"RECOMMENDATION: Maintaining current strategy. No changes needed at this time.\n"
            f"DIRECTIVE: {directive_json}\n"
            f"CONFIDENCE: LOW"
        )

    def _degrade_pricing_directive(self, directive: dict) -> dict:
        """Replace all price_multipliers with 1.0 (naive hold)."""
        actions = []
        for action in directive.get("actions", []):
            actions.append({
                "batch_id": action.get("batch_id", ""),
                "price_multiplier": 1.0,
                "flash_sale": False,
                "bundle_with": None,
            })
        return {"engine": "PRICING", "actions": actions}

    def _degrade_farmer_directive(self, directive: dict) -> dict:
        """Replace all decisions with DECLINE."""
        actions = []
        for action in directive.get("actions", []):
            actions.append({
                "offer_id": action.get("offer_id", ""),
                "decision": "DECLINE",
                "counter_price": None,
            })
        return {"engine": "FARMER", "actions": actions}

    def _degrade_trend_directive(self, directive: dict) -> dict:
        """Replace all decisions with DECLINE."""
        actions = []
        for action in directive.get("actions", []):
            actions.append({
                "category": action.get("category", ""),
                "decision": "DECLINE",
                "order_quantity_kg": None,
            })
        return {"engine": "TREND", "actions": actions}

    def _llm_generate_rejected(self, prompt: str, llm_client) -> str | None:
        """Ask LLM for a deliberately naive brief."""
        rejection_prefix = (
            "Write a brief that makes the naive, conservative, worst-case decision. "
            "Do not discount urgently expiring items. Decline all farmer offers. "
            "Decline all trend orders.\n\n"
        )
        full_prompt = rejection_prefix + prompt

        # Call LLM — expects llm_client.generate(prompt) -> str
        response = llm_client.generate(full_prompt)
        if response and len(response.strip()) > 50:
            return response.strip()
        return None

    # ------------------------------------------------------------------
    # Episode-level regret analysis
    # ------------------------------------------------------------------

    def analyse_episode_regret(
        self,
        episode_briefs: list[dict],
        episode_states: list[SimulatedMarketState],
    ) -> dict:
        """Post-episode regret analysis.

        For each brief, recompute expert decision and classify regret.
        Returns summary for DPO pair selection.
        """
        regret_scores: list[float] = []
        regret_by_type: dict[str, int] = {}
        high_regret_briefs: list[dict] = []
        max_regret = 0.0
        max_regret_tick = 0

        for i, brief in enumerate(episode_briefs):
            # Use corresponding state if available, else skip
            if i >= len(episode_states):
                continue

            state = episode_states[i]
            tick = brief.get("tick", 0)
            engine_type_str = brief.get("engine_type", "PRICING")

            try:
                engine_type = BriefEngineType(engine_type_str)
            except ValueError:
                continue

            expert = self.compute_expert_decision(state, engine_type, tick)
            actual_reward = brief.get("reward_delta", 0.0)
            regret = self.compute_regret(brief, expert, actual_reward)

            regret_scores.append(regret)

            if regret > max_regret:
                max_regret = regret
                max_regret_tick = tick

            # Classify regret type
            regret_type = self._classify_regret(brief, expert, state)
            if regret_type:
                regret_by_type[regret_type] = regret_by_type.get(regret_type, 0) + 1

            if regret > 0.7:
                high_regret_briefs.append({
                    **brief,
                    "regret_score": round(regret, 4),
                    "regret_type": regret_type,
                    "expert_reasoning": expert.reasoning,
                })

        mean_regret = sum(regret_scores) / len(regret_scores) if regret_scores else 0.0

        return {
            "mean_regret": round(mean_regret, 4),
            "max_regret": round(max_regret, 4),
            "max_regret_tick": max_regret_tick,
            "regret_by_type": regret_by_type,
            "high_regret_briefs": high_regret_briefs,
        }

    def _classify_regret(
        self,
        agent_brief: dict,
        expert: ExpertDecision,
        state: SimulatedMarketState,
    ) -> str | None:
        """Classify the type of regret based on agent vs expert divergence."""
        agent_directive = agent_brief.get("directive", {})
        agent_actions = agent_directive.get("actions", [])

        if expert.engine_type == BriefEngineType.PRICING:
            # Check if agent held price on CRITICAL batches
            critical_ids = {
                b.batch_id for b in state.batches
                if b.status == BatchStatus.ACTIVE and b.urgency == ExpiryUrgency.CRITICAL
            }
            agent_map = {a.get("batch_id"): a.get("price_multiplier", 1.0) for a in agent_actions}
            for cid in critical_ids:
                if agent_map.get(cid, 1.0) >= 0.95:
                    return "PRICING_HELD_ON_CRITICAL"

        elif expert.engine_type == BriefEngineType.FARMER:
            agent_map = {a.get("offer_id"): a.get("decision", "").upper() for a in agent_actions}
            expert_actions = expert.recommended_action.get("actions", [])
            for ea in expert_actions:
                oid = ea.get("offer_id")
                expert_dec = ea.get("decision", "").upper()
                agent_dec = agent_map.get(oid, "")

                if expert_dec == "ACCEPT" and agent_dec == "DECLINE":
                    return "FARMER_MISSED_VIABILITY"
                if expert_dec == "DECLINE" and agent_dec == "ACCEPT":
                    return "FARMER_RECKLESS_ACCEPT"

        elif expert.engine_type == BriefEngineType.TREND:
            agent_map = {a.get("category"): a.get("decision", "").upper() for a in agent_actions}
            expert_actions = expert.recommended_action.get("actions", [])
            for ea in expert_actions:
                cat = ea.get("category")
                expert_dec = ea.get("decision", "").upper()
                agent_dec = agent_map.get(cat, "")

                if expert_dec == "APPROVE" and agent_dec == "DECLINE":
                    return "TREND_MISSED_HIGH_CONFIDENCE"
                if expert_dec == "DECLINE" and agent_dec == "APPROVE":
                    return "TREND_OVERTRADE"

        return None
