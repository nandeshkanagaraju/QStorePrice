"""Converts a validated Operating Brief's DIRECTIVE into typed action objects.

This is the bridge between language and action.
The LLM writes the DIRECTIVE. This file reads it.

No LLM calls. No randomness. Never raises — returns ExecutionResult with warnings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from freshprice_env.constants import (
    ANTIHACK_EARLY_DISCOUNT_HOURS_THRESHOLD,
    ANTIHACK_EARLY_DISCOUNT_PRICE_THRESHOLD,
    ANTIHACK_RECKLESS_ACCEPT_VIABILITY_MAX,
    ANTIHACK_TREND_ORDER_VELOCITY_MULTIPLIER,
    PRICE_MULTIPLIER_MAX,
    PRICE_MULTIPLIER_MIN,
)
from freshprice_env.entities import SimulatedMarketState
from freshprice_env.enums import (
    BatchStatus,
    BriefEngineType,
    FarmerOfferStatus,
)
from freshprice_env.market_state import get_base_demand_velocity


# ---------------------------------------------------------------------------
# Action dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PricingAction:
    batch_id: str
    new_price: float
    flash_sale: bool
    bundle_with: str | None
    was_clamped: bool              # True if price was clamped to floor or ceiling
    was_antihack_blocked: bool     # True if anti-hack guard rejected the price


@dataclass
class FarmerAction:
    offer_id: str
    decision: str                  # ACCEPT | COUNTER | DECLINE
    counter_price: float | None
    was_antihack_blocked: bool


@dataclass
class TrendActionResult:
    category: str
    decision: str                  # APPROVE | DECLINE
    order_quantity_kg: float | None
    was_capped: bool               # True if quantity was capped at max


@dataclass
class ExecutionResult:
    engine_type: BriefEngineType
    pricing_actions: list[PricingAction] = field(default_factory=list)
    farmer_actions: list[FarmerAction] = field(default_factory=list)
    trend_actions: list[TrendActionResult] = field(default_factory=list)
    execution_warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# RuleExecutor
# ---------------------------------------------------------------------------

class RuleExecutor:
    """Converts a validated Operating Brief's DIRECTIVE into typed action objects."""

    @staticmethod
    def execute(
        brief: dict,
        state: SimulatedMarketState,
    ) -> ExecutionResult:
        """Parse the directive and produce typed action objects.

        Never raises — returns ExecutionResult with warnings on any issue.
        """
        engine_type = brief["engine_type"]
        directive = brief.get("directive", {})

        if engine_type == BriefEngineType.PRICING:
            actions, warnings = RuleExecutor._execute_pricing(directive, state)
            return ExecutionResult(
                engine_type=engine_type,
                pricing_actions=actions,
                execution_warnings=warnings,
            )

        if engine_type == BriefEngineType.FARMER:
            actions, warnings = RuleExecutor._execute_farmer(directive, state)
            return ExecutionResult(
                engine_type=engine_type,
                farmer_actions=actions,
                execution_warnings=warnings,
            )

        if engine_type == BriefEngineType.TREND:
            actions, warnings = RuleExecutor._execute_trend(directive, state)
            return ExecutionResult(
                engine_type=engine_type,
                trend_actions=actions,
                execution_warnings=warnings,
            )

        return ExecutionResult(
            engine_type=engine_type,
            execution_warnings=[f"Unknown engine type: {engine_type}"],
        )

    # ------------------------------------------------------------------
    # PRICING execution
    # ------------------------------------------------------------------

    @staticmethod
    def _execute_pricing(
        directive: dict, state: SimulatedMarketState,
    ) -> tuple[list[PricingAction], list[str]]:
        actions: list[PricingAction] = []
        warnings: list[str] = []

        batch_map = {
            b.batch_id: b for b in state.batches if b.status == BatchStatus.ACTIVE
        }

        for action in directive.get("actions", []):
            bid = action.get("batch_id")
            if bid is None or bid not in batch_map:
                warnings.append(f"Skipping unknown batch_id: {bid}")
                continue

            batch = batch_map[bid]
            pm = action.get("price_multiplier", 1.0)
            flash_sale = action.get("flash_sale", False)
            bundle_with = action.get("bundle_with")

            # Clamp multiplier
            was_clamped = False
            if pm < PRICE_MULTIPLIER_MIN:
                pm = PRICE_MULTIPLIER_MIN
                was_clamped = True
            elif pm > PRICE_MULTIPLIER_MAX:
                pm = PRICE_MULTIPLIER_MAX
                was_clamped = True

            # Compute new price
            new_price = batch.original_price * pm

            # Floor price enforcement
            if new_price < batch.floor_price:
                new_price = batch.floor_price
                was_clamped = True

            # Anti-hack check
            was_antihack_blocked = False
            if (
                pm < ANTIHACK_EARLY_DISCOUNT_PRICE_THRESHOLD
                and batch.hours_to_expiry > ANTIHACK_EARLY_DISCOUNT_HOURS_THRESHOLD
            ):
                was_antihack_blocked = True
                new_price = batch.current_price  # Keep existing price
                warnings.append(
                    f"Anti-hack blocked for {bid}: multiplier {pm:.2f} "
                    f"with {batch.hours_to_expiry:.1f}hrs remaining"
                )

            actions.append(PricingAction(
                batch_id=bid,
                new_price=round(new_price, 2),
                flash_sale=flash_sale,
                bundle_with=bundle_with,
                was_clamped=was_clamped,
                was_antihack_blocked=was_antihack_blocked,
            ))

        return actions, warnings

    # ------------------------------------------------------------------
    # FARMER execution
    # ------------------------------------------------------------------

    @staticmethod
    def _execute_farmer(
        directive: dict, state: SimulatedMarketState,
    ) -> tuple[list[FarmerAction], list[str]]:
        actions: list[FarmerAction] = []
        warnings: list[str] = []

        offer_map = {
            o.offer_id: o for o in state.pending_offers
            if o.status == FarmerOfferStatus.PENDING
        }

        for action in directive.get("actions", []):
            oid = action.get("offer_id")
            if oid is None or oid not in offer_map:
                warnings.append(f"Skipping unknown offer_id: {oid}")
                continue

            offer = offer_map[oid]
            decision = action.get("decision", "").upper()
            counter_price = action.get("counter_price")

            was_antihack_blocked = False

            # Anti-hack check for ACCEPT
            if decision == "ACCEPT":
                if (
                    offer.viability_score is not None
                    and offer.viability_score < ANTIHACK_RECKLESS_ACCEPT_VIABILITY_MAX
                ):
                    was_antihack_blocked = True
                    decision = "DECLINE"
                    warnings.append(
                        f"Reckless accept blocked for {oid} — "
                        f"viability {offer.viability_score:.2f} below threshold"
                    )

            if decision == "COUNTER" and counter_price is not None:
                counter_price = float(counter_price)

            actions.append(FarmerAction(
                offer_id=oid,
                decision=decision,
                counter_price=counter_price if decision == "COUNTER" else None,
                was_antihack_blocked=was_antihack_blocked,
            ))

        return actions, warnings

    # ------------------------------------------------------------------
    # TREND execution
    # ------------------------------------------------------------------

    @staticmethod
    def _execute_trend(
        directive: dict, state: SimulatedMarketState,
    ) -> tuple[list[TrendActionResult], list[str]]:
        actions: list[TrendActionResult] = []
        warnings: list[str] = []

        for action in directive.get("actions", []):
            category = action.get("category")
            decision = action.get("decision", "").upper()

            if category is None or category not in state.trend_signals:
                warnings.append(f"Skipping unknown category: {category}")
                continue

            order_qty = action.get("order_quantity_kg")
            was_capped = False

            if decision == "APPROVE" and order_qty is not None:
                order_qty = float(order_qty)

                # Compute weekly velocity cap
                base_hourly = get_base_demand_velocity(category, hour_of_day=12, day_of_week=3)
                weekly_velocity = base_hourly * 24.0 * 7.0
                cap = weekly_velocity * ANTIHACK_TREND_ORDER_VELOCITY_MULTIPLIER

                if order_qty > cap:
                    warnings.append(
                        f"Trend order for {category} capped: "
                        f"{order_qty:.0f}kg → {cap:.0f}kg"
                    )
                    order_qty = cap
                    was_capped = True

            actions.append(TrendActionResult(
                category=category,
                decision=decision,
                order_quantity_kg=order_qty if decision == "APPROVE" else None,
                was_capped=was_capped,
            ))

        return actions, warnings
