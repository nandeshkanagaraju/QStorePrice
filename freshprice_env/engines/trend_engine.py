"""Engine 3 — Social Trend Engine simulation and r3 reward component.

Does NOT run every tick. Activates when:
  1. A new trend signal is injected by the environment
  2. The agent's Operating Brief contains a TREND directive

apply_trend_demand_boost() IS called every tick — it adjusts sales velocity
for categories with active approved trends.
"""

from __future__ import annotations

import logging
import random
from dataclasses import replace

from freshprice_env.constants import (
    ANTIHACK_TREND_ORDER_VELOCITY_MULTIPLIER,
    R3_OVERTRADE_DISCOUNT_THRESHOLD,
    R3_OVERTRADE_PENALTY,
    R3_PERFECT_TIMING_BONUS,
    TREND_COOLDOWN_HRS,
    TREND_SCORE_THRESHOLD,
)
from freshprice_env.entities import (
    SimulatedBatch,
    SimulatedMarketState,
    SimulatedTrendSignal,
)
from freshprice_env.enums import (
    BatchStatus,
    BatchType,
    SignalSource,
    TrendAction,
)
from freshprice_env.market_state import (
    _BASE_VELOCITY,
    _CATEGORY_SPECS,
    get_base_demand_velocity,
)

logger = logging.getLogger(__name__)

# Ticks per hour at 15-min resolution
_TICKS_PER_HOUR: int = 4

# Wholesale discount for trend restock (30% below retail)
_TREND_WHOLESALE_DISCOUNT: float = 0.70

# Trend restock arrives slightly less fresh than direct stock
_TREND_SHELF_LIFE_FACTOR: float = 0.90


class TrendEngine:
    """Simulates Engine 3 — Social Trend Engine."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        # batch_id → {category, predicted_uplift, order_qty, signal_composite}
        # Tracks trend-sourced batches to measure actual vs predicted uplift
        self._active_trend_batches: dict[str, dict] = {}
        # category → tick when last order was placed
        # Enforces TREND_COOLDOWN_HRS × 4 ticks between orders per category
        self._category_cooldowns: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Per-tick demand boost
    # ------------------------------------------------------------------

    def apply_trend_demand_boost(
        self,
        state: SimulatedMarketState,
    ) -> SimulatedMarketState:
        """Write per-category demand multipliers for approved trend signals.

        Called every tick by the environment BEFORE pricing_engine.tick().
        Stores boosts in state.category_demand_boosts which PricingEngine
        reads during _compute_sales — avoids corrupting the reported
        sales_velocity telemetry dict.
        """
        state.category_demand_boosts.clear()
        for category, signal in state.trend_signals.items():
            if signal.action_taken != TrendAction.APPROVED:
                continue
            # Boost factor: 1.0 at threshold, scales up linearly with score
            state.category_demand_boosts[category] = (
                1.0 + (signal.composite_score - TREND_SCORE_THRESHOLD) / 100.0
            )
        return state

    # ------------------------------------------------------------------
    # Directive processing
    # ------------------------------------------------------------------

    def process_directive(
        self,
        state: SimulatedMarketState,
        directive: dict,
        current_tick: int,
    ) -> tuple[SimulatedMarketState, float]:
        """Execute a TREND directive from an Operating Brief.

        Returns:
            (updated_state, r3)
        """
        actions = directive.get("actions", [])
        if not actions:
            return state, 0.0

        approved_actions: list[dict] = []
        antihack_caps_applied = 0
        new_batches: list[SimulatedBatch] = []

        for action in actions:
            category = action.get("category")
            decision = action.get("decision", "").upper()

            if category is None:
                continue

            if decision == "APPROVE":
                result = self._process_approve(
                    state, action, category, current_tick,
                )
                if result is None:
                    continue
                batch, was_capped, signal_composite = result
                new_batches.append(batch)
                approved_actions.append({
                    "category": category,
                    "composite_score": signal_composite,
                })
                if was_capped:
                    antihack_caps_applied += 1

            elif decision == "DECLINE":
                signal = state.trend_signals.get(category)
                if signal is not None and signal.action_taken == TrendAction.PENDING:
                    state.trend_signals[category] = replace(
                        signal, action_taken=TrendAction.DECLINED,
                    )

        # Add new batches to state
        state.batches = list(state.batches) + new_batches

        r3 = self._compute_r3(approved_actions, antihack_caps_applied)
        return state, r3

    def _process_approve(
        self,
        state: SimulatedMarketState,
        action: dict,
        category: str,
        current_tick: int,
    ) -> tuple[SimulatedBatch, bool, float] | None:
        """Process an APPROVE action. Returns (batch, was_capped, composite) or None."""
        cooldown_ticks = int(TREND_COOLDOWN_HRS * _TICKS_PER_HOUR)

        # Check cooldown
        last_order_tick = self._category_cooldowns.get(category)
        if last_order_tick is not None and (current_tick - last_order_tick) < cooldown_ticks:
            logger.info(
                "Trend order for %s ignored — cooldown active (last order tick %d, current %d)",
                category, last_order_tick, current_tick,
            )
            return None

        # Get signal
        signal = state.trend_signals.get(category)
        if signal is None or not signal.is_actionable(current_tick):
            return None

        order_qty = action.get("order_quantity_kg")
        if order_qty is None or order_qty <= 0:
            return None
        order_qty = float(order_qty)

        # Anti-hack cap: max 2x avg weekly velocity
        # Use midday Wednesday (hour=12, day=3) as neutral baseline
        base_hourly = get_base_demand_velocity(category, hour_of_day=12, day_of_week=3)
        avg_weekly_velocity = base_hourly * 24.0 * 7.0
        cap = avg_weekly_velocity * ANTIHACK_TREND_ORDER_VELOCITY_MULTIPLIER

        was_capped = False
        if order_qty > cap:
            logger.warning(
                "Trend order for %s capped: requested %.1f kg, cap %.1f kg",
                category, order_qty, cap,
            )
            order_qty = cap
            was_capped = True

        # Predicted demand uplift
        base_weekly_sales = avg_weekly_velocity
        predicted_uplift_pct = (signal.composite_score - TREND_SCORE_THRESHOLD) / 100.0
        predicted_additional_units = base_weekly_sales * predicted_uplift_pct

        # Create trend restock batch
        avg_market_price = self._avg_category_price(category, state)
        shelf_life_hrs = self._standard_shelf_life(category) * _TREND_SHELF_LIFE_FACTOR

        batch_id = f"trend_{category}_{current_tick}"
        batch = SimulatedBatch(
            batch_id=batch_id,
            sku_id=f"sku_{category}_trend",
            store_id="store_001",
            category=category,
            quantity_remaining=int(order_qty),
            unit_cost=round(avg_market_price * _TREND_WHOLESALE_DISCOUNT, 2),
            current_price=round(avg_market_price, 2),
            original_price=round(avg_market_price, 2),
            hours_to_expiry=round(shelf_life_hrs, 1),
            batch_type=BatchType.TREND_RESTOCK,
            status=BatchStatus.ACTIVE,
        )

        # Register for outcome tracking
        self._active_trend_batches[batch_id] = {
            "category": category,
            "predicted_uplift": predicted_additional_units,
            "order_qty": int(order_qty),
            "signal_composite": signal.composite_score,
        }

        # Record cooldown and update signal
        self._category_cooldowns[category] = current_tick
        state.trend_signals[category] = replace(
            signal, action_taken=TrendAction.APPROVED,
        )

        return batch, was_capped, signal.composite_score

    # ------------------------------------------------------------------
    # Outcome resolution (called every tick)
    # ------------------------------------------------------------------

    def resolve_trend_outcomes(
        self,
        state: SimulatedMarketState,
    ) -> float:
        """Check trend-sourced batches for completion. Compute r3 delta.

        Called every tick by the environment after pricing tick.
        Returns r3_delta as float.
        """
        r3_delta = 0.0
        resolved_ids: list[str] = []

        for batch in state.batches:
            if batch.batch_id not in self._active_trend_batches:
                continue
            if batch.status not in (BatchStatus.CLEARED, BatchStatus.EXPIRED):
                continue

            tracking = self._active_trend_batches[batch.batch_id]
            original_qty = tracking["order_qty"]
            predicted_uplift = tracking["predicted_uplift"]

            units_sold = original_qty - batch.quantity_remaining

            # Determine discount required to clear
            if batch.original_price > 0:
                avg_discount = (batch.original_price - batch.current_price) / batch.original_price
            else:
                avg_discount = 0.0

            # Classify outcome
            if predicted_uplift > 0:
                sell_through = units_sold / original_qty if original_qty > 0 else 0.0
                # PERFECT: good sell-through and sold near full price
                if sell_through >= 0.85 and avg_discount <= 0.10:
                    r3_delta += R3_PERFECT_TIMING_BONUS
                # OVERTRADE: required > 40% discount to clear
                elif avg_discount > R3_OVERTRADE_DISCOUNT_THRESHOLD:
                    r3_delta -= R3_OVERTRADE_PENALTY
                # NORMAL: no additional delta
            else:
                # No predicted uplift — shouldn't happen with valid signals
                if avg_discount > R3_OVERTRADE_DISCOUNT_THRESHOLD:
                    r3_delta -= R3_OVERTRADE_PENALTY

            resolved_ids.append(batch.batch_id)

        for bid in resolved_ids:
            del self._active_trend_batches[bid]

        return r3_delta

    # ------------------------------------------------------------------
    # r3 computation (immediate signal at directive time)
    # ------------------------------------------------------------------

    def _compute_r3(
        self,
        approved_actions: list[dict],
        antihack_caps_applied: int,
    ) -> float:
        """Compute immediate r3 at directive execution time.

        Final r3 comes from resolve_trend_outcomes when batch clears.
        This is the immediate signal for the agent.
        """
        if not approved_actions and antihack_caps_applied == 0:
            return 0.0

        r3 = 0.0

        # Bonus for acting on high-confidence signals
        for action in approved_actions:
            if action["composite_score"] >= 80.0:
                r3 += 0.10

        # Mild penalty for over-ordering (order was capped)
        r3 -= 0.10 * antihack_caps_applied

        return r3

    # ------------------------------------------------------------------
    # Signal injection
    # ------------------------------------------------------------------

    def inject_trend_signal(
        self,
        state: SimulatedMarketState,
        category: str,
        composite_score: float,
        signal_source: SignalSource,
        suggested_order_kg: float,
        current_tick: int,
        factor_scores: dict[str, float],
    ) -> SimulatedMarketState:
        """Inject a new trend signal into the market state.

        Called by the environment at scheduled injection ticks.
        If a signal already exists for this category and is still PENDING:
        overwrite it.
        """
        signal = SimulatedTrendSignal(
            category=category,
            composite_score=composite_score,
            signal_source=signal_source,
            detected_at_tick=current_tick,
            action_taken=TrendAction.PENDING,
            suggested_order_kg=suggested_order_kg,
            recipe_simplicity=factor_scores.get("recipe_simplicity", 0.0),
            ingredient_rarity=factor_scores.get("ingredient_rarity", 0.0),
            view_velocity=factor_scores.get("view_velocity", 0.0),
            local_relevance=factor_scores.get("local_relevance", 0.0),
            historical_conversion=factor_scores.get("historical_conversion", 0.0),
        )

        # Overwrite if existing signal is still PENDING, otherwise always set
        existing = state.trend_signals.get(category)
        if existing is None or existing.action_taken == TrendAction.PENDING:
            state.trend_signals[category] = signal

        return state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _avg_category_price(
        self, category: str, state: SimulatedMarketState,
    ) -> float:
        """Average current_price of active batches in the category."""
        prices = [
            b.current_price
            for b in state.batches
            if b.category == category and b.status == BatchStatus.ACTIVE
        ]
        if prices:
            return sum(prices) / len(prices)
        # Fallback: any batch in this category
        all_prices = [b.original_price for b in state.batches if b.category == category]
        if all_prices:
            return (sum(all_prices) / len(all_prices)) * 0.80
        return 50.0

    def _standard_shelf_life(self, category: str) -> float:
        """Return the midpoint of the standard shelf life range for a category."""
        spec = _CATEGORY_SPECS.get(category)
        if spec is not None:
            shelf_min, shelf_max = spec[6], spec[7]
            return (shelf_min + shelf_max) / 2.0
        # Fallback
        return 48.0
