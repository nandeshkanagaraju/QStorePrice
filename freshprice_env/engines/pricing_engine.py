"""Engine 1 — Dynamic Pricing simulation and r1 reward component.

Runs every tick. Two responsibilities:
  1. Compute new prices for active batches based on directive + demand
  2. Compute r1 (pricing reward component) for the tick
"""

from __future__ import annotations

import logging
import random
from dataclasses import replace

from freshprice_env.constants import (
    ANTIHACK_EARLY_DISCOUNT_HOURS_THRESHOLD,
    ANTIHACK_EARLY_DISCOUNT_PRICE_THRESHOLD,
    PRICE_MULTIPLIER_MIN,
    PRICE_MULTIPLIER_MAX,
    R1_ANTIHACK_BELOW_FLOOR,
    R1_ANTIHACK_EARLY_DISCOUNT,
    R1_EXPIRED_UNIT_PENALTY,
    R1_NEAR_EXPIRY_HOURS,
    R1_URGENCY_CLEARANCE_BONUS,
    TICKS_PER_DAY,
)
from freshprice_env.entities import SimulatedBatch, SimulatedMarketState
from freshprice_env.enums import BatchStatus
from freshprice_env.market_state import get_base_demand_velocity

logger = logging.getLogger(__name__)

# 15 minutes expressed in hours — used only in pricing sales calculations.
TICK_DURATION_HOURS: float = 0.25


class PricingEngine:
    """Simulates Engine 1 — Dynamic Pricing."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        # category → tick when last flash sale was used
        # Enforces max 1 flash sale per category per day (96 ticks)
        self._flash_sale_used: dict[str, int] = {}
        # Batch IDs that have already been counted in the at-risk cost accumulator.
        # Prevents double-counting across ticks — a batch's cost is added once,
        # at the moment it first becomes URGENT or CRITICAL.
        self._at_risk_seen: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tick(
        self,
        state: SimulatedMarketState,
        directive: dict | None,
    ) -> tuple[SimulatedMarketState, float]:
        """Run one pricing tick.

        Args:
            state: Current market state.
            directive: PRICING directive from the last Operating Brief, or None
                       if no brief has fired yet this episode.

        Returns:
            (updated_state, r1)
        """
        # Snapshot max possible revenue before anything changes this tick
        max_possible_revenue = sum(
            b.quantity_remaining * b.original_price
            for b in state.batches
            if b.status == BatchStatus.ACTIVE
        )

        # Track anti-hack violations this tick (split by type for r1 penalty math)
        early_discount_violations = 0
        below_floor_violations = 0

        # 1. Apply directive prices
        if directive is not None:
            state, ed_v, bf_v = self._apply_directive(state, directive)
            early_discount_violations += ed_v
            below_floor_violations += bf_v

        # 2. Compute sales
        state, revenue_this_tick, near_expiry_units_sold = self._compute_sales(state)

        # 3. Age batches
        state, newly_expired = self._age_batches(state)

        # 4. Update at-risk accumulators
        state = self._update_at_risk_accumulators(state)

        # 5. Compute r1
        r1 = self._compute_r1(
            revenue_this_tick=revenue_this_tick,
            max_possible_revenue=max_possible_revenue,
            expired_batches=newly_expired,
            early_discount_violations=early_discount_violations,
            below_floor_violations=below_floor_violations,
            near_expiry_units_sold=near_expiry_units_sold,
        )

        return state, r1

    # ------------------------------------------------------------------
    # Step 1: Apply directive
    # ------------------------------------------------------------------

    def _apply_directive(
        self,
        state: SimulatedMarketState,
        directive: dict,
    ) -> tuple[SimulatedMarketState, int, int]:
        """Apply PRICING directive to batch prices.

        Returns:
            (updated_state, early_discount_violation_count, below_floor_violation_count)
        """
        actions = directive.get("actions", [])
        if not actions:
            return state, 0, 0

        # Index batches by id for fast lookup
        batch_map: dict[str, int] = {
            b.batch_id: i for i, b in enumerate(state.batches)
        }

        updated_batches = list(state.batches)
        early_discount_violations = 0
        below_floor_violations = 0

        for action in actions:
            batch_id = action.get("batch_id")
            if batch_id is None or batch_id not in batch_map:
                continue

            idx = batch_map[batch_id]
            batch = updated_batches[idx]

            if batch.status != BatchStatus.ACTIVE:
                continue

            price_multiplier = action.get("price_multiplier", 1.0)
            flash_sale = action.get("flash_sale", False)

            # Clamp multiplier to valid range
            price_multiplier = max(PRICE_MULTIPLIER_MIN, min(PRICE_MULTIPLIER_MAX, price_multiplier))

            # Anti-hack guard: early deep discount
            if (
                price_multiplier < ANTIHACK_EARLY_DISCOUNT_PRICE_THRESHOLD
                and batch.hours_to_expiry > ANTIHACK_EARLY_DISCOUNT_HOURS_THRESHOLD
            ):
                logger.warning(
                    "Anti-hack: early deep discount blocked for %s "
                    "(multiplier=%.2f, hours_remaining=%.1f)",
                    batch_id, price_multiplier, batch.hours_to_expiry,
                )
                early_discount_violations += 1
                # Do NOT apply the price — keep current price
                continue

            # Flash sale handling
            if flash_sale:
                last_used_tick = self._flash_sale_used.get(batch.category)
                if last_used_tick is not None and (state.tick - last_used_tick) < TICKS_PER_DAY:
                    # Category already used flash sale today — silently ignore
                    logger.info(
                        "Flash sale ignored for %s: category %s already used at tick %d",
                        batch_id, batch.category, last_used_tick,
                    )
                else:
                    # Apply flash sale price
                    flash_price = batch.floor_price + (batch.original_price * 0.10)
                    new_price = max(batch.floor_price, flash_price)
                    updated_batches[idx] = replace(batch, current_price=round(new_price, 2))
                    self._flash_sale_used[batch.category] = state.tick
                continue

            # Normal price update
            new_price = batch.original_price * price_multiplier

            # Anti-hack guard: below-floor price (SDD Section 06 BELOW_FLOOR_PRICE).
            # 1-cent tolerance avoids spurious flags when a directive round-trips
            # through rule_executor — float-point math (floor / original × original)
            # can land 1 ULP under floor without any actual violation.
            if new_price < batch.floor_price - 0.01:
                logger.warning(
                    "Anti-hack: below-floor price for %s "
                    "(proposed=%.2f, floor=%.2f) — clamping",
                    batch_id, new_price, batch.floor_price,
                )
                below_floor_violations += 1
                new_price = batch.floor_price
            elif new_price < batch.floor_price:
                # Within tolerance — clamp silently
                new_price = batch.floor_price

            updated_batches[idx] = replace(batch, current_price=round(new_price, 2))

        state.batches = updated_batches
        return state, early_discount_violations, below_floor_violations

    # ------------------------------------------------------------------
    # Step 2: Compute sales
    # ------------------------------------------------------------------

    def _compute_sales(
        self,
        state: SimulatedMarketState,
    ) -> tuple[SimulatedMarketState, float, int]:
        """Simulate sales for this tick.

        Returns:
            (updated_state, revenue_this_tick, near_expiry_units_sold)
        """
        revenue_this_tick = 0.0
        near_expiry_units_sold = 0
        updated_batches = list(state.batches)
        updated_velocity = dict(state.sales_velocity)

        for i, batch in enumerate(updated_batches):
            if batch.status != BatchStatus.ACTIVE or batch.quantity_remaining <= 0:
                continue

            # Base demand
            base_velocity = get_base_demand_velocity(
                batch.category, state.hour_of_day, state.day_of_week,
            )

            # Price elasticity: higher discount → more demand
            if batch.current_price > 0:
                price_elasticity_factor = (batch.original_price / batch.current_price) ** 1.5
            else:
                price_elasticity_factor = 1.0

            adjusted_velocity = base_velocity * price_elasticity_factor

            # Demand noise ±15%
            noise = self.rng.uniform(0.85, 1.15)
            effective_velocity = adjusted_velocity * noise

            # Units sold this tick — probabilistic rounding so fractional units
            # don't truncate to zero. Without this, low-velocity categories
            # (anything with base_velocity × time_mult × 0.25 < 1) would never
            # sell at full price and the agent could not learn pricing dynamics.
            expected_units = effective_velocity * TICK_DURATION_HOURS
            whole = int(expected_units)
            frac = expected_units - whole
            if self.rng.random() < frac:
                whole += 1
            units_sold = min(batch.quantity_remaining, whole)

            if units_sold > 0:
                revenue = units_sold * batch.current_price
                revenue_this_tick += revenue

                # Track near-expiry clearance for bonus
                if batch.hours_to_expiry <= R1_NEAR_EXPIRY_HOURS:
                    near_expiry_units_sold += units_sold

                # Update batch
                new_remaining = batch.quantity_remaining - units_sold
                new_status = BatchStatus.CLEARED if new_remaining == 0 else batch.status
                updated_batches[i] = replace(
                    batch,
                    quantity_remaining=new_remaining,
                    status=new_status,
                )

                # Update at-risk revenue accumulator
                if batch.is_at_risk:
                    state.revenue_recovered_accumulator += revenue

            # Record effective velocity
            updated_velocity[batch.batch_id] = round(effective_velocity, 3)

        state.batches = updated_batches
        state.sales_velocity = updated_velocity
        return state, revenue_this_tick, near_expiry_units_sold

    # ------------------------------------------------------------------
    # Step 3: Age batches
    # ------------------------------------------------------------------

    def _age_batches(
        self,
        state: SimulatedMarketState,
    ) -> tuple[SimulatedMarketState, list[SimulatedBatch]]:
        """Decrement hours_to_expiry by TICK_DURATION_HOURS for all active batches.

        Returns:
            (updated_state, newly_expired_batches)
        """
        updated_batches = list(state.batches)
        newly_expired: list[SimulatedBatch] = []

        for i, batch in enumerate(updated_batches):
            if batch.status != BatchStatus.ACTIVE:
                continue

            old_hours = batch.hours_to_expiry
            new_hours = max(0.0, old_hours - TICK_DURATION_HOURS)

            if old_hours > 0.0 and new_hours <= 0.0:
                # Batch just expired
                expired_batch = replace(
                    batch,
                    hours_to_expiry=0.0,
                    status=BatchStatus.EXPIRED,
                )
                updated_batches[i] = expired_batch
                newly_expired.append(expired_batch)
            else:
                updated_batches[i] = replace(batch, hours_to_expiry=round(new_hours, 2))

        state.batches = updated_batches
        return state, newly_expired

    # ------------------------------------------------------------------
    # Step 4: Update at-risk accumulators
    # ------------------------------------------------------------------

    def _update_at_risk_accumulators(
        self,
        state: SimulatedMarketState,
    ) -> SimulatedMarketState:
        """Track cost of at-risk inventory for WRR denominator.

        When a batch transitions into URGENT or CRITICAL for the first time
        this episode, its remaining cost (unit_cost × quantity_remaining) is
        added to state.at_risk_cost_accumulator. This happens exactly once
        per batch — self._at_risk_seen prevents double-counting.

        The cost is locked in at the moment the batch becomes at-risk, even
        if it is later cleared. This prevents gaming WRR by pre-emptively
        clearing fresh stock before it reaches at-risk thresholds.
        """
        for batch in state.batches:
            if batch.status != BatchStatus.ACTIVE:
                continue
            if not batch.is_at_risk:
                continue
            if batch.batch_id in self._at_risk_seen:
                continue
            # First time this batch is at-risk — lock in its cost
            self._at_risk_seen.add(batch.batch_id)
            at_risk_cost = batch.unit_cost * batch.quantity_remaining
            state.at_risk_cost_accumulator += at_risk_cost
        return state

    # ------------------------------------------------------------------
    # Step 5: Compute r1
    # ------------------------------------------------------------------

    def _compute_r1(
        self,
        revenue_this_tick: float,
        max_possible_revenue: float,
        expired_batches: list[SimulatedBatch],
        early_discount_violations: int,
        below_floor_violations: int,
        near_expiry_units_sold: int,
    ) -> float:
        """Compute the r1 pricing reward component.

        r1 = revenue_this_tick / max_possible_revenue  (base ratio)
           + R1_URGENCY_CLEARANCE_BONUS per near-expiry unit sold
           - R1_EXPIRED_UNIT_PENALTY per expired unit
           - R1_ANTIHACK_EARLY_DISCOUNT per early-discount violation
           - R1_ANTIHACK_BELOW_FLOOR per below-floor violation

        r1 is NOT clamped — negative values are valid learning signals.
        """
        # Base ratio
        if max_possible_revenue > 0:
            r1 = revenue_this_tick / max_possible_revenue
        else:
            r1 = 0.0

        # Bonus: near-expiry clearance
        r1 += R1_URGENCY_CLEARANCE_BONUS * near_expiry_units_sold

        # Penalty: expired units
        total_expired_units = sum(b.quantity_remaining for b in expired_batches)
        r1 -= R1_EXPIRED_UNIT_PENALTY * total_expired_units

        # Penalty: anti-hack violations (split by type per SDD Section 06)
        r1 -= R1_ANTIHACK_EARLY_DISCOUNT * early_discount_violations
        r1 -= R1_ANTIHACK_BELOW_FLOOR * below_floor_violations

        return r1
