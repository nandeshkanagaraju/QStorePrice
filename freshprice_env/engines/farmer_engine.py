"""Engine 2 — Farmer Offer Engine simulation and r2 reward component.

Does NOT run every tick. Activates when:
  1. A new farmer offer arrives (env injects into state)
  2. The agent's Operating Brief contains a FARMER directive

Also resolves outcomes for farmer-sourced batches every tick.
"""

from __future__ import annotations

import logging
import random
from dataclasses import replace

from freshprice_env.constants import (
    ANTIHACK_RECKLESS_ACCEPT_VIABILITY_MAX,
    FARMER_OPS_COST_PER_KG,
    R2_CLEARED_BATCH_BONUS,
    R2_MISSED_OPPORTUNITY_PENALTY,
    R2_MISSED_OPPORTUNITY_VIABILITY_THRESHOLD,
    R2_RECKLESS_ACCEPT_PENALTY,
    RISK_BUFFER_PROFIT_CONTRIBUTION_PCT,
    VIABILITY_SHELF_LIFE_SAFETY_FACTOR,
)
from freshprice_env.entities import (
    SimulatedBatch,
    SimulatedFarmerOffer,
    SimulatedMarketState,
)
from freshprice_env.enums import (
    BatchStatus,
    BatchType,
    ExpiryUrgency,
    FarmerOfferStatus,
)
from freshprice_env.market_state import get_base_demand_velocity, _BASE_VELOCITY

logger = logging.getLogger(__name__)


class FarmerEngine:
    """Simulates Engine 2 — Farmer Offer Engine."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        # batch_id → {cost, quantity_original, offer_id}
        # Tracks farmer-sourced batches for outcome calculation
        self._active_batch_outcomes: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Viability scoring
    # ------------------------------------------------------------------

    def score_offer(
        self,
        offer: SimulatedFarmerOffer,
        state: SimulatedMarketState,
    ) -> SimulatedFarmerOffer:
        """Run the 5-factor viability check, return offer with viability_score set.

        Called by the environment when a new offer arrives, before the brief fires.
        The agent sees the viability_score in the state when writing its brief.
        """
        score_1 = self._factor_shelf_life(offer, state)
        score_2 = self._factor_inventory_conflict(offer, state)
        score_3 = self._factor_break_even(offer, state)
        score_4 = self._factor_worst_case_pl(offer, state)
        score_5 = self._factor_demand_timing(offer, state)

        composite = (score_1 + score_2 + score_3 + score_4 + score_5) / 5.0
        return replace(offer, viability_score=round(composite, 4))

    def _factor_shelf_life(
        self, offer: SimulatedFarmerOffer, state: SimulatedMarketState,
    ) -> float:
        """Factor 1: Can we sell the quantity within shelf life?"""
        avg_velocity = get_base_demand_velocity(
            offer.product_category, state.hour_of_day, state.day_of_week,
        )
        if avg_velocity <= 0:
            return 0.0
        hours_needed = offer.quantity_kg / avg_velocity
        safe_hours = hours_needed * VIABILITY_SHELF_LIFE_SAFETY_FACTOR
        if safe_hours <= 0:
            return 1.0
        ratio = offer.seller_shelf_life_hrs / safe_hours
        return min(1.0, ratio)

    def _factor_inventory_conflict(
        self, offer: SimulatedFarmerOffer, state: SimulatedMarketState,
    ) -> float:
        """Factor 2: Do we already have at-risk stock in this category?"""
        for batch in state.batches:
            if (
                batch.category == offer.product_category
                and batch.urgency in (ExpiryUrgency.URGENT, ExpiryUrgency.CRITICAL)
                and batch.status == BatchStatus.ACTIVE
            ):
                return 0.0
        return 1.0

    def _factor_break_even(
        self, offer: SimulatedFarmerOffer, state: SimulatedMarketState,
    ) -> float:
        """Factor 3: Can we sell above break-even?"""
        break_even_price = offer.offered_price_per_kg + FARMER_OPS_COST_PER_KG
        avg_market_price = self._avg_category_price(offer.product_category, state)
        if break_even_price <= 0:
            return 1.0
        ratio = avg_market_price / break_even_price
        # 0 at break-even, 1.0 at 50% above break-even
        return max(0.0, min(1.0, (ratio - 1.0) / 0.5))

    def _factor_worst_case_pl(
        self, offer: SimulatedFarmerOffer, state: SimulatedMarketState,
    ) -> float:
        """Factor 4: Worst-case P&L — can we survive at 60% sell-through?"""
        break_even_price = offer.offered_price_per_kg + FARMER_OPS_COST_PER_KG
        worst_case_revenue = (offer.quantity_kg * 0.60) * (break_even_price * 1.10)
        worst_case_cost = offer.quantity_kg * offer.offered_price_per_kg

        if worst_case_revenue >= worst_case_cost:
            return 1.0
        if worst_case_revenue >= worst_case_cost * 0.85:
            return 0.5  # FLAG range
        return 0.0

    def _factor_demand_timing(
        self, offer: SimulatedFarmerOffer, state: SimulatedMarketState,
    ) -> float:
        """Factor 5: Is current demand strong for this category?"""
        base_flat = _BASE_VELOCITY.get(offer.product_category, 1.0)
        if base_flat <= 0:
            return 0.0
        current_vel = get_base_demand_velocity(
            offer.product_category, state.hour_of_day, state.day_of_week,
        )
        demand_index = current_vel / base_flat
        return min(1.0, demand_index)

    # ------------------------------------------------------------------
    # Directive processing
    # ------------------------------------------------------------------

    def process_directive(
        self,
        state: SimulatedMarketState,
        directive: dict,
    ) -> tuple[SimulatedMarketState, float]:
        """Execute a FARMER directive from an Operating Brief.

        Returns:
            (updated_state, r2)
        """
        actions = directive.get("actions", [])
        if not actions:
            return state, 0.0

        # Index offers by id
        offer_map: dict[str, int] = {
            o.offer_id: i for i, o in enumerate(state.pending_offers)
        }

        accepted_offers: list[SimulatedFarmerOffer] = []
        missed_opportunities: list[SimulatedFarmerOffer] = []
        antihack_violations = 0

        offers_to_remove: set[str] = set()
        new_batches: list[SimulatedBatch] = []
        updated_offers = list(state.pending_offers)

        for action in actions:
            offer_id = action.get("offer_id")
            decision = action.get("decision", "").upper()

            if offer_id is None or offer_id not in offer_map:
                continue

            idx = offer_map[offer_id]
            offer = updated_offers[idx]

            if offer.status != FarmerOfferStatus.PENDING:
                continue

            if decision == "ACCEPT":
                state, offer, batch, violation = self._process_accept(
                    state, offer, offer.offered_price_per_kg,
                )
                updated_offers[idx] = offer
                if violation:
                    antihack_violations += 1
                elif batch is not None:
                    new_batches.append(batch)
                    accepted_offers.append(offer)
                    offers_to_remove.add(offer_id)

            elif decision == "COUNTER":
                counter_price = action.get("counter_price")
                if counter_price is None:
                    continue
                state, offer, batch = self._process_counter(
                    state, offer, float(counter_price),
                )
                updated_offers[idx] = offer
                if offer.status == FarmerOfferStatus.ACCEPTED and batch is not None:
                    new_batches.append(batch)
                    accepted_offers.append(offer)
                offers_to_remove.add(offer_id)

            elif decision == "DECLINE":
                offer = replace(offer, status=FarmerOfferStatus.DECLINED)
                updated_offers[idx] = offer
                offers_to_remove.add(offer_id)
                if (
                    offer.viability_score is not None
                    and offer.viability_score >= R2_MISSED_OPPORTUNITY_VIABILITY_THRESHOLD
                ):
                    missed_opportunities.append(offer)

        # Update state
        state.pending_offers = [
            o for o in updated_offers if o.offer_id not in offers_to_remove
        ]
        state.batches = list(state.batches) + new_batches

        r2 = self._compute_r2(accepted_offers, missed_opportunities, antihack_violations, state)
        return state, r2

    def _process_accept(
        self,
        state: SimulatedMarketState,
        offer: SimulatedFarmerOffer,
        intake_price: float,
    ) -> tuple[SimulatedMarketState, SimulatedFarmerOffer, SimulatedBatch | None, bool]:
        """Accept an offer. Returns (state, updated_offer, new_batch_or_None, is_violation)."""
        # Score if not yet scored
        if offer.viability_score is None:
            offer = self.score_offer(offer, state)

        # Anti-hack guard: reckless acceptance
        if offer.viability_score < ANTIHACK_RECKLESS_ACCEPT_VIABILITY_MAX:
            logger.warning(
                "Anti-hack: reckless accept blocked for %s (viability=%.2f)",
                offer.offer_id, offer.viability_score,
            )
            # Do NOT accept — keep as pending, flag violation
            return state, offer, None, True

        avg_market_price = self._avg_category_price(offer.product_category, state)
        batch_id = f"farmer_{offer.offer_id}"

        batch = SimulatedBatch(
            batch_id=batch_id,
            sku_id=f"sku_{offer.product_category}_farmer",
            store_id="store_001",
            category=offer.product_category,
            quantity_remaining=int(offer.quantity_kg),
            unit_cost=intake_price,
            current_price=round(avg_market_price, 2),
            original_price=round(avg_market_price, 2),
            hours_to_expiry=float(offer.seller_shelf_life_hrs),
            batch_type=BatchType.FARMER_SURPLUS,
            status=BatchStatus.ACTIVE,
        )

        # Track for outcome resolution
        self._active_batch_outcomes[batch_id] = {
            "cost": intake_price * offer.quantity_kg,
            "quantity_original": int(offer.quantity_kg),
            "offer_id": offer.offer_id,
        }

        updated_offer = replace(offer, status=FarmerOfferStatus.ACCEPTED)
        return state, updated_offer, batch, False

    def _process_counter(
        self,
        state: SimulatedMarketState,
        offer: SimulatedFarmerOffer,
        counter_price: float,
    ) -> tuple[SimulatedMarketState, SimulatedFarmerOffer, SimulatedBatch | None]:
        """Counter an offer. Simulates farmer response via rng.

        Returns:
            (state, updated_offer, new_batch_or_None)
        """
        offer = replace(
            offer,
            counter_price=counter_price,
            status=FarmerOfferStatus.COUNTERED,
        )

        # Simulated farmer response
        ratio = counter_price / offer.offered_price_per_kg if offer.offered_price_per_kg > 0 else 0
        if ratio >= 0.90:
            acceptance_prob = 0.70
        elif ratio >= 0.80:
            acceptance_prob = 0.40
        else:
            acceptance_prob = 0.0

        if self.rng.random() < acceptance_prob:
            # Farmer accepts the counter — process as acceptance at counter price
            state, offer, batch, _ = self._process_accept(state, offer, counter_price)
            return state, offer, batch
        else:
            # Farmer declines
            offer = replace(offer, status=FarmerOfferStatus.DECLINED)
            return state, offer, None

    # ------------------------------------------------------------------
    # Outcome resolution (called every tick)
    # ------------------------------------------------------------------

    def resolve_outcomes(
        self,
        state: SimulatedMarketState,
    ) -> float:
        """Check farmer-sourced batches for completion. Update risk buffer.

        Called every tick by the environment.
        Returns r2_delta for resolved batches this tick.
        """
        r2_delta = 0.0
        resolved_ids: list[str] = []

        for batch in state.batches:
            if batch.batch_id not in self._active_batch_outcomes:
                continue
            if batch.status not in (BatchStatus.CLEARED, BatchStatus.EXPIRED):
                continue

            outcome = self._active_batch_outcomes[batch.batch_id]
            original_qty = outcome["quantity_original"]
            total_cost = outcome["cost"]

            units_sold = original_qty - batch.quantity_remaining
            sell_through = units_sold / original_qty if original_qty > 0 else 0.0

            # Approximate average sell price
            avg_sell_price = (batch.original_price + batch.floor_price) / 2.0
            actual_revenue = units_sold * avg_sell_price
            ops_cost = original_qty * FARMER_OPS_COST_PER_KG
            net_profit = actual_revenue - total_cost - ops_cost

            # Update risk buffer
            if net_profit > 0:
                state.risk_buffer_balance += net_profit * RISK_BUFFER_PROFIT_CONTRIBUTION_PCT
            else:
                compensation_draw_pct = 1.0 if sell_through >= 0.60 else 0.40
                state.risk_buffer_balance -= abs(net_profit) * compensation_draw_pct

            # r2 bonus for fully cleared batches
            if batch.status == BatchStatus.CLEARED:
                r2_delta += R2_CLEARED_BATCH_BONUS

            resolved_ids.append(batch.batch_id)

        for bid in resolved_ids:
            del self._active_batch_outcomes[bid]

        return r2_delta

    # ------------------------------------------------------------------
    # r2 computation
    # ------------------------------------------------------------------

    def _compute_r2(
        self,
        accepted_offers: list[SimulatedFarmerOffer],
        missed_opportunities: list[SimulatedFarmerOffer],
        antihack_violations: int,
        state: SimulatedMarketState,
    ) -> float:
        """Compute r2 for this directive execution.

        r2 is 0.0 base — only changes when farmer actions happen.
        """
        if not accepted_offers and not missed_opportunities and antihack_violations == 0:
            return 0.0

        r2 = 0.0

        for offer in accepted_offers:
            avg_market_price = self._avg_category_price(offer.product_category, state)
            expected_profit = offer.quantity_kg * (
                avg_market_price - offer.offered_price_per_kg - FARMER_OPS_COST_PER_KG
            )
            if expected_profit > 0 and offer.viability_score is not None:
                r2 += R2_CLEARED_BATCH_BONUS * offer.viability_score

        for _ in missed_opportunities:
            r2 -= R2_MISSED_OPPORTUNITY_PENALTY

        r2 -= R2_RECKLESS_ACCEPT_PENALTY * antihack_violations

        return r2

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _avg_category_price(
        self, category: str, state: SimulatedMarketState,
    ) -> float:
        """Average current_price of active batches in the category.

        Falls back to original_price × 0.80 if no batches exist.
        """
        prices = [
            b.current_price
            for b in state.batches
            if b.category == category and b.status == BatchStatus.ACTIVE
        ]
        if prices:
            return sum(prices) / len(prices)
        # Fallback: look at any batch in this category for original_price
        all_prices = [
            b.original_price
            for b in state.batches
            if b.category == category
        ]
        if all_prices:
            return (sum(all_prices) / len(all_prices)) * 0.80
        # Last resort — should not happen in a well-initialised state
        return 50.0
