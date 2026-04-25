"""RandomAgent — samples uniformly valid Operating Briefs each step.

Sets the floor: WRR ~0.05. Any trained agent should beat this easily.
Used as the lower-bound in the comparison table.
"""

from __future__ import annotations

import json
import random


class RandomAgent:
    """Generates syntactically valid but semantically random Operating Briefs."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def act(self, observation: str, info: dict) -> str:
        """Return a random but parseable Operating Brief."""
        engine_type = info.get("engine_type", "PRICING")

        if engine_type == "PRICING":
            return self._random_pricing_brief(observation)
        if engine_type == "FARMER":
            return self._random_farmer_brief(observation, info)
        if engine_type == "TREND":
            return self._random_trend_brief(observation, info)
        return self._random_pricing_brief(observation)

    # ------------------------------------------------------------------
    # Per-engine brief generators
    # ------------------------------------------------------------------

    def _random_pricing_brief(self, obs: str) -> str:
        # Extract batch IDs from observation (simple heuristic)
        import re
        batch_ids = re.findall(r"batch_\d{4}", obs) or ["batch_0001"]
        # Pick 1-3 random batches
        selected = self._rng.sample(batch_ids, min(3, len(batch_ids)))
        actions = []
        for bid in selected:
            mult = round(self._rng.uniform(0.40, 1.0), 2)
            actions.append({"batch_id": bid, "price_multiplier": mult, "flash_sale": False})

        directive = json.dumps({"engine": "PRICING", "actions": actions})
        confidence = self._rng.choice(["HIGH", "MEDIUM", "LOW"])

        return (
            "SITUATION: Current inventory assessed.\n"
            "SIGNAL ANALYSIS: N/A\n"
            "VIABILITY CHECK: N/A\n"
            "RECOMMENDATION: Apply random price adjustments to selected batches.\n"
            f"DIRECTIVE: {directive}\n"
            f"CONFIDENCE: {confidence}"
        )

    def _random_farmer_brief(self, obs: str, info: dict) -> str:
        import re
        offer_ids = re.findall(r"offer_\d{3}", obs) or ["offer_001"]
        decisions = ["ACCEPT", "DECLINE", "COUNTER"]
        actions = []
        for oid in offer_ids[:2]:
            decision = self._rng.choice(decisions)
            action: dict = {"offer_id": oid, "decision": decision}
            if decision == "COUNTER":
                action["counter_price"] = round(self._rng.uniform(10.0, 50.0), 2)
            actions.append(action)

        directive = json.dumps({"engine": "FARMER", "actions": actions})
        confidence = self._rng.choice(["HIGH", "MEDIUM", "LOW"])

        return (
            "SITUATION: Farmer offers evaluated.\n"
            "SIGNAL ANALYSIS: N/A\n"
            "VIABILITY CHECK: N/A\n"
            "RECOMMENDATION: Random farmer offer decisions applied.\n"
            f"DIRECTIVE: {directive}\n"
            f"CONFIDENCE: {confidence}"
        )

    def _random_trend_brief(self, obs: str, info: dict) -> str:
        import re
        categories = re.findall(
            r"(fruits|vegetables|dairy|mushrooms|leafy_greens|herbs|bakery|packaged)", obs
        ) or ["fruits"]
        cat = self._rng.choice(categories)
        decision = self._rng.choice(["APPROVE", "DECLINE"])
        order_kg = round(self._rng.uniform(5.0, 20.0), 1)

        actions = [{"category": cat, "decision": decision, "order_quantity_kg": order_kg}]
        directive = json.dumps({"engine": "TREND", "actions": actions})
        confidence = self._rng.choice(["HIGH", "MEDIUM", "LOW"])

        return (
            "SITUATION: Trend signals monitored.\n"
            "SIGNAL ANALYSIS: N/A\n"
            "VIABILITY CHECK: N/A\n"
            f"RECOMMENDATION: Random trend decision for {cat}.\n"
            f"DIRECTIVE: {directive}\n"
            f"CONFIDENCE: {confidence}"
        )
