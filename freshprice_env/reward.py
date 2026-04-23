"""WRRRewardEngine — combines r1 + r2 + r3 into the unified WRR metric.

Individual reward components are computed by their engines:
  r1 → PricingEngine.tick()
  r2 → FarmerEngine.process_directive() + FarmerEngine.resolve_outcomes()
  r3 → TrendEngine.process_directive() + TrendEngine.resolve_outcomes()

This class:
  1. Accumulates r1, r2, r3 across the episode
  2. Computes the final WRR at end of episode
  3. Runs the constitutional anti-hack audit before DPO pair generation
  4. Produces the reward dict that GRPOTrainer receives
"""

from __future__ import annotations

from freshprice_env.constants import (
    TREND_COOLDOWN_HRS,
)
from freshprice_env.entities import SimulatedMarketState

# Constitutional audit thresholds
CONSTITUTIONAL_MAX_VIOLATIONS: int = 5
CONSTITUTIONAL_R1_NEGATIVE_PCT_THRESHOLD: float = 0.30
CONSTITUTIONAL_R2_MEAN_FLOOR: float = -1.0
CONSTITUTIONAL_TREND_REPEAT_TICKS: int = 200
CONSTITUTIONAL_TREND_REPEAT_MAX: int = 3


class WRRRewardEngine:
    """Accumulates per-tick rewards and produces episode-level metrics."""

    def __init__(self) -> None:
        self._r1_history: list[float] = []
        self._r2_history: list[float] = []
        self._r3_history: list[float] = []
        self._antihack_violations: list[dict] = []
        self._brief_quality_scores: list[float] = []
        self._tick_count: int = 0

    # ------------------------------------------------------------------
    # Per-tick recording
    # ------------------------------------------------------------------

    def record_tick(
        self,
        r1: float,
        r2: float,
        r3: float,
        tick: int,
    ) -> None:
        """Called every tick after all engines have run.

        r2 and r3 will be 0.0 on most ticks — only non-zero when
        a brief fires with a FARMER or TREND directive.
        """
        self._r1_history.append(r1)
        self._r2_history.append(r2)
        self._r3_history.append(r3)
        self._tick_count = tick + 1

    def record_antihack_violation(
        self,
        tick: int,
        engine: str,
        violation_type: str,
        detail: str,
    ) -> None:
        """Called by individual engines when a guard fires."""
        self._antihack_violations.append({
            "tick": tick,
            "engine": engine,
            "violation_type": violation_type,
            "detail": detail,
        })

    def record_brief_quality(self, quality_score: float) -> None:
        """Called after each Operating Brief is generated and scored.

        Stored separately from reward — this is the independent research metric.
        """
        self._brief_quality_scores.append(quality_score)

    # ------------------------------------------------------------------
    # Episode reward
    # ------------------------------------------------------------------

    def compute_episode_reward(
        self,
        state: SimulatedMarketState,
    ) -> dict:
        """Compute the full reward dict for GRPOTrainer and WandB.

        Called once at episode end by the environment.
        """
        wrr = state.wrr

        r1_mean = self._mean_nonzero(self._r1_history)
        r2_mean = self._mean_nonzero(self._r2_history)
        r3_mean = self._mean_nonzero(self._r3_history)

        brief_quality_mean = (
            sum(self._brief_quality_scores) / len(self._brief_quality_scores)
            if self._brief_quality_scores
            else 0.0
        )

        antihack_count = len(self._antihack_violations)
        episode_valid = antihack_count <= CONSTITUTIONAL_MAX_VIOLATIONS

        return {
            "wrr": wrr,
            "r1_pricing": r1_mean,
            "r2_farmer": r2_mean,
            "r3_trend": r3_mean,
            "brief_quality_score": brief_quality_mean,
            "anti_hack_violations": antihack_count,
            "ticks_completed": self._tick_count,
            "episode_valid": episode_valid,
        }

    # ------------------------------------------------------------------
    # Constitutional audit
    # ------------------------------------------------------------------

    def constitutional_audit(self) -> dict:
        """Audit trajectory for reward hacking before DPO pair generation.

        A trajectory fails if ANY of:
          - antihack_violation_count > 5
          - r1 was negative for > 30% of ticks
          - r2_mean < -1.0 (systematically accepting reckless offers)
          - More than 3 trend orders in same category within 200 ticks

        Trajectories that fail are excluded from DPO training set even if
        their WRR is high. This is the reward hacking prevention layer.
        """
        reasons: list[str] = []

        # Check 1: total anti-hack violations
        violation_count = len(self._antihack_violations)
        if violation_count > CONSTITUTIONAL_MAX_VIOLATIONS:
            reasons.append(
                f"Anti-hack violations ({violation_count}) exceed maximum "
                f"({CONSTITUTIONAL_MAX_VIOLATIONS})"
            )

        # Check 2: r1 negative for > 30% of ticks
        if self._r1_history:
            negative_count = sum(1 for r in self._r1_history if r < 0.0)
            negative_pct = negative_count / len(self._r1_history)
            if negative_pct > CONSTITUTIONAL_R1_NEGATIVE_PCT_THRESHOLD:
                reasons.append(
                    f"r1 negative for {negative_pct:.1%} of ticks "
                    f"(threshold: {CONSTITUTIONAL_R1_NEGATIVE_PCT_THRESHOLD:.0%})"
                )

        # Check 3: r2 mean below floor
        r2_nonzero = [r for r in self._r2_history if r != 0.0]
        if r2_nonzero:
            r2_mean = sum(r2_nonzero) / len(r2_nonzero)
            if r2_mean < CONSTITUTIONAL_R2_MEAN_FLOOR:
                reasons.append(
                    f"r2 mean ({r2_mean:.2f}) below floor "
                    f"({CONSTITUTIONAL_R2_MEAN_FLOOR})"
                )

        # Check 4: trend order flooding per category
        trend_violations = self._check_trend_flooding()
        if trend_violations:
            for cat, count in trend_violations.items():
                reasons.append(
                    f"Trend order flooding in {cat}: {count} orders within "
                    f"{CONSTITUTIONAL_TREND_REPEAT_TICKS} ticks "
                    f"(max {CONSTITUTIONAL_TREND_REPEAT_MAX})"
                )

        return {
            "passed": len(reasons) == 0,
            "reasons": reasons,
            "violation_details": list(self._antihack_violations),
        }

    def _check_trend_flooding(self) -> dict[str, int]:
        """Check for repeated trend orders in the same category within a tick window."""
        trend_orders = [
            v for v in self._antihack_violations
            if v["engine"] == "TREND" and v["violation_type"] == "ORDER_PLACED"
        ]
        if not trend_orders:
            # Also check from non-violation trend orders tracked in r3 history
            # Trend flooding is detected from the violation log — if no TREND
            # violations exist, check r3 history for suspiciously frequent non-zero ticks
            return self._check_r3_frequency()

        # Group by category and check density
        from collections import defaultdict
        cat_ticks: dict[str, list[int]] = defaultdict(list)
        for v in trend_orders:
            cat = v.get("detail", "").split(":")[0] if ":" in v.get("detail", "") else "unknown"
            cat_ticks[cat].append(v["tick"])

        flooding: dict[str, int] = {}
        for cat, ticks in cat_ticks.items():
            ticks.sort()
            for i in range(len(ticks)):
                window_count = sum(
                    1 for j in range(i, len(ticks))
                    if ticks[j] - ticks[i] < CONSTITUTIONAL_TREND_REPEAT_TICKS
                )
                if window_count > CONSTITUTIONAL_TREND_REPEAT_MAX:
                    flooding[cat] = window_count
                    break
        return flooding

    def _check_r3_frequency(self) -> dict[str, int]:
        """Fallback: detect trend flooding from r3 history frequency.

        If r3 is non-zero more than CONSTITUTIONAL_TREND_REPEAT_MAX times
        within any CONSTITUTIONAL_TREND_REPEAT_TICKS window, flag it.
        """
        nonzero_ticks = [i for i, r in enumerate(self._r3_history) if r != 0.0]
        if len(nonzero_ticks) <= CONSTITUTIONAL_TREND_REPEAT_MAX:
            return {}

        for i in range(len(nonzero_ticks)):
            window_count = sum(
                1 for j in range(i, len(nonzero_ticks))
                if nonzero_ticks[j] - nonzero_ticks[i] < CONSTITUTIONAL_TREND_REPEAT_TICKS
            )
            if window_count > CONSTITUTIONAL_TREND_REPEAT_MAX:
                return {"unknown_category": window_count}
        return {}

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all accumulators for a new episode."""
        self._r1_history = []
        self._r2_history = []
        self._r3_history = []
        self._antihack_violations = []
        self._brief_quality_scores = []
        self._tick_count = 0

    # ------------------------------------------------------------------
    # WandB logging
    # ------------------------------------------------------------------

    def to_wandb_log(
        self,
        state: SimulatedMarketState,
        curriculum_level: int,
        episode_num: int,
    ) -> dict:
        """Produce the flat dict for wandb.log() after each episode.

        Metric names match the WandB spec exactly.
        """
        reward = self.compute_episode_reward(state)
        audit = self.constitutional_audit()

        return {
            "wrr": reward["wrr"],
            "r1_pricing": reward["r1_pricing"],
            "r2_farmer": reward["r2_farmer"],
            "r3_trend": reward["r3_trend"],
            "brief_quality_score": reward["brief_quality_score"],
            "anti_hack_violations": reward["anti_hack_violations"],
            "curriculum_level": curriculum_level,
            "episode_num": episode_num,
            "episode_valid": reward["episode_valid"],
            "constitutional_passed": audit["passed"],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_nonzero(values: list[float]) -> float:
        """Mean of non-zero values, or 0.0 if none."""
        nonzero = [v for v in values if v != 0.0]
        if not nonzero:
            return 0.0
        return sum(nonzero) / len(nonzero)
