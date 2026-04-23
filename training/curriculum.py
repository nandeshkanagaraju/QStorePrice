"""Curriculum progression manager for the 5 training scenarios.

Tracks episode performance, decides when to promote, and provides
the current scenario to the training loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from freshprice_env.constants import (
    CURRICULUM_PROMOTION_WINDOW,
    CURRICULUM_PROMOTION_WRR_THRESHOLD,
)
from freshprice_env.enums import CurriculumScenario

logger = logging.getLogger(__name__)

# Maximum curriculum level (CRISIS_WEEK = 4)
_MAX_LEVEL: int = CurriculumScenario.CRISIS_WEEK.value


@dataclass
class EpisodeResult:
    """Record of a single training episode."""

    episode_num: int
    scenario: CurriculumScenario
    wrr: float
    brief_quality_score: float
    anti_hack_violations: int
    constitutional_passed: bool
    episode_valid: bool


class CurriculumManager:
    """Manages curriculum progression across the 5 training scenarios."""

    def __init__(self) -> None:
        self.current_scenario: CurriculumScenario = CurriculumScenario.STABLE_WEEK
        self.current_level: int = 0
        self.episodes_in_level: int = 0
        self.total_episodes: int = 0
        # Sliding window — keep last 10 results for current level only
        self._recent_results: list[EpisodeResult] = []
        # Records each promotion: {from_level, to_level, episode_num, avg_wrr}
        self._promotion_history: list[dict] = []

    # ------------------------------------------------------------------
    # Episode recording and promotion
    # ------------------------------------------------------------------

    def record_episode(self, result: EpisodeResult) -> bool:
        """Record an episode result and check for promotion.

        Returns True if promotion occurred this episode, False otherwise.

        Only episodes that are both episode_valid AND constitutional_passed
        count toward the promotion window. Invalid or constitutionally-failed
        episodes are recorded but do not advance the promotion window.
        """
        self.total_episodes += 1
        self.episodes_in_level += 1
        self._recent_results.append(result)

        # Keep sliding window at max 10 entries
        if len(self._recent_results) > 10:
            self._recent_results = self._recent_results[-10:]

        # Already at max level — no promotion possible
        if self.current_level >= _MAX_LEVEL:
            return False

        # Collect valid episodes from recent results
        valid_results = [
            r for r in self._recent_results
            if r.episode_valid and r.constitutional_passed
        ]

        # Need at least CURRICULUM_PROMOTION_WINDOW valid episodes
        if len(valid_results) < CURRICULUM_PROMOTION_WINDOW:
            return False

        # Check the last CURRICULUM_PROMOTION_WINDOW valid episodes
        window = valid_results[-CURRICULUM_PROMOTION_WINDOW:]
        avg_wrr = sum(r.wrr for r in window) / len(window)

        if avg_wrr >= CURRICULUM_PROMOTION_WRR_THRESHOLD:
            self._promote(avg_wrr)
            return True

        return False

    def _promote(self, avg_wrr: float) -> None:
        """Advance to the next scenario."""
        old_level = self.current_level
        new_level = old_level + 1

        self._promotion_history.append({
            "from_level": old_level,
            "to_level": new_level,
            "episode_num": self.total_episodes,
            "avg_wrr": round(avg_wrr, 4),
        })

        self.current_level = new_level
        self.current_scenario = CurriculumScenario(new_level)
        self.episodes_in_level = 0
        self._recent_results = []

        logger.info(
            "Curriculum promoted: level %d (%s) → level %d (%s) at episode %d (avg WRR: %.3f)",
            old_level, CurriculumScenario(old_level).name,
            new_level, self.current_scenario.name,
            self.total_episodes, avg_wrr,
        )

    # ------------------------------------------------------------------
    # Evaluation scheduling
    # ------------------------------------------------------------------

    def should_run_evaluation(self, eval_interval: int = 10) -> bool:
        """Returns True every eval_interval episodes within the current level.

        Evaluation episodes use fixed seeds and do not count toward
        promotion or trajectory collection.
        """
        if self.episodes_in_level == 0:
            return False
        return self.episodes_in_level % eval_interval == 0

    def get_eval_seeds(self, n: int = 5) -> list[int]:
        """Return a fixed list of evaluation seeds for the current level.

        Level-specific: level 0 → [0,1,2,3,4], level 2 → [2000,2001,2002,2003,2004].
        Same seeds every eval run at this level for reproducibility.
        """
        return [self.current_level * 1000 + i for i in range(n)]

    # ------------------------------------------------------------------
    # Status reporting
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return a dict suitable for WandB logging and terminal display."""
        valid_results = [
            r for r in self._recent_results
            if r.episode_valid and r.constitutional_passed
        ]

        if valid_results:
            recent_wrr_mean = sum(r.wrr for r in valid_results) / len(valid_results)
        else:
            recent_wrr_mean = 0.0

        wrr_to_promotion = max(0.0, CURRICULUM_PROMOTION_WRR_THRESHOLD - recent_wrr_mean)

        return {
            "curriculum_level": self.current_level,
            "scenario_name": self.current_scenario.name,
            "episodes_in_level": self.episodes_in_level,
            "total_episodes": self.total_episodes,
            "recent_wrr_mean": round(recent_wrr_mean, 4),
            "recent_wrr_window": len(valid_results),
            "wrr_to_promotion": round(wrr_to_promotion, 4),
            "promotions": list(self._promotion_history),
        }
