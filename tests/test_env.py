"""Smoke tests for the FreshPrice environment.

Run with:
    python -m unittest tests.test_env
or
    pytest tests/

Covers:
  - reset() across all five CurriculumScenarios
  - step() with the canonical fallback brief returns finite reward
  - get_base_demand_velocity() respects time-of-day ordering
  - monitoring.metrics roundtrip (record + dashboard)
  - SFT data generator produces valid 6-section completions
"""

from __future__ import annotations

import math
import unittest

from freshprice_env.enums import CurriculumScenario
from freshprice_env.freshprice_env import FreshPriceEnv
from freshprice_env.market_state import get_base_demand_velocity
from freshprice_env.monitoring import metrics


FALLBACK_BRIEF = """\
SITUATION: Inventory assessed; one batch is at-risk and the rest are stable.

SIGNAL ANALYSIS: N/A

VIABILITY CHECK: N/A

RECOMMENDATION: Apply conservative discounts to URGENT batches.

DIRECTIVE:
{"engine": "PRICING", "actions": []}

CONFIDENCE: MEDIUM
"""


class TestFreshPriceEnv(unittest.TestCase):

    def test_reset_all_scenarios(self):
        for scenario in CurriculumScenario:
            with self.subTest(scenario=scenario.name):
                env = FreshPriceEnv(scenario=scenario, seed=42)
                obs, info = env.reset()
                self.assertIsInstance(obs, str)
                self.assertGreater(len(obs), 100, f"obs too short for {scenario.name}")
                self.assertIn("engine_type", info)

    def test_step_returns_finite_reward(self):
        env = FreshPriceEnv(scenario=CurriculumScenario.STABLE_WEEK, seed=42)
        env.reset()
        obs, reward, done, truncated, info = env.step(FALLBACK_BRIEF)
        self.assertTrue(math.isfinite(reward), "reward must be finite")
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
        self.assertIn("parse_success", info)

    def test_three_step_episode_progresses(self):
        env = FreshPriceEnv(scenario=CurriculumScenario.STABLE_WEEK, seed=42)
        env.reset()
        rewards = []
        for _ in range(3):
            _, reward, done, _, _ = env.step(FALLBACK_BRIEF)
            rewards.append(reward)
            if done:
                break
        self.assertEqual(len(rewards), 3, "three steps should fit before termination")

    def test_reward_engine_does_not_explode(self):
        env = FreshPriceEnv(scenario=CurriculumScenario.CRISIS_WEEK, seed=99)
        env.reset()
        obs, reward, done, truncated, info = env.step(FALLBACK_BRIEF)
        self.assertGreater(reward, -10, "single-step reward should not be wildly negative")
        self.assertLess(reward, 10, "single-step reward should not be wildly positive")


class TestDemandFunction(unittest.TestCase):

    def test_evening_peak_higher_than_night(self):
        evening = get_base_demand_velocity("vegetables", hour_of_day=19, day_of_week=2)
        night = get_base_demand_velocity("vegetables", hour_of_day=2, day_of_week=2)
        self.assertGreater(evening, night, "evening (peak) demand should exceed deep-night demand")

    def test_saturday_above_weekday(self):
        saturday = get_base_demand_velocity("vegetables", hour_of_day=12, day_of_week=5)
        wednesday = get_base_demand_velocity("vegetables", hour_of_day=12, day_of_week=2)
        self.assertGreater(saturday, wednesday, "Saturday (1.5x) should exceed Wednesday (1.0x)")

    def test_unknown_category_does_not_raise(self):
        # Unknown categories fall back to default base velocity 1.0
        v = get_base_demand_velocity("does-not-exist", hour_of_day=12, day_of_week=0)
        self.assertGreater(v, 0)


class TestMonitoring(unittest.TestCase):

    def setUp(self):
        metrics.reset()

    def test_record_and_snapshot(self):
        metrics.record_step(scenario="STABLE_WEEK", tick=8, engine_type="PRICING",
                            reward=0.12, quality_score=0.7)
        metrics.record_episode(scenario="STABLE_WEEK", wrr=0.42,
                               brief_quality_score=0.8, anti_hack_violations=1, steps=84)
        snap = metrics.get_dashboard()
        self.assertEqual(snap["summary"]["episodes_total"], 1)
        self.assertEqual(snap["summary"]["steps_total"], 1)
        self.assertEqual(len(snap["recent_episodes"]), 1)
        self.assertIn("STABLE_WEEK", snap["by_scenario"])

    def test_reset_clears(self):
        metrics.record_episode(scenario="STABLE_WEEK", wrr=0.1)
        metrics.reset()
        snap = metrics.get_dashboard()
        self.assertEqual(snap["summary"]["episodes_total"], 0)


class TestSFTGenerator(unittest.TestCase):

    def test_pricing_examples_have_required_sections(self):
        from training.generate_sft_data import generate_pricing_examples
        examples = generate_pricing_examples(n_per_difficulty=1)
        self.assertEqual(len(examples), 3)
        required = ["SITUATION:", "RECOMMENDATION:", "DIRECTIVE:", "CONFIDENCE:"]
        for ex in examples:
            for section in required:
                self.assertIn(section, ex["completion"])


if __name__ == "__main__":
    unittest.main()
