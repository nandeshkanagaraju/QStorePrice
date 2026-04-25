"""In-memory metrics store for QStorePrice runs.

Ported from qstoregym (older sibling project) and adapted to FreshPrice's
brief-driven model. The schema captures WRR, brief quality, anti-hack
violations, and constitutional audit pass/fail rather than raw price
multipliers.

All state lives in a single in-process singleton (`metrics`). Thread-safe via
an internal lock. For production, swap the `_records` deques for Prometheus
counters or a time-series DB.

Usage:
    from freshprice_env.monitoring import metrics
    metrics.record_step(scenario="STABLE_WEEK", tick=8, reward=0.12, ...)
    metrics.record_episode(scenario="STABLE_WEEK", wrr=0.42, ...)
    snapshot = metrics.get_dashboard()  # JSON-serialisable dict
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque

_MAX_EPISODES = 500
_MAX_STEPS = 5000


@dataclass
class EpisodeRecord:
    """Metrics captured at the end of a single episode."""

    scenario: str
    agent_type: str  # "llm", "rule_based", "random", "deterministic"
    wrr: float
    r1_pricing: float
    r2_farmer: float
    r3_trend: float
    brief_quality_score: float
    anti_hack_violations: int
    constitutional_passed: bool
    episode_valid: bool
    steps: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "agent_type": self.agent_type,
            "wrr": round(self.wrr, 4),
            "r1_pricing": round(self.r1_pricing, 4),
            "r2_farmer": round(self.r2_farmer, 4),
            "r3_trend": round(self.r3_trend, 4),
            "brief_quality_score": round(self.brief_quality_score, 4),
            "anti_hack_violations": self.anti_hack_violations,
            "constitutional_passed": self.constitutional_passed,
            "episode_valid": self.episode_valid,
            "steps": self.steps,
            "timestamp": self.timestamp,
        }


@dataclass
class StepRecord:
    """Per-step record for the rolling reward curve."""

    scenario: str
    tick: int
    engine_type: str
    reward: float
    quality_score: float
    parse_success: bool
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "tick": self.tick,
            "engine_type": self.engine_type,
            "reward": round(self.reward, 4),
            "quality_score": round(self.quality_score, 4),
            "parse_success": self.parse_success,
            "timestamp": self.timestamp,
        }


class MetricsStore:
    """Thread-safe singleton holding episode + step records."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._episodes: Deque[EpisodeRecord] = deque(maxlen=_MAX_EPISODES)
        self._steps: Deque[StepRecord] = deque(maxlen=_MAX_STEPS)
        self._started_at = time.time()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_step(
        self,
        scenario: str,
        tick: int,
        engine_type: str,
        reward: float,
        quality_score: float = 0.0,
        parse_success: bool = True,
    ) -> None:
        with self._lock:
            self._steps.append(StepRecord(
                scenario=scenario,
                tick=tick,
                engine_type=engine_type,
                reward=float(reward),
                quality_score=float(quality_score),
                parse_success=bool(parse_success),
            ))

    def record_episode(
        self,
        scenario: str,
        wrr: float,
        r1_pricing: float = 0.0,
        r2_farmer: float = 0.0,
        r3_trend: float = 0.0,
        brief_quality_score: float = 0.0,
        anti_hack_violations: int = 0,
        constitutional_passed: bool = True,
        episode_valid: bool = True,
        steps: int = 0,
        agent_type: str = "llm",
    ) -> None:
        with self._lock:
            self._episodes.append(EpisodeRecord(
                scenario=scenario,
                agent_type=agent_type,
                wrr=float(wrr),
                r1_pricing=float(r1_pricing),
                r2_farmer=float(r2_farmer),
                r3_trend=float(r3_trend),
                brief_quality_score=float(brief_quality_score),
                anti_hack_violations=int(anti_hack_violations),
                constitutional_passed=bool(constitutional_passed),
                episode_valid=bool(episode_valid),
                steps=int(steps),
            ))

    def reset(self) -> None:
        """Wipe all records. Used by tests."""
        with self._lock:
            self._episodes.clear()
            self._steps.clear()
            self._started_at = time.time()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_episode_scores(self, scenario: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            return [
                ep.to_dict()
                for ep in self._episodes
                if scenario is None or ep.scenario == scenario
            ]

    def get_reward_curve(self, scenario: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            return [
                s.to_dict()
                for s in self._steps
                if scenario is None or s.scenario == scenario
            ]

    def get_dashboard(self) -> dict[str, Any]:
        """JSON-serialisable snapshot for the admin dashboard."""
        with self._lock:
            n_eps = len(self._episodes)
            n_steps = len(self._steps)

            if n_eps == 0:
                summary: dict[str, Any] = {
                    "episodes_total": 0,
                    "steps_total": n_steps,
                    "wrr_mean": 0.0,
                    "wrr_max": 0.0,
                    "quality_mean": 0.0,
                    "violations_total": 0,
                    "constitutional_pass_rate": 1.0,
                }
            else:
                wrrs = [e.wrr for e in self._episodes]
                summary = {
                    "episodes_total": n_eps,
                    "steps_total": n_steps,
                    "wrr_mean": round(sum(wrrs) / n_eps, 4),
                    "wrr_max": round(max(wrrs), 4),
                    "quality_mean": round(
                        sum(e.brief_quality_score for e in self._episodes) / n_eps, 4
                    ),
                    "violations_total": sum(e.anti_hack_violations for e in self._episodes),
                    "constitutional_pass_rate": round(
                        sum(1 for e in self._episodes if e.constitutional_passed) / n_eps, 4
                    ),
                }

            # Per-scenario breakdown
            by_scenario: dict[str, dict[str, Any]] = {}
            for ep in self._episodes:
                bucket = by_scenario.setdefault(ep.scenario, {"n": 0, "wrr_sum": 0.0})
                bucket["n"] += 1
                bucket["wrr_sum"] += ep.wrr
            for sc, b in by_scenario.items():
                b["wrr_mean"] = round(b["wrr_sum"] / b["n"], 4)
                del b["wrr_sum"]

            return {
                "summary": summary,
                "by_scenario": by_scenario,
                "recent_episodes": [
                    e.to_dict() for e in list(self._episodes)[-10:]
                ],
                "recent_steps": [
                    s.to_dict() for s in list(self._steps)[-50:]
                ],
                "uptime_seconds": round(time.time() - self._started_at, 1),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }


# Module-level singleton
metrics = MetricsStore()
