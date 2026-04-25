"""OpenEnv adapter for FreshPriceEnv.

The hackathon judging criteria mandate "Usage of OpenEnv (latest release)".
The internal env (`FreshPriceEnv`) is a Gymnasium-style env so that `app.py`,
`inference.py`, the training pipeline, and the evaluator keep working as-is.
This module wraps it in the OpenEnv `Environment[Action, Observation, State]`
contract without duplicating logic.

Install:
    pip install openenv-core

Smoke test:
    python scripts/check_openenv.py
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import Action, Observation, State
except ImportError as exc:
    raise ImportError(
        "openenv-core is not installed. Run: pip install openenv-core"
    ) from exc

from freshprice_env.enums import BriefEngineType, CurriculumScenario
from freshprice_env.freshprice_env import FreshPriceEnv


class BriefAction(Action):
    """The Operating Brief text the LLM produces each cycle."""

    brief_text: str = Field(
        ...,
        description="Raw Operating Brief with SITUATION/SIGNAL/VIABILITY/"
        "RECOMMENDATION/DIRECTIVE/CONFIDENCE sections.",
    )


class BriefObservation(Observation):
    """Prompt the LLM sees + parse/validation telemetry from the last step."""

    prompt: str = Field(
        ...,
        description="The next Operating Brief prompt (current inventory, "
        "market context, task instructions).",
    )
    tick: int = Field(default=0, description="Simulation tick at start of this brief cycle.")
    engine_type: str = Field(default="PRICING", description="Engine the brief targets.")
    parse_success: bool = Field(default=True)
    validation_success: bool = Field(default=True)
    used_fallback: bool = Field(default=False)
    quality_score: float = Field(default=0.0)


class FreshPriceState(State):
    """Snapshot of the simulated grocery store at the current tick."""

    scenario: str = Field(default="STABLE_WEEK")
    tick: int = Field(default=0)
    day_of_week: int = Field(default=0)
    hour_of_day: int = Field(default=0)
    wrr_so_far: float = Field(default=0.0)
    active_batches: int = Field(default=0)
    critical_batches: int = Field(default=0)
    pending_offers: int = Field(default=0)
    active_trends: int = Field(default=0)
    risk_buffer_balance: float = Field(default=0.0)
    engine_type: str = Field(default="PRICING")
    episode_complete: bool = Field(default=False)


class FreshPriceOpenEnv(Environment[BriefAction, BriefObservation, FreshPriceState]):
    """OpenEnv-compliant wrapper around `FreshPriceEnv`.

    Delegates all simulation to the internal Gym env so there is one source of
    truth for engine logic, reward computation, and termination.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = False
    REQUIRES_SINGLE_THREAD_EXECUTOR: bool = True

    def __init__(
        self,
        scenario: CurriculumScenario | str = CurriculumScenario.STABLE_WEEK,
        seed: int = 42,
    ) -> None:
        if isinstance(scenario, str):
            scenario = CurriculumScenario[scenario]
        self._inner = FreshPriceEnv(scenario=scenario, seed=seed)
        self._last_info: dict[str, Any] = {}
        self._last_reward: float = 0.0
        self._final_reward: dict[str, Any] | None = None
        self._episode_id: str | None = None

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> BriefObservation:
        prompt, info = self._inner.reset(seed=seed)
        self._last_info = info
        self._last_reward = 0.0
        self._final_reward = None
        self._episode_id = episode_id
        return BriefObservation(
            prompt=prompt,
            tick=info.get("tick", 0),
            engine_type=info.get("engine_type", BriefEngineType.PRICING.value),
            done=False,
            reward=0.0,
            metadata={"scenario": info.get("scenario"), "episode_id": episode_id},
        )

    def step(
        self,
        action: BriefAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> BriefObservation:
        prompt, reward, terminated, truncated, info = self._inner.step(action.brief_text)
        self._last_info = info
        self._last_reward = float(reward)
        if "final_reward" in info:
            self._final_reward = info["final_reward"]

        return BriefObservation(
            prompt=prompt,
            tick=info.get("tick", 0),
            engine_type=info.get("next_engine_type", info.get("engine_type", "PRICING")),
            parse_success=info.get("parse_success", True),
            validation_success=info.get("validation_success", True),
            used_fallback=info.get("used_fallback", False),
            quality_score=info.get("quality_score", 0.0),
            done=bool(terminated or truncated),
            reward=float(reward),
            metadata={
                "final_reward": self._final_reward,
                "constitutional_audit": info.get("constitutional_audit"),
                "execution_warnings": info.get("execution_warnings", []),
                "episode_id": self._episode_id,
            },
        )

    @property
    def state(self) -> FreshPriceState:
        snapshot = self._inner.state()
        if snapshot.get("status") == "not_started":
            return FreshPriceState(episode_id=self._episode_id, step_count=0)
        return FreshPriceState(
            episode_id=self._episode_id,
            step_count=snapshot.get("tick", 0),
            scenario=snapshot.get("scenario", "STABLE_WEEK"),
            tick=snapshot.get("tick", 0),
            day_of_week=snapshot.get("day_of_week", 0),
            hour_of_day=snapshot.get("hour_of_day", 0),
            wrr_so_far=snapshot.get("wrr_so_far", 0.0),
            active_batches=snapshot.get("active_batches", 0),
            critical_batches=snapshot.get("critical_batches", 0),
            pending_offers=snapshot.get("pending_offers", 0),
            active_trends=snapshot.get("active_trends", 0),
            risk_buffer_balance=snapshot.get("risk_buffer_balance", 0.0),
            engine_type=snapshot.get("engine_type", "PRICING"),
            episode_complete=snapshot.get("episode_complete", False),
        )

    def close(self) -> None:
        if hasattr(self._inner, "close"):
            self._inner.close()
