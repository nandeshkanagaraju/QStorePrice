"""HTTP/WebSocket client for the QStorePrice OpenEnv environment.

Judges and trainers connect to a running QStorePrice server (locally or on a
Hugging Face Space) through this client. It is intentionally thin: all
simulation logic lives on the server side.

Typical usage:

    # Sync (most common for training loops)
    env = QStorePriceEnv(base_url="http://localhost:8000").sync()
    with env:
        result = env.reset(seed=42)
        result = env.step(BriefAction(brief_text="..."))

    # Async
    async with QStorePriceEnv(base_url="http://localhost:8000") as env:
        result = await env.reset(seed=42)
        result = await env.step(BriefAction(brief_text="..."))

The client never imports server internals (engines, brief pipeline, market
state). It only knows about the typed BriefAction, BriefObservation, and
FreshPriceState dataclasses defined in models.py.
"""

from __future__ import annotations

from typing import Any, Dict

try:
    from openenv.core import EnvClient
    from openenv.core.client_types import StepResult
except ImportError as exc:
    raise ImportError(
        "openenv-core is not installed. Run: pip install openenv-core"
    ) from exc

from models import BriefAction, BriefObservation, FreshPriceState


class QStorePriceEnv(EnvClient[BriefAction, BriefObservation, FreshPriceState]):
    """WebSocket client for the QStorePrice perishable-goods environment.

    Subclasses EnvClient so it gets reset(), step(), state(), health(), close(),
    and the async/sync context-manager protocol for free. We implement the three
    payload-translation methods required by the base class.
    """

    def _step_payload(self, action: BriefAction) -> Dict[str, Any]:
        return {"brief_text": action.brief_text}

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[BriefObservation]:
        obs_data = payload.get("observation", {})
        observation = BriefObservation(
            prompt=obs_data.get("prompt", ""),
            tick=obs_data.get("tick", 0),
            engine_type=obs_data.get("engine_type", "PRICING"),
            parse_success=obs_data.get("parse_success", True),
            validation_success=obs_data.get("validation_success", True),
            used_fallback=obs_data.get("used_fallback", False),
            quality_score=obs_data.get("quality_score", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> FreshPriceState:
        return FreshPriceState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            scenario=payload.get("scenario", "STABLE_WEEK"),
            tick=payload.get("tick", 0),
            day_of_week=payload.get("day_of_week", 0),
            hour_of_day=payload.get("hour_of_day", 0),
            wrr_so_far=payload.get("wrr_so_far", 0.0),
            active_batches=payload.get("active_batches", 0),
            critical_batches=payload.get("critical_batches", 0),
            pending_offers=payload.get("pending_offers", 0),
            active_trends=payload.get("active_trends", 0),
            risk_buffer_balance=payload.get("risk_buffer_balance", 0.0),
            engine_type=payload.get("engine_type", "PRICING"),
            episode_complete=payload.get("episode_complete", False),
        )


__all__ = ["QStorePriceEnv", "BriefAction", "BriefObservation", "FreshPriceState"]
