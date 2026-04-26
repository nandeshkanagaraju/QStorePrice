"""Single-session HTTP API for the React sim dashboard.

OpenEnv's stateless POST /reset + /step handlers create a fresh env per request,
so multi-step episodes cannot run over plain HTTP. This router keeps one
``FreshPriceEnv`` in memory for local/demo use (one concurrent browser session).
"""

from __future__ import annotations

import json
import threading
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from freshprice_env.enums import BatchStatus, CurriculumScenario, FarmerOfferStatus, TrendAction
from freshprice_env.freshprice_env import FreshPriceEnv

_lock = threading.Lock()
_env: FreshPriceEnv | None = None
_current_obs: str = ""
_step_count: int = 0
_wrr_history: list[float] = []

router = APIRouter(prefix="/api/sim", tags=["Sim Dashboard"])


def _snapshot_prices(env: FreshPriceEnv) -> dict[str, float]:
    if env._state is None:
        return {}
    return {
        b.batch_id: float(b.current_price)
        for b in env._state.batches
        if b.status == BatchStatus.ACTIVE
    }


def _batch_rows(
    state: Any,
    prev_prices: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if state is None:
        return rows
    for b in state.batches:
        if b.status != BatchStatus.ACTIVE:
            continue
        prev = prev_prices.get(b.batch_id) if prev_prices else None
        cur = round(b.current_price, 2)
        row: dict[str, Any] = {
            "batch_id": b.batch_id,
            "category": b.category,
            "urgency": b.urgency.value,
            "current_price": cur,
            "original_price": round(b.original_price, 2),
            "discount_pct": round(b.discount_pct, 1),
            "hours_to_expiry": round(b.hours_to_expiry, 1),
            "quantity": b.quantity_remaining,
        }
        if prev is not None and abs(prev - cur) > 1e-6:
            row["previous_price"] = round(prev, 2)
        rows.append(row)
    return sorted(rows, key=lambda r: r["batch_id"])


def _build_suggested_brief(env: FreshPriceEnv, engine: str) -> str:
    """Operating brief that uses *real* batch / offer / trend ids so prices actually move."""
    st = env._state
    if st is None:
        return ""
    eng = (engine or "PRICING").upper()

    if eng == "PRICING":
        active = [b for b in st.batches if b.status == BatchStatus.ACTIVE]
        active.sort(key=lambda x: x.batch_id)
        actions: list[dict[str, Any]] = []
        for b in active[:10]:
            u = b.urgency.value
            if u == "CRITICAL":
                pm = 0.62
            elif u == "URGENT":
                pm = 0.74
            elif u == "WATCH":
                pm = 0.86
            else:
                pm = 0.94
            actions.append(
                {
                    "batch_id": b.batch_id,
                    "price_multiplier": pm,
                    "flash_sale": False,
                    "bundle_with": None,
                }
            )
        directive: dict[str, Any] = {"engine": "PRICING", "actions": actions}
    elif eng == "FARMER":
        pending = [o for o in st.pending_offers if o.status == FarmerOfferStatus.PENDING]
        if pending:
            o = pending[0]
            directive = {
                "engine": "FARMER",
                "actions": [{"offer_id": o.offer_id, "decision": "ACCEPT", "counter_price": None}],
            }
        else:
            directive = {"engine": "FARMER", "actions": []}
    elif eng == "TREND":
        acts: list[dict[str, Any]] = []
        for _tid, sig in st.trend_signals.items():
            if sig.action_taken != TrendAction.PENDING:
                continue
            if not sig.is_actionable(st.tick):
                continue
            acts.append(
                {
                    "category": sig.category,
                    "decision": "APPROVE",
                    "order_quantity_kg": round(max(5.0, float(sig.suggested_order_kg)), 1),
                }
            )
            if len(acts) >= 2:
                break
        if not acts and st.trend_signals:
            sig = next(iter(st.trend_signals.values()))
            acts = [{"category": sig.category, "decision": "APPROVE", "order_quantity_kg": 12.0}]
        directive = {"engine": "TREND", "actions": acts}
    else:
        directive = {"engine": "PRICING", "actions": []}

    djson = json.dumps(directive, separators=(",", ":"))
    return (
        "SITUATION: Automated store refresh for this decision window.\n\n"
        "SIGNAL ANALYSIS: N/A\n\n"
        "VIABILITY CHECK: N/A\n\n"
        "RECOMMENDATION: Execute the attached DIRECTIVE for this cycle.\n\n"
        f"DIRECTIVE:\n{djson}\n\n"
        "CONFIDENCE: MEDIUM"
    )


class ResetBody(BaseModel):
    scenario: str = Field(default="STABLE_WEEK")
    seed: int = Field(default=42, ge=0)


class StepBody(BaseModel):
    brief_text: str = Field(..., min_length=1)


@router.post("/reset")
def sim_reset(body: ResetBody) -> dict[str, Any]:
    global _env, _current_obs, _step_count, _wrr_history
    try:
        scenario = CurriculumScenario[body.scenario]
    except KeyError as exc:
        raise HTTPException(400, f"Unknown scenario: {body.scenario}") from exc

    with _lock:
        _env = FreshPriceEnv(scenario=scenario, seed=body.seed)
        _current_obs, info = _env.reset(seed=body.seed)
        _step_count = 0
        st = _env.state()
        wrr = float(st.get("wrr_so_far", 0.0)) if st.get("status") != "not_started" else 0.0
        _wrr_history = [wrr]
        eng = str(info.get("engine_type", "PRICING"))
        suggested = _build_suggested_brief(_env, eng)

        return {
            "ok": True,
            "scenario": scenario.name,
            "seed": body.seed,
            "observation": _current_obs,
            "engine_type": eng,
            "state": st,
            "batches": _batch_rows(_env._state, None),
            "step_count": _step_count,
            "wrr_history": list(_wrr_history),
            "suggested_brief": suggested,
            "done": False,
        }


@router.post("/step")
def sim_step(body: StepBody) -> dict[str, Any]:
    global _env, _current_obs, _step_count, _wrr_history

    with _lock:
        if _env is None:
            raise HTTPException(400, "Call POST /api/sim/reset first.")

        prev_prices = _snapshot_prices(_env)
        obs, reward, done, truncated, info = _env.step(body.brief_text)
        _step_count += 1
        st = _env.state()
        wrr = float(st.get("wrr_so_far", 0.0)) if st.get("status") != "not_started" else 0.0
        _wrr_history.append(wrr)

        if done or truncated:
            final = info.get("final_reward", {})
            _current_obs = obs
            return {
                "ok": True,
                "observation": obs,
                "reward": float(reward),
                "done": True,
                "truncated": bool(truncated),
                "state": st,
                "batches": _batch_rows(_env._state, prev_prices),
                "step_count": _step_count,
                "wrr_history": list(_wrr_history),
                "parse_success": info.get("parse_success", True),
                "quality_score": float(info.get("quality_score", 0.0)),
                "final_reward": final,
                "next_engine_type": None,
                "suggested_brief": "",
            }

        next_eng = str(info.get("next_engine_type", "PRICING"))
        suggested = _build_suggested_brief(_env, next_eng)
        _current_obs = obs
        return {
            "ok": True,
            "observation": obs,
            "reward": float(reward),
            "done": False,
            "truncated": False,
            "state": st,
            "batches": _batch_rows(_env._state, prev_prices),
            "step_count": _step_count,
            "wrr_history": list(_wrr_history),
            "parse_success": info.get("parse_success", True),
            "quality_score": float(info.get("quality_score", 0.0)),
            "next_engine_type": next_eng,
            "suggested_brief": suggested,
        }


@router.get("/state")
def sim_state() -> dict[str, Any]:
    with _lock:
        if _env is None:
            return {"started": False}
        st = _env.state()
        et = getattr(_env, "_last_engine_type", None)
        eng = et.value if et is not None else str(st.get("engine_type", "PRICING"))
        return {
            "started": True,
            "state": st,
            "batches": _batch_rows(_env._state, None),
            "step_count": _step_count,
            "wrr_history": list(_wrr_history),
            "suggested_brief": _build_suggested_brief(_env, eng),
        }
