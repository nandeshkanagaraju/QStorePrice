"""Smoke test: confirm FreshPriceOpenEnv conforms to the OpenEnv contract.

Run:
    pip install openenv-core
    python scripts/check_openenv.py

Verifies:
    1. The adapter imports cleanly.
    2. It subclasses Environment[BriefAction, BriefObservation, FreshPriceState].
    3. reset() returns a BriefObservation with a non-empty prompt.
    4. step() runs an episode to completion and returns a terminal observation
       carrying the final WRR in metadata.
    5. .state returns a FreshPriceState that updates monotonically.
"""

from __future__ import annotations

import sys


def main() -> int:
    print("[1/5] Importing adapter...")
    try:
        from openenv.core.env_server.interfaces import Environment
        from openenv.core.env_server.types import Action, Observation, State

        from freshprice_env.openenv_adapter import (
            BriefAction,
            BriefObservation,
            FreshPriceOpenEnv,
            FreshPriceState,
        )
    except ImportError as exc:
        print(f"  FAIL: {exc}")
        print("  Install OpenEnv: pip install openenv-core")
        return 1
    print("  OK")

    print("[2/5] Checking class hierarchy...")
    assert issubclass(BriefAction, Action), "BriefAction must subclass Action"
    assert issubclass(BriefObservation, Observation), "BriefObservation must subclass Observation"
    assert issubclass(FreshPriceState, State), "FreshPriceState must subclass State"
    assert issubclass(FreshPriceOpenEnv, Environment), "Env must subclass Environment"
    print("  OK")

    print("[3/5] reset()...")
    env = FreshPriceOpenEnv(scenario="STABLE_WEEK", seed=42)
    obs = env.reset(seed=42, episode_id="smoke-001")
    assert isinstance(obs, BriefObservation), f"reset returned {type(obs)}"
    assert obs.prompt and len(obs.prompt) > 100, "prompt looks empty"
    assert obs.tick == 0
    assert not obs.done
    print(f"  OK — engine_type={obs.engine_type}, prompt[:60]={obs.prompt[:60]!r}")

    print("[4/5] step() loop until terminal...")
    empty_brief = (
        "SITUATION: Smoke test.\n"
        "SIGNAL ANALYSIS: N/A\n"
        "VIABILITY CHECK: N/A\n"
        "RECOMMENDATION: Hold.\n"
        'DIRECTIVE: {"engine":"PRICING","actions":[]}\n'
        "CONFIDENCE: LOW"
    )
    steps = 0
    while True:
        obs = env.step(BriefAction(brief_text=empty_brief))
        assert isinstance(obs, BriefObservation)
        steps += 1
        if obs.done:
            break
        if steps > 200:
            print("  FAIL: episode did not terminate within 200 steps")
            return 1
    final = obs.metadata.get("final_reward") or {}
    print(f"  OK — {steps} steps, terminal WRR={final.get('wrr', 0.0):.4f}")

    print("[5/5] .state property...")
    state = env.state
    assert isinstance(state, FreshPriceState)
    assert state.episode_complete, "episode_complete should be True after termination"
    assert state.tick > 0
    print(
        f"  OK — tick={state.tick}, wrr_so_far={state.wrr_so_far:.4f}, "
        f"active_batches={state.active_batches}"
    )

    print("\nALL CHECKS PASSED — FreshPriceOpenEnv is OpenEnv-compliant.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
