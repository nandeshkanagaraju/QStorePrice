"""Top-level Action / Observation / State models for the OpenEnv contract.

This module is the public surface that clients import. It follows OpenEnv's
"Type-Safe by Design" pattern — every action, observation, and state object
is a Python dataclass that the server validates on every payload. Consumers
write:

    from models import BriefAction, BriefObservation, FreshPriceState

without having to know about the internal `freshprice_env.openenv_adapter`
module path.

Keeping these classes in one importable module is what makes the package
pip-installable as a "client SDK" from the live demo HF Space (the demo
backing our Gemma 4 Good Hackathon submission).
"""

from __future__ import annotations

from freshprice_env.openenv_adapter import (
    BriefAction,
    BriefObservation,
    FreshPriceState,
)

__all__ = ["BriefAction", "BriefObservation", "FreshPriceState"]
