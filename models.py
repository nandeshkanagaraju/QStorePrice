"""Top-level Action / Observation / State models for the OpenEnv contract.

This module is the public surface that clients import (PDF page 26 step 1
and page 28: "Type-Safe by Design — Define your data structures with Python
dataclasses"). It re-exports the canonical OpenEnv types so that consumers
write:

    from models import BriefAction, BriefObservation, FreshPriceState

without having to know about the internal `freshprice_env.openenv_adapter`
module path.

Keeping these classes in one importable module is what makes the package
pip-installable as a "client SDK" from the HF Space (PDF page 42).
"""

from __future__ import annotations

from freshprice_env.openenv_adapter import (
    BriefAction,
    BriefObservation,
    FreshPriceState,
)

__all__ = ["BriefAction", "BriefObservation", "FreshPriceState"]
