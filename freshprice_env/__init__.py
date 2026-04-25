"""FreshPrice AI — OpenEnv RL environment for perishable goods intelligence."""

from freshprice_env.enums import (
    BatchStatus,
    BatchType,
    BriefConfidence,
    BriefEngineType,
    CompensationPolicy,
    CurriculumScenario,
    ExpiryUrgency,
    FarmerOfferStatus,
    SellerAction,
    SignalSource,
    TrendAction,
    ViabilityOutcome,
)
from freshprice_env.freshprice_env import FreshPriceEnv


def _load_openenv_adapter():
    """Lazy import — openenv-core is an optional dependency."""
    from freshprice_env.openenv_adapter import (
        BriefAction,
        BriefObservation,
        FreshPriceOpenEnv,
        FreshPriceState,
    )
    return BriefAction, BriefObservation, FreshPriceState, FreshPriceOpenEnv


__all__ = [
    "FreshPriceEnv",
    "_load_openenv_adapter",
    "BatchStatus",
    "BatchType",
    "BriefConfidence",
    "BriefEngineType",
    "CompensationPolicy",
    "CurriculumScenario",
    "ExpiryUrgency",
    "FarmerOfferStatus",
    "SellerAction",
    "SignalSource",
    "TrendAction",
    "ViabilityOutcome",
]
