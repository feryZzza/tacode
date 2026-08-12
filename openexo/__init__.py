"""Cross-dataset open exoskeleton learning toolkit."""

from .adapters import TrialArrays, TrialSpec, discover_all, load_trial
from .dataset import TrialWindowDataset
from .model import MaskAwareMomentTCN
from .schema import CANONICAL_FEATURES, CANONICAL_TARGETS, features_for_profile

__all__ = [
    "CANONICAL_FEATURES",
    "CANONICAL_TARGETS",
    "MaskAwareMomentTCN",
    "TrialArrays",
    "TrialSpec",
    "TrialWindowDataset",
    "discover_all",
    "features_for_profile",
    "load_trial",
]
