"""benchmark_suite.scoring — scorer implementations and the ScoreRecord contract."""
from benchmark_suite.scoring.base import (
    Scorer,
    ScoreRecord,
    ScorerRegistry,
    ScoreStatus,
)
from benchmark_suite.scoring.metadata_collector import (
    build_metadata,
    collect_hardware,
    collect_model_info,
    collect_software,
)
from benchmark_suite.scoring.throughput import ThroughputScorerImpl

__all__ = [
    "ScoreRecord",
    "ScoreStatus",
    "Scorer",
    "ScorerRegistry",
    "ThroughputScorerImpl",
    "collect_metadata",
]


def collect_metadata(
    *,
    submitter: str,
    date_str: str,
    notes: str = "",
) -> dict[str, object]:
    """One-shot: auto-detect hardware + software + model, then build_metadata."""
    return build_metadata(
        submitter=submitter,
        date_str=date_str,
        hardware=collect_hardware(),
        software=collect_software(),
        model=collect_model_info(""),  # caller can override model_path
        notes=notes,
    )
