"""benchmark_suite/scoring/base.py — ScoreRecord contract and ScorerRegistry."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from benchmark_suite.recipe import BenchSection, Recipe


@dataclass(frozen=True)
class ScoreStatus:
    """Outcome of one scorer run."""

    SUCCESS: str = "success"
    FAILURE: str = "failure"
    SKIPPED: str = "skipped"


@dataclass
class ScoreRecord:
    """One scorer's contribution to a result dir.

    Serialized to summary.json + summary.csv by report.py.
    """

    kind: str
    cell_id: str
    status: str
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metrics: dict[str, float | int | str] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "cell_id": self.cell_id,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "error": self.error,
            "notes": self.notes,
        }


class Scorer(ABC):
    """Base class for a scorer implementation. One concrete class per (kind, variant)."""

    kind: ClassVar[str] = ""

    def __init__(self, config: object) -> None:
        self.config = config

    @abstractmethod
    def score(
        self, recipe: Recipe, *, result_dir: Path, endpoint_url: str | None = None
    ) -> ScoreRecord:
        ...


class ScorerRegistry:
    """Maps scorer config kind to scorer implementation class."""

    _registry: ClassVar[dict[str, type[Scorer]]] = {}

    @classmethod
    def register[T: type[Scorer]](cls, scorer_cls: T) -> T:
        """Class-decorator: register a scorer implementation by its kind."""
        if not scorer_cls.kind:
            raise ValueError(
                f"scorer class {scorer_cls.__name__} must have a non-empty kind"
            )
        if not issubclass(scorer_cls, Scorer):
            raise ValueError(
                f"{scorer_cls.__name__} must be a subclass of Scorer"
            )
        cls._registry[scorer_cls.kind] = scorer_cls
        return scorer_cls

    @classmethod
    def get(cls, kind: str) -> type[Scorer] | None:
        """Look up a registered scorer implementation by kind."""
        return cls._registry.get(kind)

    @classmethod
    def all(cls) -> dict[str, type[Scorer]]:
        """All registered scorer kinds, in registration order."""
        return dict(cls._registry)

    @classmethod
    def build_scorers(cls, bench: BenchSection) -> list[Scorer]:
        """Instantiate a Scorer for each ScorerConfig in bench.scoring."""
        scorers: list[Scorer] = []
        for cfg in bench.scoring:
            scorer_cls = cls._registry.get(cfg.kind)
            if scorer_cls is None:
                raise ValueError(f"no scorer registered for kind: {cfg.kind}")
            scorers.append(scorer_cls(cfg))
        return scorers


def scorer[T: type[Scorer]](cls: T) -> T:
    """Shorthand for ScorerRegistry.register decorator."""
    return ScorerRegistry.register(cls)
