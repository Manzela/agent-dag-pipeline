"""
Output Schema — Typed Pipeline Output for Serialization.

Defines the JSON output structure for pipeline batch runs.
Produces a single ``products_per_location.json`` file containing
all results grouped by store location.

Output Structure::

    {
      "pipeline_version": "3.0.0",
      "run_timestamp": "2026-05-08T15:00:00Z",
      "summary": { "total": 10, "passed": 7, "failed": 3, ... },
      "locations": {
        "madrid_001": {
          "store": { ... },
          "products": [
            { "product_id": "SKU-001", "success": true, "content": { ... } },
            ...
          ]
        }
      }
    }
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class ProductResult(BaseModel):
    """Individual product result within a location."""
    model_config = ConfigDict(frozen=True)

    product_id: str
    success: bool
    duration_ms: float = 0.0
    failure_node: str = ""
    failure_reason: str = ""
    content: Optional[dict[str, Any]] = None
    metadata: Optional[dict[str, Any]] = None


class LocationOutput(BaseModel):
    """All products processed for a single store location."""
    model_config = ConfigDict(frozen=True)

    store: dict[str, Any] = Field(default_factory=dict)
    products: list[ProductResult] = Field(default_factory=list)
    summary: dict[str, int] = Field(default_factory=dict)


class RunSummary(BaseModel):
    """Aggregate summary statistics for the entire pipeline run."""
    model_config = ConfigDict(frozen=True)

    total_products: int = 0
    total_stores: int = 0
    passed: int = 0
    failed: int = 0
    pass_rate: float = 0.0
    total_duration_ms: float = 0.0
    avg_duration_ms: float = 0.0


class PipelineRunOutput(BaseModel):
    """Complete pipeline run output, serializable to JSON.

    This is the top-level output schema for a batch pipeline run.
    It contains all results grouped by store location, plus aggregate
    statistics for observability.
    """

    pipeline_version: str = ""
    run_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    config_used: dict[str, Any] = Field(default_factory=dict)
    summary: RunSummary = Field(default_factory=RunSummary)
    locations: dict[str, LocationOutput] = Field(default_factory=dict)

    def add_result(
        self,
        store_id: str,
        store_data: dict[str, Any],
        result: ProductResult,
    ) -> None:
        """Add a product result to its store location group.

        Parameters
        ----------
        store_id : str
            The store identifier.
        store_data : dict
            Raw store context data for reference.
        result : ProductResult
            The individual product result.
        """
        if store_id not in self.locations:
            # Pydantic frozen model — we need to use model_copy or direct dict mutation
            # Since this is the mutable output builder, we relax frozen for this model
            object.__setattr__(
                self, "locations",
                {**self.locations, store_id: LocationOutput(store=store_data)},
            )

        loc = self.locations[store_id]
        new_products = [*loc.products, result]
        passed = sum(1 for p in new_products if p.success)
        failed = sum(1 for p in new_products if not p.success)

        object.__setattr__(
            self, "locations",
            {
                **self.locations,
                store_id: LocationOutput(
                    store=store_data,
                    products=new_products,
                    summary={"passed": passed, "failed": failed, "total": len(new_products)},
                ),
            },
        )

    def compute_summary(self) -> None:
        """Recompute aggregate summary from all location results."""
        total = 0
        passed = 0
        failed = 0
        total_ms = 0.0

        for loc in self.locations.values():
            for p in loc.products:
                total += 1
                total_ms += p.duration_ms
                if p.success:
                    passed += 1
                else:
                    failed += 1

        summary = RunSummary(
            total_products=total,
            total_stores=len(self.locations),
            passed=passed,
            failed=failed,
            pass_rate=round(passed / total, 4) if total > 0 else 0.0,
            total_duration_ms=round(total_ms, 2),
            avg_duration_ms=round(total_ms / total, 2) if total > 0 else 0.0,
        )
        object.__setattr__(self, "summary", summary)

    def write_json(self, output_path: str | Path) -> Path:
        """Serialize the full output to a JSON file.

        Parameters
        ----------
        output_path : str or Path
            Directory or file path for the output. If a directory,
            writes ``products_per_location.json`` inside it.

        Returns
        -------
        Path
            The path to the written file.
        """
        path = Path(output_path)

        if path.suffix != ".json":
            path.mkdir(parents=True, exist_ok=True)
            path = path / "products_per_location.json"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)

        self.compute_summary()

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                self.model_dump(mode="json", exclude_none=True),
                f,
                indent=2,
                ensure_ascii=False,
            )

        return path
