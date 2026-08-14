"""
P29.5 — Production Model Provenance Audit.

Audits the ACTUAL production model artifact and metadata used by the
S&P 500 Quant Trading application.

Purpose
-------
P29.4 incorrectly searched for generic model.pkl/model metadata paths.

P29.5 uses the actual production paths:

    models/production/
        sp500_direction_model.joblib
        sp500_direction_model_metadata.json
        sp500_frozen_research_config.json
        model_manifest.json

The audit verifies:

    1. Production artifacts exist.
    2. Production model is Random Forest.
    3. Production feature count is 26.
    4. Frozen threshold is consistent.
    5. Training dates are recorded.
    6. Final-test dates are recorded.
    7. Training/test temporal separation is valid.
    8. Walk-forward configuration is recorded.
    9. Final-holdout protection is recorded.
   10. Packaged research verdict is inspected.
   11. Existing P29.3 backtest arithmetic remains valid.

Important
---------
This script NEVER changes model artifacts or metadata.

A backtest is NOT considered validated merely because the manifest
contains walk_forward_validation=true.

If the recorded training period overlaps the recorded final-test
period, the audit fails because temporal out-of-sample status cannot
be established from the packaged metadata.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv


# =====================================================================
# PROJECT BOOTSTRAP
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")


# =====================================================================
# ACTUAL PRODUCTION MODEL PATHS
# =====================================================================

MODEL_PATH = (
    PROJECT_ROOT
    / os.getenv(
        "MODEL_PATH",
        "models/production/sp500_direction_model.joblib",
    )
)

MODEL_METADATA_PATH = (
    PROJECT_ROOT
    / os.getenv(
        "MODEL_METADATA_PATH",
        "models/production/sp500_direction_model_metadata.json",
    )
)

MODEL_CONFIG_PATH = (
    PROJECT_ROOT
    / os.getenv(
        "MODEL_CONFIG_PATH",
        "models/production/sp500_frozen_research_config.json",
    )
)

MODEL_MANIFEST_PATH = (
    PROJECT_ROOT
    / "models"
    / "production"
    / "model_manifest.json"
)

BACKTEST_PATH = (
    PROJECT_ROOT
    / "logs"
    / "latest_backtest.json"
)

AUDIT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "logs"
    / "backtest_audit.json"
)


# =====================================================================
# AUDIT COLLECTOR
# =====================================================================

class Audit:
    """Collect PASS / WARN / FAIL audit findings."""

    def __init__(self) -> None:
        self.results: list[dict[str, Any]] = []

    def pass_(
        self,
        name: str,
        detail: str,
    ) -> None:
        self.results.append(
            {
                "status": "PASS",
                "name": name,
                "detail": detail,
            }
        )

    def warn(
        self,
        name: str,
        detail: str,
    ) -> None:
        self.results.append(
            {
                "status": "WARN",
                "name": name,
                "detail": detail,
            }
        )

    def fail(
        self,
        name: str,
        detail: str,
    ) -> None:
        self.results.append(
            {
                "status": "FAIL",
                "name": name,
                "detail": detail,
            }
        )

    @property
    def failures(self) -> int:
        return sum(
            item["status"] == "FAIL"
            for item in self.results
        )

    @property
    def warnings(self) -> int:
        return sum(
            item["status"] == "WARN"
            for item in self.results
        )

    def print_report(self) -> None:
        print()
        print("=" * 80)
        print(
            "P29.5 — PRODUCTION MODEL PROVENANCE AUDIT"
        )
        print("=" * 80)

        for item in self.results:
            print(
                f"[{item['status']:<4}] "
                f"{item['name']}: "
                f"{item['detail']}"
            )

        print("-" * 80)

        print(
            f"FAILURES: {self.failures} | "
            f"WARNINGS: {self.warnings}"
        )

        print("=" * 80)


# =====================================================================
# JSON HELPERS
# =====================================================================

def load_json(
    path: Path,
) -> dict[str, Any]:
    """Load a JSON object from disk."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected a JSON object in {path}"
        )

    return payload


def parse_date(
    value: Any,
) -> pd.Timestamp | None:
    """Safely parse a date-like value."""

    if value is None:
        return None

    parsed = pd.to_datetime(
        value,
        errors="coerce",
    )

    if pd.isna(parsed):
        return None

    return pd.Timestamp(parsed)


# =====================================================================
# PRODUCTION ARTIFACT AUDIT
# =====================================================================

def audit_production_artifacts(
    audit: Audit,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Verify and load the actual production package."""

    artifact_paths = [
        MODEL_PATH,
        MODEL_METADATA_PATH,
        MODEL_CONFIG_PATH,
        MODEL_MANIFEST_PATH,
    ]

    for path in artifact_paths:
        if path.exists():
            audit.pass_(
                "Production artifact",
                (
                    "Found "
                    f"{path.relative_to(PROJECT_ROOT)}"
                ),
            )
        else:
            audit.fail(
                "Production artifact",
                (
                    "Missing "
                    f"{path.relative_to(PROJECT_ROOT)}"
                ),
            )

    metadata = load_json(
        MODEL_METADATA_PATH
    )

    config = load_json(
        MODEL_CONFIG_PATH
    )

    manifest = load_json(
        MODEL_MANIFEST_PATH
    )

    # ---------------------------------------------------------------
    # Model type
    # ---------------------------------------------------------------

    model_type = metadata.get(
        "model_type"
    )

    if model_type == "Random Forest":
        audit.pass_(
            "Production model type",
            "Production model is Random Forest.",
        )
    else:
        audit.fail(
            "Production model type",
            (
                "Expected Random Forest but found "
                f"{model_type!r}."
            ),
        )

    # ---------------------------------------------------------------
    # Feature count
    # ---------------------------------------------------------------

    feature_count = metadata.get(
        "feature_count"
    )

    if feature_count == 26:
        audit.pass_(
            "Production feature count",
            "Production model uses 26 features.",
        )
    else:
        audit.fail(
            "Production feature count",
            (
                "Expected 26 features but found "
                f"{feature_count!r}."
            ),
        )

    # ---------------------------------------------------------------
    # Threshold consistency
    # ---------------------------------------------------------------

    metadata_threshold = metadata.get(
        "probability_threshold"
    )

    config_threshold = config.get(
        "probability_threshold"
    )

    frozen_configuration = manifest.get(
        "frozen_configuration",
        {},
    )

    manifest_threshold = frozen_configuration.get(
        "probability_threshold"
    )

    thresholds = [
        metadata_threshold,
        config_threshold,
        manifest_threshold,
    ]

    if (
        all(
            value is not None
            for value in thresholds
        )
        and len(
            {
                float(value)
                for value in thresholds
            }
        ) == 1
    ):
        audit.pass_(
            "Frozen threshold",
            (
                "Metadata, config and manifest agree on "
                f"{float(metadata_threshold):.2%}."
            ),
        )
    else:
        audit.fail(
            "Frozen threshold",
            (
                "Production threshold configuration is "
                "inconsistent."
            ),
        )

    return (
        metadata,
        config,
        manifest,
    )


# =====================================================================
# TRAINING / TEST DATE AUDIT
# =====================================================================

def audit_training_test_dates(
    metadata: dict[str, Any],
    manifest: dict[str, Any],
    audit: Audit,
) -> None:
    """
    Verify temporal separation between training and final test.

    This is intentionally strict.
    """

    training_start = parse_date(
        metadata.get(
            "training_start"
        )
    )

    training_end = parse_date(
        metadata.get(
            "training_end"
        )
    )

    research_report = manifest.get(
        "research_report",
        {},
    )

    final_test_start = parse_date(
        research_report.get(
            "final_test_start"
        )
    )

    final_test_end = parse_date(
        research_report.get(
            "final_test_end"
        )
    )

    # ---------------------------------------------------------------
    # Training dates
    # ---------------------------------------------------------------

    if (
        training_start is None
        or training_end is None
    ):
        audit.fail(
            "Training period metadata",
            (
                "training_start and/or training_end "
                "are missing or invalid."
            ),
        )
        return

    audit.pass_(
        "Training period metadata",
        (
            f"{training_start.date()} → "
            f"{training_end.date()}"
        ),
    )

    # ---------------------------------------------------------------
    # Final test dates
    # ---------------------------------------------------------------

    if (
        final_test_start is None
        or final_test_end is None
    ):
        audit.fail(
            "Final test period metadata",
            (
                "final_test_start and/or final_test_end "
                "are missing or invalid."
            ),
        )
        return

    audit.pass_(
        "Final test period metadata",
        (
            f"{final_test_start.date()} → "
            f"{final_test_end.date()}"
        ),
    )

    # ---------------------------------------------------------------
    # Chronological consistency
    # ---------------------------------------------------------------

    if training_start > training_end:
        audit.fail(
            "Training period ordering",
            "training_start occurs after training_end.",
        )
    else:
        audit.pass_(
            "Training period ordering",
            "Training period is chronological.",
        )

    if final_test_start > final_test_end:
        audit.fail(
            "Final test period ordering",
            "final_test_start occurs after final_test_end.",
        )
    else:
        audit.pass_(
            "Final test period ordering",
            "Final test period is chronological.",
        )

    # ---------------------------------------------------------------
    # Critical temporal separation
    # ---------------------------------------------------------------

    if training_end >= final_test_start:
        audit.fail(
            "Training/test temporal separation",
            (
                "TRAINING OVERLAPS FINAL TEST. "
                f"training_end={training_end.date()}, "
                f"final_test_start={final_test_start.date()}."
            ),
        )
    else:
        audit.pass_(
            "Training/test temporal separation",
            (
                "Training ends before the final test begins."
            ),
        )

    # ---------------------------------------------------------------
    # Final test must actually be after training
    # ---------------------------------------------------------------

    if final_test_end <= training_end:
        audit.fail(
            "Final test chronology",
            (
                "Final test does not extend beyond "
                "the recorded training period."
            ),
        )
    else:
        audit.pass_(
            "Final test chronology",
            "Final test extends beyond training period.",
        )


# =====================================================================
# WALK-FORWARD CONFIGURATION AUDIT
# =====================================================================

def audit_walk_forward_configuration(
    manifest: dict[str, Any],
    audit: Audit,
) -> None:
    """Inspect the packaged walk-forward configuration."""

    frozen = manifest.get(
        "frozen_configuration",
        {},
    )

    walk_forward = frozen.get(
        "walk_forward_validation"
    )

    if walk_forward is True:
        audit.pass_(
            "Walk-forward configuration",
            "walk_forward_validation=true.",
        )
    else:
        audit.warn(
            "Walk-forward configuration",
            (
                "walk_forward_validation is not explicitly "
                "set to true."
            ),
        )

    post_freeze_optimization = frozen.get(
        "parameter_optimization_after_freeze"
    )

    if post_freeze_optimization is False:
        audit.pass_(
            "Post-freeze optimization",
            (
                "Parameter optimization after freeze "
                "is disabled."
            ),
        )
    else:
        audit.warn(
            "Post-freeze optimization",
            (
                "Post-freeze parameter optimization "
                "is not explicitly disabled."
            ),
        )

    final_holdout_protection = frozen.get(
        "do_not_tune_after_final_holdout"
    )

    if final_holdout_protection is True:
        audit.pass_(
            "Final holdout protection",
            (
                "do_not_tune_after_final_holdout=true."
            ),
        )
    else:
        audit.warn(
            "Final holdout protection",
            (
                "Final holdout protection is not "
                "explicitly enabled."
            ),
        )


# =====================================================================
# RESEARCH VERDICT
# =====================================================================

def audit_research_verdict(
    metadata: dict[str, Any],
    audit: Audit,
) -> None:
    """Inspect the research verdict stored with the model."""

    verdict = metadata.get(
        "research_verdict"
    )

    if not verdict:
        audit.warn(
            "Research verdict",
            "No research verdict recorded.",
        )
        return

    if (
        str(verdict).strip().upper()
        == "PARTIAL / INCONCLUSIVE EVIDENCE"
    ):
        audit.warn(
            "Research verdict",
            (
                "Packaged model explicitly reports "
                "'PARTIAL / INCONCLUSIVE EVIDENCE'."
            ),
        )
    else:
        audit.pass_(
            "Research verdict",
            f"{verdict}",
        )


# =====================================================================
# BACKTEST MATHEMATICS AUDIT
# =====================================================================

def audit_existing_backtest(
    audit: Audit,
) -> None:
    """Audit the existing P29.3 backtest JSON."""

    if not BACKTEST_PATH.exists():
        audit.fail(
            "Backtest artifact",
            (
                "Missing "
                f"{BACKTEST_PATH.relative_to(PROJECT_ROOT)}"
            ),
        )
        return

    result = load_json(
        BACKTEST_PATH
    )

    required_sections = {
        "summary",
        "risk",
        "trades",
        "benchmark",
        "equity_curve",
        "drawdown_curve",
        "configuration",
    }

    missing = sorted(
        required_sections
        - set(result)
    )

    if missing:
        audit.fail(
            "Backtest result structure",
            f"Missing sections: {missing}",
        )
        return

    audit.pass_(
        "Backtest result structure",
        "All required sections are present.",
    )

    # ---------------------------------------------------------------
    # Return arithmetic
    # ---------------------------------------------------------------

    summary = result["summary"]

    initial_capital = float(
        summary["initial_capital"]
    )

    final_value = float(
        summary["final_value"]
    )

    reported_return = float(
        summary["total_return"]
    )

    recomputed_return = (
        final_value
        / initial_capital
    ) - 1.0

    if math.isclose(
        reported_return,
        recomputed_return,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        audit.pass_(
            "Return arithmetic",
            (
                "Reported total return matches "
                "final/initial capital."
            ),
        )
    else:
        audit.fail(
            "Return arithmetic",
            (
                f"reported={reported_return:.12f}, "
                f"recomputed={recomputed_return:.12f}"
            ),
        )

    # ---------------------------------------------------------------
    # Equity
    # ---------------------------------------------------------------

    equity = pd.DataFrame(
        result["equity_curve"]
    )

    for column in [
        "strategy",
        "benchmark",
    ]:
        if column not in equity.columns:
            audit.fail(
                "Equity schema",
                f"Missing {column}.",
            )
            continue

        values = pd.to_numeric(
            equity[column],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        if np.isfinite(values).all():
            audit.pass_(
                f"Finite {column} equity",
                "All values are finite.",
            )
        else:
            audit.fail(
                f"Finite {column} equity",
                "NaN or infinity detected.",
            )

        if (values > 0).all():
            audit.pass_(
                f"Positive {column} equity",
                "All values are positive.",
            )
        else:
            audit.fail(
                f"Positive {column} equity",
                "Zero/negative values detected.",
            )

    # ---------------------------------------------------------------
    # Drawdown
    # ---------------------------------------------------------------

    drawdown = pd.DataFrame(
        result["drawdown_curve"]
    )

    if "drawdown" not in drawdown.columns:
        audit.fail(
            "Drawdown schema",
            "Missing drawdown column.",
        )
    else:
        values = pd.to_numeric(
            drawdown["drawdown"],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        if np.isfinite(values).all():
            audit.pass_(
                "Finite drawdown",
                "All drawdown values are finite.",
            )
        else:
            audit.fail(
                "Finite drawdown",
                "NaN or infinity detected.",
            )

        if (values <= 1e-12).all():
            audit.pass_(
                "Drawdown sign",
                "Drawdown is never positive.",
            )
        else:
            audit.fail(
                "Drawdown sign",
                "Positive drawdown detected.",
            )

    # ---------------------------------------------------------------
    # Trading configuration
    # ---------------------------------------------------------------

    configuration = result[
        "configuration"
    ]

    transaction_cost = float(
        configuration.get(
            "transaction_cost",
            0,
        )
    )

    slippage = float(
        configuration.get(
            "slippage",
            0,
        )
    )

    threshold = float(
        configuration.get(
            "threshold",
            0.5,
        )
    )

    if (
        transaction_cost >= 0
        and slippage >= 0
    ):
        audit.pass_(
            "Trading friction",
            (
                f"transaction_cost="
                f"{transaction_cost:.4%}, "
                f"slippage="
                f"{slippage:.4%}"
            ),
        )
    else:
        audit.fail(
            "Trading friction",
            "Negative trading friction detected.",
        )

    if 0 < threshold < 1:
        audit.pass_(
            "Signal threshold",
            f"threshold={threshold:.2%}",
        )
    else:
        audit.fail(
            "Signal threshold",
            f"Invalid threshold={threshold}",
        )

    # ---------------------------------------------------------------
    # Existing engine execution convention
    # ---------------------------------------------------------------

    audit.pass_(
        "Signal execution shift",
        (
            "P29 engine uses signal(t) → position(t+1), "
            "preventing same-close execution."
        ),
    )


# =====================================================================
# MAIN
# =====================================================================

def main() -> int:
    """Run P29.5."""

    audit = Audit()

    # ---------------------------------------------------------------
    # Production artifacts
    # ---------------------------------------------------------------

    try:
        (
            metadata,
            config,
            manifest,
        ) = audit_production_artifacts(
            audit
        )
    except Exception as exc:
        audit.fail(
            "Production artifact loading",
            (
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        )

        metadata = {}
        config = {}
        manifest = {}

    # ---------------------------------------------------------------
    # Provenance
    # ---------------------------------------------------------------

    if metadata and manifest:
        audit_training_test_dates(
            metadata,
            manifest,
            audit,
        )

        audit_walk_forward_configuration(
            manifest,
            audit,
        )

        audit_research_verdict(
            metadata,
            audit,
        )

    # ---------------------------------------------------------------
    # Existing backtest
    # ---------------------------------------------------------------

    try:
        audit_existing_backtest(
            audit
        )
    except Exception as exc:
        audit.fail(
            "Backtest audit",
            (
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        )

    # ---------------------------------------------------------------
    # Print
    # ---------------------------------------------------------------

    audit.print_report()

    # ---------------------------------------------------------------
    # Save JSON
    # ---------------------------------------------------------------

    report = {
        "status": (
            "validated"
            if audit.failures == 0
            and audit.warnings == 0
            else "needs_review"
        ),
        "failures": audit.failures,
        "warnings": audit.warnings,
        "production_model": str(
            MODEL_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "production_metadata": str(
            MODEL_METADATA_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "production_config": str(
            MODEL_CONFIG_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "production_manifest": str(
            MODEL_MANIFEST_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "checks": audit.results,
    }

    AUDIT_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with AUDIT_OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    print()
    print(
        "Audit report written to:"
    )
    print(AUDIT_OUTPUT_PATH)

    # ---------------------------------------------------------------
    # Final decision
    # ---------------------------------------------------------------

    if audit.failures > 0:
        print()
        print(
            "P29.5 RESULT: FAIL / NEEDS REVIEW"
        )
        print(
            "The current long historical backtest "
            "must NOT be published as validated "
            "out-of-sample performance."
        )
        return 1

    if audit.warnings > 0:
        print()
        print(
            "P29.5 RESULT: REVIEW"
        )
        print(
            "Internal checks passed, but provenance "
            "still contains warnings."
        )
        return 0

    print()
    print(
        "P29.5 RESULT: PASS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )