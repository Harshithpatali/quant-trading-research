"""
Daily production prediction job.

P21 runs after the daily market-data update has completed successfully.

Responsibilities:
    1. Load the latest validated S&P 500 history from Supabase.
    2. Generate the prediction through PredictionService.
    3. Validate the returned production response.
    4. Write a JSON prediction record to logs/latest_prediction.json.
    5. Print the result for GitHub Actions.

This script does NOT retrain or modify the frozen model.

Persistence note
----------------
The prediction is written to a JSON artifact for the current GitHub
Actions run. A dedicated Supabase prediction-history table will be added
only after its schema is explicitly defined; this avoids inventing a
database schema or writing prediction records into sp500_daily.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Make direct execution work from:
# D:\quant-trading-research\scripts\daily_prediction.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services.prediction_service import predict_current_market
from src.utils.logging_config import configure_logging


LOG_DIR = PROJECT_ROOT / "logs"
PREDICTION_FILE = LOG_DIR / "latest_prediction.json"

logger = configure_logging(level="INFO")


def validate_prediction_result(
    result: dict[str, Any],
) -> None:
    """Validate the minimum production prediction contract."""

    required_keys = {
        "status",
        "prediction_date",
        "signal",
        "prediction",
        "position",
        "probability_up",
        "probability_down",
        "threshold",
        "confidence",
        "model",
        "data",
    }

    missing = required_keys.difference(
        result.keys()
    )

    if missing:
        raise RuntimeError(
            "Prediction response is missing required fields: "
            + ", ".join(sorted(missing))
        )

    if result["status"] != "success":
        raise RuntimeError(
            f"Prediction service returned status={result['status']!r}."
        )

    if result["signal"] not in {
        "LONG",
        "CASH",
    }:
        raise RuntimeError(
            f"Unsupported production signal: {result['signal']!r}"
        )

    if result["prediction"] not in {
        0,
        1,
    }:
        raise RuntimeError(
            "Production prediction must be 0 or 1."
        )

    probability_up = float(
        result["probability_up"]
    )

    probability_down = float(
        result["probability_down"]
    )

    threshold = float(
        result["threshold"]
    )

    confidence = float(
        result["confidence"]
    )

    for name, value in {
        "probability_up": probability_up,
        "probability_down": probability_down,
        "threshold": threshold,
        "confidence": confidence,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise RuntimeError(
                f"{name} is outside [0, 1]: {value}"
            )

    if abs(
        probability_up
        + probability_down
        - 1.0
    ) > 1e-12:
        raise RuntimeError(
            "Prediction probabilities do not sum to 1."
        )

    expected_prediction = int(
        probability_up >= threshold
    )

    if result["prediction"] != expected_prediction:
        raise RuntimeError(
            "Prediction does not match the frozen probability threshold."
        )

    expected_signal = (
        "LONG"
        if expected_prediction == 1
        else "CASH"
    )

    if result["signal"] != expected_signal:
        raise RuntimeError(
            "Signal does not match the prediction."
        )

    data = result["data"]

    if data.get("validation_passed") is not True:
        raise RuntimeError(
            "Prediction was generated from data that did not pass "
            "production validation."
        )


def write_prediction_record(
    result: dict[str, Any],
) -> Path:
    """Write the latest prediction as a JSON artifact."""

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    record = {
        "recorded_at": datetime.now(
            timezone.utc
        ).isoformat(),
        **result,
    }

    PREDICTION_FILE.write_text(
        json.dumps(
            record,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return PREDICTION_FILE


def run_daily_prediction() -> dict[str, Any]:
    """Generate, validate, and record the current prediction."""

    logger.info(
        "Starting daily production prediction."
    )

    result = predict_current_market()

    validate_prediction_result(
        result
    )

    prediction_file = write_prediction_record(
        result
    )

    logger.info(
        (
            "Daily prediction completed | "
            "date=%s | signal=%s | probability_up=%.6f | "
            "threshold=%.6f"
        ),
        result["prediction_date"],
        result["signal"],
        result["probability_up"],
        result["threshold"],
    )

    logger.info(
        "Prediction record written to %s.",
        prediction_file,
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    return result


def main() -> int:
    """CLI entry point."""

    try:
        run_daily_prediction()
        return 0

    except Exception as exc:
        logger.exception(
            "Daily production prediction failed."
        )

        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
