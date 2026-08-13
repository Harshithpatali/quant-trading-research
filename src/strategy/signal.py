"""
Production trading-signal generation.

This module converts the frozen model's probability output into the
application's trading decision.

Important:
- The model itself is not modified here.
- The research threshold is read from the frozen configuration by
  ModelPredictor.
- No new optimization or threshold tuning is performed.
- This module is intentionally deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


VALID_SIGNALS = {"LONG", "CASH"}


@dataclass(frozen=True)
class TradingSignal:
    """Immutable production trading decision."""

    signal: str
    prediction: int
    probability_up: float
    probability_down: float
    threshold: float
    confidence: float
    model_type: str | None = None
    feature_date: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "signal": self.signal,
            "prediction": self.prediction,
            "probability_up": self.probability_up,
            "probability_down": self.probability_down,
            "threshold": self.threshold,
            "confidence": self.confidence,
            "model_type": self.model_type,
            "feature_date": self.feature_date,
        }


def _validate_probability(
    probability_up: float,
) -> float:
    """Validate and normalize the model probability."""
    try:
        probability = float(probability_up)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "probability_up must be numeric."
        ) from exc

    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "probability_up must be between 0 and 1."
        )

    return probability


def _validate_threshold(
    threshold: float,
) -> float:
    """Validate the frozen decision threshold."""
    try:
        value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "threshold must be numeric."
        ) from exc

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    return value


def generate_signal(
    probability_up: float,
    threshold: float,
    *,
    model_type: str | None = None,
    feature_date: str | None = None,
) -> TradingSignal:
    """
    Convert model probability into a deterministic trading signal.

    Decision rule:

        probability_up >= threshold  -> LONG
        probability_up <  threshold  -> CASH

    The threshold is supplied by the frozen model configuration.
    This function does not optimize or change it.
    """
    probability = _validate_probability(
        probability_up
    )

    frozen_threshold = _validate_threshold(
        threshold
    )

    probability_down = 1.0 - probability

    prediction = int(
        probability >= frozen_threshold
    )

    signal = (
        "LONG"
        if prediction == 1
        else "CASH"
    )

    # Confidence is the distance from a 50/50 probability.
    # It is descriptive only and does not affect the decision.
    confidence = abs(
        probability - 0.5
    ) * 2.0

    return TradingSignal(
        signal=signal,
        prediction=prediction,
        probability_up=probability,
        probability_down=probability_down,
        threshold=frozen_threshold,
        confidence=confidence,
        model_type=model_type,
        feature_date=feature_date,
    )


def signal_from_prediction(
    prediction_result: Mapping[str, Any],
) -> TradingSignal:
    """
    Convert the dictionary returned by ModelPredictor into a signal.

    The predictor's probability and frozen threshold are used as the
    source of truth. The supplied `signal` and `prediction` fields are
    independently recomputed rather than blindly trusted.
    """
    if "probability_up" not in prediction_result:
        raise ValueError(
            "Prediction result is missing probability_up."
        )

    if "threshold" not in prediction_result:
        raise ValueError(
            "Prediction result is missing threshold."
        )

    return generate_signal(
        probability_up=prediction_result[
            "probability_up"
        ],
        threshold=prediction_result[
            "threshold"
        ],
        model_type=prediction_result.get(
            "model_type"
        ),
        feature_date=prediction_result.get(
            "feature_date"
        ),
    )


def validate_signal(
    signal: TradingSignal,
) -> None:
    """Validate a generated production signal."""
    if signal.signal not in VALID_SIGNALS:
        raise ValueError(
            f"Unsupported signal: {signal.signal!r}"
        )

    if signal.prediction not in {0, 1}:
        raise ValueError(
            "Prediction must be 0 or 1."
        )

    if signal.signal == "LONG" and signal.prediction != 1:
        raise ValueError(
            "LONG signal must have prediction=1."
        )

    if signal.signal == "CASH" and signal.prediction != 0:
        raise ValueError(
            "CASH signal must have prediction=0."
        )

    if abs(
        signal.probability_up +
        signal.probability_down -
        1.0
    ) > 1e-12:
        raise ValueError(
            "Probability values must sum to 1."
        )

    expected_prediction = int(
        signal.probability_up >=
        signal.threshold
    )

    if signal.prediction != expected_prediction:
        raise ValueError(
            "Signal does not match the frozen threshold rule."
        )


def get_position_size(
    signal: TradingSignal,
) -> float:
    """
    Return the binary portfolio position represented by the signal.

    LONG = 1.0
    CASH = 0.0

    This is deliberately not a position-sizing optimizer.
    Risk sizing can be added later as a separate production component.
    """
    validate_signal(signal)

    return 1.0 if signal.signal == "LONG" else 0.0


def signal_summary(
    signal: TradingSignal,
) -> str:
    """Create a concise human-readable signal summary."""
    validate_signal(signal)

    return (
        f"{signal.signal} | "
        f"P(up)={signal.probability_up:.4f} | "
        f"threshold={signal.threshold:.4f} | "
        f"confidence={signal.confidence:.2%}"
    )


__all__ = [
    "VALID_SIGNALS",
    "TradingSignal",
    "generate_signal",
    "signal_from_prediction",
    "validate_signal",
    "get_position_size",
    "signal_summary",
]
