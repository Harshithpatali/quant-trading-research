from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import pandas as pd
import requests
import yfinance as yf

from src.config import (
    SUPABASE_KEY,
    SUPABASE_TABLE,
    SUPABASE_URL,
)
from src.data.validator import validate_ohlcv
from src.utils.logging_config import configure_logging


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

YAHOO_TICKER = "^GSPC"

SUPABASE_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
]

CANONICAL_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj.Close",
    "Volume",
]

logger = configure_logging(
    level="INFO"
)


# ---------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------

def validate_configuration() -> None:
    """Ensure the required production configuration exists."""

    missing = []

    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")

    if not SUPABASE_KEY:
        missing.append("SUPABASE_KEY")

    if not SUPABASE_TABLE:
        missing.append("SUPABASE_TABLE")

    if missing:
        raise RuntimeError(
            "Missing required environment configuration: "
            + ", ".join(missing)
        )

    if not SUPABASE_URL.startswith(
        "https://"
    ):
        raise RuntimeError(
            "SUPABASE_URL must use HTTPS."
        )


# ---------------------------------------------------------------------
# Yahoo Finance download
# ---------------------------------------------------------------------

def download_latest_market_data() -> pd.DataFrame:
    """
    Download recent S&P 500 daily OHLCV observations.

    A short recent window is sufficient because this job only needs to
    discover newly available observations. Historical backfill is
    intentionally handled separately from the daily production job.
    """

    logger.info(
        "Downloading latest %s market data.",
        YAHOO_TICKER,
    )

    raw = yf.download(
        YAHOO_TICKER,
        period="10d",
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=False,
    )

    if raw is None or raw.empty:
        raise RuntimeError(
            "Yahoo Finance returned no market data."
        )

    # yfinance may return a MultiIndex even for a single ticker.
    if isinstance(
        raw.columns,
        pd.MultiIndex,
    ):
        if YAHOO_TICKER in raw.columns.get_level_values(-1):
            raw = raw.xs(
                YAHOO_TICKER,
                axis=1,
                level=-1,
            )
        elif YAHOO_TICKER in raw.columns.get_level_values(0):
            raw = raw.xs(
                YAHOO_TICKER,
                axis=1,
                level=0,
            )

    rename_map = {
        "Date": "Date",
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Adj Close": "Adj.Close",
        "Adj.Close": "Adj.Close",
        "Volume": "Volume",
    }

    raw = raw.rename(
        columns=rename_map
    )

    raw.index.name = "Date"

    if "Date" not in raw.columns:
        raw = raw.reset_index()

    missing = [
        column
        for column in CANONICAL_COLUMNS
        if column not in raw.columns
    ]

    if missing:
        raise RuntimeError(
            "Yahoo Finance response is missing columns: "
            + ", ".join(missing)
        )

    data = raw[
        CANONICAL_COLUMNS
    ].copy()

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce",
    ).dt.tz_localize(None)

    for column in CANONICAL_COLUMNS[1:]:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = (
        data
        .dropna(subset=["Date"])
        .sort_values("Date")
        .drop_duplicates(
            subset=["Date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    logger.info(
        "Downloaded %d observations: %s to %s.",
        len(data),
        data["Date"].min().strftime("%Y-%m-%d"),
        data["Date"].max().strftime("%Y-%m-%d"),
    )

    return data


# ---------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------

def supabase_headers() -> dict[str, str]:
    """Return headers required by the Supabase REST endpoint."""
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=minimal",
    }


def supabase_table_url() -> str:
    """Build the PostgREST table endpoint."""
    return (
        SUPABASE_URL.rstrip("/")
        + "/rest/v1/"
        + SUPABASE_TABLE
    )


def convert_to_supabase_records(
    data: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Convert canonical OHLCV rows to the actual Supabase schema."""

    records: list[dict[str, Any]] = []

    for row in data.itertuples(
        index=False
    ):
        records.append(
            {
                "date": row.Date.strftime(
                    "%Y-%m-%d"
                ),
                "open": float(row.Open),
                "high": float(row.High),
                "low": float(row.Low),
                "close": float(row.Close),
                "adj_close": float(row._5),
                "volume": int(row.Volume),
            }
        )

    return records


def upsert_records(
    records: list[dict[str, Any]],
) -> None:
    """Upsert validated records into Supabase."""

    if not records:
        logger.info(
            "No new records to upsert."
        )
        return

    url = (
        supabase_table_url()
        + "?on_conflict=date"
    )

    logger.info(
        "Upserting %d observations into Supabase table %s.",
        len(records),
        SUPABASE_TABLE,
    )

    response = requests.post(
        url,
        headers=supabase_headers(),
        json=records,
        timeout=60,
    )

    if response.status_code not in {
        200,
        201,
        204,
    }:
        raise RuntimeError(
            "Supabase upsert failed with HTTP "
            f"{response.status_code}: "
            f"{response.text[:1000]}"
        )

    logger.info(
        "Supabase upsert completed successfully."
    )


def fetch_latest_supabase_rows(
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Fetch the latest rows for post-write verification."""

    url = (
        supabase_table_url()
        + f"?select=date,open,high,low,close,adj_close,volume"
        + f"&order=date.desc&limit={int(limit)}"
    )

    response = requests.get(
        url,
        headers=supabase_headers(),
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(
            "Supabase verification request failed with HTTP "
            f"{response.status_code}: "
            f"{response.text[:1000]}"
        )

    payload = response.json()

    if not isinstance(
        payload,
        list,
    ):
        raise RuntimeError(
            "Supabase verification returned an unexpected response."
        )

    return payload


# ---------------------------------------------------------------------
# Post-write verification
# ---------------------------------------------------------------------

def verify_supabase_data(
    expected_dates: set[str],
) -> None:
    """
    Verify that the dates written by this job are present in Supabase.
    """

    latest_rows = fetch_latest_supabase_rows(
        limit=max(
            10,
            len(expected_dates) + 2,
        )
    )

    actual_dates = {
        str(row["date"])
        for row in latest_rows
        if "date" in row
    }

    missing_dates = (
        expected_dates - actual_dates
    )

    if missing_dates:
        raise RuntimeError(
            "Supabase verification failed. "
            "Missing expected dates: "
            + ", ".join(
                sorted(missing_dates)
            )
        )

    logger.info(
        "Supabase verification passed for %d dates.",
        len(expected_dates),
    )


# ---------------------------------------------------------------------
# Main production job
# ---------------------------------------------------------------------

def run_daily_update() -> dict[str, Any]:
    """Execute the complete daily market-data update."""

    started_at = datetime.now(
        timezone.utc
    )

    validate_configuration()

    data = download_latest_market_data()

    # The production validator rejects malformed OHLCV data before
    # anything reaches Supabase.
    validation = validate_ohlcv(
        data,
        source_name="Yahoo Finance latest S&P 500 data",
        raise_on_error=True,
        require_all_columns=True,
        require_sorted_dates=True,
        require_unique_dates=True,
    )

    records = convert_to_supabase_records(
        data
    )

    expected_dates = {
        record["date"]
        for record in records
    }

    upsert_records(
        records
    )

    verify_supabase_data(
        expected_dates
    )

    completed_at = datetime.now(
        timezone.utc
    )

    result = {
        "status": "success",
        "source": YAHOO_TICKER,
        "table": SUPABASE_TABLE,
        "rows_downloaded": len(data),
        "rows_upserted": len(records),
        "date_start": validation[
            "date_start"
        ],
        "date_end": validation[
            "date_end"
        ],
        "validation_passed": validation[
            "valid"
        ],
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
    }

    logger.info(
        (
            "Daily market-data update completed | "
            "rows=%d | date_end=%s"
        ),
        len(records),
        validation["date_end"],
    )

    return result


def main() -> int:
    """CLI entry point."""
    try:
        result = run_daily_update()
        print(result)
        return 0

    except Exception as exc:
        logger.exception(
            "Daily market-data update failed."
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
