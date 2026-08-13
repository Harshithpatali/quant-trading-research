from __future__ import annotations

from datetime import date, datetime
from typing import Optional

import pandas as pd
import requests

from src.config import SUPABASE_KEY, SUPABASE_TABLE, SUPABASE_URL

SUPABASE_COLUMNS = ["date", "open", "high", "low", "close", "adj_close", "volume"]
EXPECTED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Adj.Close", "Volume"]
COLUMN_MAPPING = dict(zip(SUPABASE_COLUMNS, EXPECTED_COLUMNS))
NUMERIC_COLUMNS = ["Open", "High", "Low", "Close", "Adj.Close", "Volume"]
DEFAULT_PAGE_SIZE = 1000
REQUEST_TIMEOUT = 30

class SupabaseDataLoader:
    """Load S&P 500 OHLCV data from Supabase."""
    def __init__(self, table=SUPABASE_TABLE, supabase_url=SUPABASE_URL,
                 supabase_key=SUPABASE_KEY, page_size=DEFAULT_PAGE_SIZE,
                 timeout=REQUEST_TIMEOUT):
        self.table = table
        self.supabase_url = supabase_url.rstrip("/")
        self.supabase_key = supabase_key
        self.page_size = page_size
        self.timeout = timeout
        if not self.table:
            raise ValueError("Supabase table name cannot be empty.")
        if not self.supabase_url.startswith("https://"):
            raise ValueError("Supabase URL must use HTTPS.")
        if not self.supabase_key:
            raise ValueError("Supabase key is not configured.")

    @property
    def endpoint(self):
        return f"{self.supabase_url}/rest/v1/{self.table}"

    @property
    def headers(self):
        return {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Accept": "application/json",
        }

    def _request_page(self, offset, start_date=None, end_date=None):
        params = {
            "select": ",".join(SUPABASE_COLUMNS),
            "order": "date.asc",
            "limit": self.page_size,
            "offset": offset,
        }
        if start_date and end_date:
            params["and"] = f"(date.gte.{start_date},date.lte.{end_date})"
        elif start_date:
            params["date"] = f"gte.{start_date}"
        elif end_date:
            params["date"] = f"lte.{end_date}"

        try:
            response = requests.get(self.endpoint, headers=self.headers,
                                    params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            raise ConnectionError("Unable to connect to the Supabase data service.") from exc
        if not response.ok:
            raise RuntimeError(
                f"Supabase data request failed with HTTP {response.status_code}: {response.text[:500]}"
            )
        payload = response.json()
        if not isinstance(payload, list):
            raise RuntimeError("Supabase returned an unexpected response format.")
        return payload

    @staticmethod
    def _normalise(rows):
        if not rows:
            return pd.DataFrame(columns=EXPECTED_COLUMNS)
        df = pd.DataFrame(rows)
        missing = [c for c in SUPABASE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError("Supabase table is missing required columns: " + ", ".join(missing))
        df = df[SUPABASE_COLUMNS].rename(columns=COLUMN_MAPPING)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        for column in NUMERIC_COLUMNS:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        return (df.sort_values("Date")
                  .drop_duplicates(subset=["Date"], keep="last")
                  .reset_index(drop=True))

    def load(self, start_date: Optional[str | date | datetime] = None,
             end_date: Optional[str | date | datetime] = None):
        start = self._format_date(start_date) if start_date is not None else None
        end = self._format_date(end_date) if end_date is not None else None
        if start and end and start > end:
            raise ValueError("start_date cannot be later than end_date.")
        rows, offset = [], 0
        while True:
            page = self._request_page(offset, start, end)
            if not page:
                break
            rows.extend(page)
            if len(page) < self.page_size:
                break
            offset += self.page_size
        return self._normalise(rows)

    def load_latest(self, limit=253):
        if limit < 1:
            raise ValueError("limit must be greater than zero.")
        params = {"select": ",".join(SUPABASE_COLUMNS), "order": "date.desc", "limit": limit}
        try:
            response = requests.get(self.endpoint, headers=self.headers,
                                    params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            raise ConnectionError("Unable to connect to the Supabase data service.") from exc
        if not response.ok:
            raise RuntimeError(
                f"Supabase latest-data request failed with HTTP {response.status_code}: {response.text[:500]}"
            )
        payload = response.json()
        if not isinstance(payload, list):
            raise RuntimeError("Supabase returned an unexpected response format.")
        return self._normalise(payload).sort_values("Date").tail(limit).reset_index(drop=True)

    def count_rows(self):
        headers = {**self.headers, "Prefer": "count=exact"}
        params = {"select": "date", "limit": 1}
        try:
            response = requests.get(self.endpoint, headers=headers,
                                    params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            raise ConnectionError("Unable to connect to the Supabase data service.") from exc
        if not response.ok:
            raise RuntimeError(
                f"Supabase row-count request failed with HTTP {response.status_code}: {response.text[:500]}"
            )
        content_range = response.headers.get("Content-Range")
        if not content_range or "/" not in content_range:
            raise RuntimeError("Supabase did not return a valid Content-Range header.")
        total = content_range.rsplit("/", 1)[1]
        if total == "*":
            raise RuntimeError("Supabase returned an unknown row count.")
        return int(total)

    @staticmethod
    def _format_date(value):
        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%d")
        if isinstance(value, date):
            return value.strftime("%Y-%m-%d")
        if isinstance(value, str):
            parsed = pd.to_datetime(value, errors="coerce")
            if pd.isna(parsed):
                raise ValueError(f"Invalid date value: {value!r}")
            return parsed.strftime("%Y-%m-%d")
        raise TypeError("Date must be a string, date, or datetime.")

def load_sp500_data(start_date=None, end_date=None):
    return SupabaseDataLoader().load(start_date=start_date, end_date=end_date)

def load_latest_sp500_data(limit=253):
    return SupabaseDataLoader().load_latest(limit=limit)
