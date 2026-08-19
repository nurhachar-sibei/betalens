from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from betalens.eventstudy.eventstudy import EventStudy, _get_window_prices, _get_window_returns


class FakeDatafeed:
    def __init__(self, prices_by_code: dict[str, pd.Series]) -> None:
        self.prices_by_code = prices_by_code

    def query_time_range(self, *, codes, start_date, end_date, metric):
        prices = self.prices_by_code.get(codes[0], pd.Series(dtype=float))
        if prices.empty:
            return pd.DataFrame(columns=["datetime", "value"])
        mask = (
            (prices.index.normalize() >= pd.Timestamp(start_date))
            & (prices.index.normalize() <= pd.Timestamp(end_date))
        )
        selected = prices.loc[mask]
        return pd.DataFrame({"datetime": selected.index, "value": selected.to_numpy()})


def _calendar(begin_date, end_date, period, exchange="SHSE"):
    assert period == "D"
    return [value.date() for value in pd.bdate_range(begin_date, end_date)]


def test_daily_return_uses_the_day_close_to_close_change() -> None:
    days = pd.DatetimeIndex(["2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"])
    prices = pd.Series([99.0, 100.0, 102.0, 103.0], index=days)

    window = _get_window_returns(
        prices, pd.Timestamp("2024-01-05 10:00:00"), 1, 1, trade_days=days
    )

    assert window is not None
    assert np.isclose(window.loc[-1], 100.0 / 99.0 - 1)
    assert np.isclose(window.loc[0], 0.02)
    assert np.isclose(window.loc[1], 103.0 / 102.0 - 1)


def test_price_window_is_aligned_to_zero_on_day0() -> None:
    days = pd.DatetimeIndex(["2024-01-04", "2024-01-05", "2024-01-08"])
    prices = pd.Series([100.0, 102.0, 103.0], index=days)

    window = _get_window_prices(
        prices, pd.Timestamp("2024-01-05 10:00:00"), 1, 1, 15, days
    )

    assert window is not None
    assert np.isclose(window.loc[0], 0.0)
    assert np.isclose(window.loc[-1], 100.0 / 102.0 - 1)
    assert np.isclose(window.loc[1], 103.0 / 102.0 - 1)


def test_holding_returns_are_always_produced_without_mode() -> None:
    returns = pd.DataFrame({0: {0: 0.01, 1: 0.02, 2: 0.03}})
    matrix, stats = EventStudy(None)._calc_holding_returns(
        returns, {"days": [1, 2], "months": []}
    )

    assert list(stats["holding_period"]) == ["1日", "2日"]
    assert np.isclose(matrix.loc[1, 0], 0.02)
    assert np.isclose(matrix.loc[2, 0], 1.02 * 1.03 - 1)


def test_analyze_returns_daily_holding_and_price_outputs_only() -> None:
    days = pd.bdate_range("2024-01-01", periods=12)
    prices = pd.Series(np.arange(100.0, 112.0), index=days)
    datafeed = FakeDatafeed({"TEST": prices})
    events = pd.Series([1], index=pd.DatetimeIndex(["2024-01-05 10:00:00"]))

    with patch("betalens.eventstudy.eventstudy.get_absolute_trade_days", side_effect=_calendar):
        result = EventStudy(datafeed).analyze(
            events,
            code="TEST",
            window_before=1,
            window_after=1,
            holding_periods={"days": [1, 2], "months": []},
        )

    assert "cumulative_stats" not in result
    assert "cumulative_returns_matrix" not in result
    assert {"daily_stats", "holding_stats", "price_matrix"}.issubset(result)
    assert np.isclose(result["price_matrix"].loc[0, 0], 0.0)
    assert list(result["holding_stats"]["holding_period"]) == ["1日", "2日"]


def test_insufficient_event_window_is_skipped_without_calendar_error() -> None:
    calendar = pd.DatetimeIndex(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
    )
    prices = pd.Series([100.0, 101.0, 102.0, 103.0], index=calendar[:4])
    datafeed = FakeDatafeed({"TEST": prices})
    events = pd.Series(
        [1, 1],
        index=pd.DatetimeIndex(["2024-01-04 10:00:00", "2024-01-08 10:00:00"]),
    )

    with patch(
        "betalens.eventstudy.eventstudy.get_absolute_trade_days",
        return_value=list(calendar.date),
    ):
        result = EventStudy(datafeed).analyze(
            events,
            code="TEST",
            window_before=1,
            window_after=1,
            holding_periods={"days": [], "months": []},
        )

    assert result["event_count"] == 1
    assert list(result["event_dates"]) == list(events.index)
    assert list(result["returns_matrix"].columns) == [0]


def test_all_insufficient_event_windows_return_empty_result_without_error() -> None:
    calendar = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"])
    events = pd.Series(
        [1], index=pd.DatetimeIndex(["2024-01-04 10:00:00"])
    )

    with patch(
        "betalens.eventstudy.eventstudy.get_absolute_trade_days",
        return_value=list(calendar.date),
    ):
        result = EventStudy(FakeDatafeed({})).analyze(
            events,
            code="TEST",
            window_before=1,
            window_after=1,
            holding_periods={"days": [1], "months": []},
        )

    assert "error" not in result
    assert result["event_count"] == 0
    assert result["trade_days"].empty
    assert result["returns_matrix"].empty
