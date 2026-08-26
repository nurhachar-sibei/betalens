from __future__ import annotations

import numpy as np
import pandas as pd

from betalens.backtest import BacktestBase


def test_buy_and_hold_nav_matches_normalized_price_series() -> None:
    """A zero-cost buy-and-hold portfolio should track the stock price exactly."""
    dates = pd.date_range("2024-01-02", periods=5, freq="D")
    prices = pd.Series(
        [100.0, 110.0, 90.0, 120.0, 130.0],
        index=dates,
        name="000001.SZ",
    )
    price_panel = prices.to_frame()

    # Repeating a 100% target weight is still buy-and-hold here: the initial
    # amount buys an exact number of lots, so every later rebalance is a no-op.
    weights = pd.DataFrame(
        {"000001.SZ": 1.0, "cash": 0.0},
        index=dates,
    )
    backtest = BacktestBase(
        weights,
        symbol="buy_and_hold",
        amount=100_000.0,
        lot_size=100,
        check_trade_status=False,
        ftc=0.0,
        ptc=0.0,
        verbose=False,
        preloaded_cost_price=price_panel,
        preloaded_close_price=price_panel,
    )

    expected_nav = prices / prices.iloc[0]
    expected_nav.index = expected_nav.index.rename(backtest.nav.index.name)

    pd.testing.assert_index_equal(backtest.nav.index, expected_nav.index)
    np.testing.assert_allclose(
        backtest.nav.to_numpy(),
        expected_nav.to_numpy(),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        backtest.position["000001.SZ"].to_numpy(),
        np.full(len(dates), 1_000.0),
        rtol=0.0,
        atol=1e-10,
    )
    assert len(backtest.rebalance_log) == 1
    assert backtest.rebalance_log.iloc[0]["direction"] == "buy"
