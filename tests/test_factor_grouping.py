from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from betalens.factor.factor import get_single_factor_weight, single_characteristic


FACTOR_ROOT = Path(__file__).parents[1] / "betalens-factor"
if str(FACTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(FACTOR_ROOT))

from factor_template import group_balance_statistics  # noqa: E402


def _sample() -> pd.DataFrame:
    values = [1, 1, 1, 1, 2, 2, 3, 3, 4, 5]
    return pd.DataFrame({
        "input_ts": [pd.Timestamp("2024-01-05")] * len(values),
        "code": [f"{index:06d}.SZ" for index in range(len(values))],
        "characteristic": values,
    })


def test_value_grouping_never_splits_equal_values_and_may_reduce_groups():
    labeled = single_characteristic(
        _sample(),
        "characteristic",
        {"characteristic": 5},
        grouping_mode="value",
    ).reset_index()

    assert labeled["characteristic_label"].nunique() == 4
    assert labeled.groupby("characteristic")["characteristic_label"].nunique().max() == 1
    assert sorted(labeled["characteristic_label"].unique()) == [0, 1, 2, 3]


def test_equal_count_grouping_has_exact_group_count_and_balanced_sizes():
    labeled = single_characteristic(
        _sample(),
        "characteristic",
        {"characteristic": 5},
        grouping_mode="equal_count",
    ).reset_index()

    counts = labeled.groupby("characteristic_label").size()
    assert list(counts.index) == [0, 1, 2, 3, 4]
    assert counts.max() - counts.min() <= 1
    assert labeled.groupby("characteristic")["characteristic_label"].nunique().max() > 1


def test_equal_count_grouping_rejects_too_few_stocks():
    with pytest.raises(ValueError, match="无法严格生成 5 个等额分组"):
        single_characteristic(
            _sample().head(4),
            "characteristic",
            {"characteristic": 5},
            grouping_mode="equal_count",
        )


def test_classic_long_short_uses_actual_extreme_labels_for_value_groups():
    labeled = single_characteristic(
        _sample(),
        "characteristic",
        {"characteristic": 5},
        grouping_mode="value",
    )
    weights = get_single_factor_weight(labeled, {
        "factor_key": "characteristic",
        "mode": "classic-long-short",
        "long": [4],
        "short": [0],
        "grouping_mode": "value",
    })
    row = weights.iloc[0]
    positive_codes = set(row[row > 0].index)
    negative_codes = set(row[row < 0].index)

    assert positive_codes == {"000008.SZ", "000009.SZ"}
    assert negative_codes == {"000000.SZ", "000001.SZ", "000002.SZ", "000003.SZ"}
    assert row[row > 0].sum() == pytest.approx(1.0)
    assert row[row < 0].sum() == pytest.approx(-1.0)


def test_freeplay_selector_compatibility_depends_on_grouping_mode():
    value_labeled = single_characteristic(
        _sample(), "characteristic", {"characteristic": 5}, grouping_mode="value"
    )
    with pytest.raises(ValueError, match="必须使用 'max'/'min'"):
        get_single_factor_weight(value_labeled, {
            "factor_key": "characteristic",
            "mode": "freeplay",
            "long": [4],
            "short": [0],
            "grouping_mode": "value",
        })

    value_weights = get_single_factor_weight(value_labeled, {
        "factor_key": "characteristic",
        "mode": "freeplay",
        "long": ["max"],
        "short": ["min"],
        "grouping_mode": "value",
    })
    assert value_weights.iloc[0].gt(0).sum() == 2
    assert value_weights.iloc[0].lt(0).sum() == 4

    equal_labeled = single_characteristic(
        _sample(), "characteristic", {"characteristic": 5}, grouping_mode="equal_count"
    )
    equal_weights = get_single_factor_weight(equal_labeled, {
        "factor_key": "characteristic",
        "mode": "freeplay",
        "long": [4],
        "short": [0],
        "grouping_mode": "equal_count",
    })
    assert equal_weights.iloc[0].gt(0).sum() == 2
    assert equal_weights.iloc[0].lt(0).sum() == 2


def test_group_balance_profiling_reports_count_and_value_separation():
    labeled = single_characteristic(
        _sample(), "characteristic", {"characteristic": 5}, grouping_mode="equal_count"
    )
    by_date, summary = group_balance_statistics(labeled, "characteristic")
    row = by_date.iloc[0]

    assert row["actual_groups"] == 5
    assert row["group_count_range"] <= 1
    assert row["same_value_boundary_count"] >= 1
    assert not bool(row["value_separation_sufficient"])
    assert summary.iloc[0]["count_balanced_ratio"] == pytest.approx(1.0)
