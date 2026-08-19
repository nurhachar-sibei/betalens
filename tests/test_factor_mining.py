from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

import betalens.factor.mining as mining
from betalens.factor.mining_cache import CacheRequest, MiningCache
from betalens.factor.mining_optuna import (
    create_coarse_study,
    create_fine_grid_study,
    generate_fine_candidates,
    suggest_params,
    tell_trial,
)


def test_optuna_search_supports_log_and_composite_categories():
    specs = {
        "window": {"type": "int", "low": 10, "high": 80, "step": 1, "scale": "log"},
        "pair": {"type": "categorical", "choices": [[9, 4], [7, 4]]},
    }
    study = create_coarse_study({"sampler": "random", "seed": 7})
    trial = study.ask()
    params = suggest_params(trial, specs)
    tell_trial(study, trial, 1.0)
    assert 10 <= params["window"] <= 80
    assert params["pair"] in [[9, 4], [7, 4]]

    candidates = generate_fine_candidates(
        specs,
        [{"window": 20, "pair": [9, 4]}],
        {"points_per_dimension": 3, "max_candidates": 6},
        coarse_candidates=[
            {"window": 10, "pair": [9, 4]},
            {"window": 20, "pair": [7, 4]},
            {"window": 40, "pair": [9, 4]},
            {"window": 80, "pair": [7, 4]},
        ],
    )
    assert len(candidates) <= 6
    assert all(10 <= row["window"] <= 40 for row in candidates)

    grid = create_fine_grid_study({"window": [20, 40], "pair": [[9, 4], [7, 4]]})
    categorical_specs = {
        "window": {"type": "categorical", "choices": [20, 40]},
        "pair": specs["pair"],
    }
    seen = []
    for _ in range(4):
        trial = grid.ask()
        seen.append(suggest_params(trial, categorical_specs))
        tell_trial(grid, trial, 0.0)
    assert len({str(row) for row in seen}) == 4


def test_cache_open_or_build_and_slice(tmp_path):
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    values = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0]}, index=index)
    calls = 0

    def builder():
        nonlocal calls
        calls += 1
        return {
            "inputs": {"x": values},
            "price": values,
            "execution_price": values,
            "trade_status": pd.DataFrame(1, index=index, columns=["A"]),
            "industry_by_scheme": {},
            "pit": {day.date(): {"A"} for day in index},
            "universe": ["A"],
            "metadata": {"version": "test"},
        }

    request = CacheRequest(tmp_path, "signature")
    first = MiningCache.open_or_build(request, builder)
    second = MiningCache.open_or_build(request, builder)
    assert calls == 1
    pd.testing.assert_frame_equal(
        first.load("x", "2024-01-02", "2024-01-03"),
        values.loc["2024-01-02":"2024-01-03"],
        check_freq=False,
    )
    assert second.universe == ["A"]
    (tmp_path / "READY.json").write_text("not-json", encoding="utf-8")
    rebuilt = MiningCache.open_or_build(request, builder)
    assert calls == 2
    assert rebuilt.load("price").shape == values.shape


def test_selected_nav_publication_keeps_only_selected_candidate(tmp_path):
    staging = tmp_path / ".nav_staging" / "run-id"
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    nav = pd.Series([1.0, 1.1, 1.2], index=index)
    mining._persist_nav(nav, "selected", "full", {"_nav_staging": str(staging)})
    mining._persist_nav(nav, "filtered", "full", {"_nav_staging": str(staging)})
    published = mining._publish_selected_nav(staging, tmp_path, {"selected"})
    assert published is not None
    assert (published / "selected" / "full.npy").exists()
    assert not (published / "filtered").exists()
    payload = np.load(published / "selected" / "full.npy", allow_pickle=False)
    assert payload["nav"].tolist() == [1.0, 1.1, 1.2]


def _synthetic_data() -> mining.MiningData:
    index = pd.date_range("2024-01-01", periods=15, freq="D")
    return mining.MiningData(
        inputs={"x": pd.DataFrame({"A": range(15)}, index=index)},
        price=pd.DataFrame({"A": range(100, 115)}, index=index),
        execution_price=pd.DataFrame({"A": range(100, 115)}, index=index),
        trade_status=None,
        pit=None,
        universe=["A"],
    )


def test_execution_modes_compute_once_and_isolate_windows(monkeypatch):
    data = _synthetic_data()
    windows = [
        mining.MiningWindow("4/4/0", "2024-01-05", "2024-01-08", 4, 4),
        mining.MiningWindow("4/4/1", "2024-01-09", "2024-01-12", 4, 4),
    ]
    calls = {"compute": 0, "nav": 0}

    class Factor:
        name = "fake"
        compute_kwargs = {}
        weight_mode = "classic-long-short"

        def compute(self, **kwargs):
            calls["compute"] += 1
            return kwargs["x"]

    precomputed = types.ModuleType("_mining_precomputed_test")
    precomputed.make_mining_spec = lambda params: mining.MiningSpec(Factor())
    monkeypatch.setitem(sys.modules, precomputed.__name__, precomputed)
    index = data.price.index
    monkeypatch.setattr(
        mining,
        "_build_weights",
        lambda *args, **kwargs: pd.DataFrame({"A": 1.0}, index=index),
    )

    def vector_nav(weights, price):
        calls["nav"] += 1
        return pd.Series(range(1, len(price) + 1), index=price.index, dtype=float)

    monkeypatch.setattr(mining, "_vector_nav", vector_nav)
    evaluation = {
        "span": ["2024-01-01", "2024-01-15"],
        "engine": "vector",
        "rebal_freq": "D",
    }
    rows = mining._evaluate_candidate(
        precomputed.__name__, "FAKE", {"p": 1}, "coarse", windows, data, evaluation
    )
    assert calls == {"compute": 1, "nav": 1}
    assert all(row["error"] is None for row in rows)

    seen = []
    factory_calls = []
    rolling = types.ModuleType("_mining_rolling_test")

    def fit_window(window_data, params, window, context):
        dates = pd.DatetimeIndex(window_data.inputs["x"].index).normalize()
        assert dates.max() <= pd.Timestamp(window.end)
        assert dates.min() >= pd.Timestamp(window.start) - pd.Timedelta(days=2)
        seen.append(window.window_id)
        return pd.DataFrame({"A": 1.0}, index=dates)

    def rolling_factory(params):
        value = mining.MiningSpec(
            Factor(), execution_mode="rolling_fit", fit_window=fit_window, warmup_days=2
        )
        factory_calls.append(value)
        return value

    rolling.make_mining_spec = rolling_factory
    monkeypatch.setitem(sys.modules, rolling.__name__, rolling)
    calls["nav"] = 0
    rows = mining._evaluate_candidate(
        rolling.__name__, "ROLL", {"p": 1}, "coarse", windows, data, evaluation
    )
    assert seen == ["4/4/0", "4/4/1"]
    assert len(factory_calls) == 1 + len(windows)
    assert len({id(value) for value in factory_calls}) == len(factory_calls)
    assert calls["nav"] == 2
    assert all(row["error"] is None for row in rows)


def test_legacy_mining_section_has_migration_error(tmp_path):
    config = tmp_path / "factor.yaml"
    config.write_text("mining:\n  grid: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="parameter_space.yaml"):
        mining._load_yaml(config)


def test_configuration_validation_rejects_unknown_fields_and_windows():
    with pytest.raises(ValueError, match="unknown fields"):
        mining.validate_parameter_specs(
            {"window": {"type": "int", "low": 10, "high": 20, "typo": 1}}
        )
    with pytest.raises(ValueError, match="must not exceed"):
        mining._validate_window_config({"lengths": [20], "steps": [21]})


def test_summary_supports_configured_metric_aggregation():
    frame = pd.DataFrame(
        [
            {"factor_id": "F", "candidate_id": "C", "window_id": "1", "params_json": "{}", "ann_ret": 0.1, "sharpe": 1.0, "mdd": 0.2, "turnover": 0.3, "rank_ic": 0.1, "error": None},
            {"factor_id": "F", "candidate_id": "C", "window_id": "2", "params_json": "{}", "ann_ret": 0.3, "sharpe": 2.0, "mdd": 0.1, "turnover": 0.2, "rank_ic": 0.2, "error": None},
        ]
    )
    result = mining._summary(
        frame,
        {"objective": {"metric": "ann_ret", "aggregate": "median"}, "top_k": 1},
    )
    assert result.loc[0, "objective"] == pytest.approx(0.2)
    assert result.loc[0, "selection_status"] == "selected"


def test_exact_engine_uses_preloaded_frames():
    index = pd.date_range("2024-01-02", periods=12, freq="B") + pd.Timedelta(hours=15)
    price = pd.DataFrame(
        {
            "A": [10 + value * 0.1 for value in range(12)],
            "B": [20 - value * 0.1 for value in range(12)],
        },
        index=index,
    )
    status = pd.DataFrame(1, index=index.normalize(), columns=["A", "B"])
    rebalance = pd.DatetimeIndex([index[1].normalize(), index[6].normalize()]) + pd.Timedelta(minutes=10)
    weights = pd.DataFrame(
        {"A": [0.5, 0.5], "B": [-0.5, -0.5], "cash": [1.0, 1.0]},
        index=rebalance,
    )
    data = mining.MiningData(
        inputs={},
        price=price,
        execution_price=price,
        trade_status=status,
        pit=None,
        universe=["A", "B"],
    )
    factor = SimpleNamespace(
        name="EXACT_SMOKE",
        backtest_metric="close",
        table_name="daily_market",
    )
    nav = mining._exact_nav(weights, data, mining.MiningSpec(factor), 1_000_000, 24)
    assert not nav.empty
    assert nav.notna().all()


def test_run_logging_is_live_and_persisted(tmp_path, monkeypatch, capsys):
    module = types.ModuleType("_mining_logging_test")

    class Factor:
        name = "LOG_TEST"
        compute_kwargs = {}
        weight_mode = "classic-long-short"

        @staticmethod
        def compute(**kwargs):
            return kwargs["x"]

    module.make_mining_spec = lambda params: mining.MiningSpec(Factor(), warmup_days=0)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    data = _synthetic_data()
    windows = [
        mining.MiningWindow("4/4/0", "2024-01-05", "2024-01-08", 4, 4),
        mining.MiningWindow("4/4/1", "2024-01-09", "2024-01-12", 4, 4),
    ]
    monkeypatch.setattr(mining, "_windows", lambda *args, **kwargs: windows)
    monkeypatch.setattr(mining, "_fetch_data", lambda *args, **kwargs: data)
    monkeypatch.setattr(
        mining,
        "_build_weights",
        lambda *args, **kwargs: pd.DataFrame({"A": 1.0}, index=data.price.index),
    )
    monkeypatch.setattr(
        mining,
        "_vector_nav",
        lambda weights, price: pd.Series(
            np.linspace(1.0, 1.2, len(price)), index=price.index
        ),
    )
    parameter_path = tmp_path / "parameter_space.yaml"
    performance_path = tmp_path / "performance.yaml"
    output_dir = tmp_path / "output"
    parameter_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "factor_class": "test",
                "factors": {
                    "LOG_TEST": {
                        "module": module.__name__,
                        "execution_mode": "precomputed",
                        "parameters": {
                            "window": {"type": "int", "low": 3, "high": 3, "step": 1}
                        },
                    }
                },
                "evaluation": {
                    "span": ["2024-01-01", "2024-01-15"],
                    "engine": "vector",
                    "rebal_freq": "D",
                },
                "windows": {"lengths": [4], "steps": [4]},
                "search": {
                    "coarse": {"sampler": "random", "n_trials": 1, "seed": 1},
                    "fine": {"top_k": 1, "points_per_dimension": 1, "max_candidates": 1},
                },
                "selection": {
                    "objective": {"metric": "sharpe", "aggregate": "median", "direction": "maximize"},
                    "top_k": 1,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    performance_path.write_text(
        yaml.safe_dump(
            {
                "runtime": {"backend": "serial", "workers": 1},
                "cache": {"enabled": False, "format": "npy-memmap"},
                "output": {
                    "directory": str(output_dir),
                    "window_results": "csv",
                    "persist_full_nav": "none",
                },
                "logging": {"level": "INFO", "task_logs": True, "heartbeat_seconds": 0},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = mining.run_mining(parameter_path, performance_path)
    terminal = capsys.readouterr().out
    assert result.output_dir == output_dir
    for marker in (
        "开始参数挖掘",
        "开始处理第1/1个因子",
        "粗搜候选已生成",
        "开始评价第1/",
        "粗搜进度",
        "细搜筛选完成",
        "结果文件已写入",
        "参数挖掘完成",
    ):
        assert marker in terminal
    assert "信息 [主进程 PID=" in terminal
    assert "窗口方案=4日窗口、每4日滑动、第1个窗口" in terminal
    manifest = yaml.safe_load((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    audit = Path(manifest["audit_log"])
    assert audit.exists()
    audit_text = audit.read_text(encoding="utf-8")
    assert "开始参数挖掘" in audit_text
    assert "完成第1/" in audit_text
    assert "参数挖掘完成" in audit_text
    assert not any(
        getattr(handler, "_betalens_mining_handler", False)
        for handler in mining._LOGGER.handlers
    )


def test_failed_run_updates_manifest_and_audit_log(tmp_path):
    parameter_path = tmp_path / "parameter_space.yaml"
    performance_path = tmp_path / "performance.yaml"
    output_dir = tmp_path / "output"
    parameter_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "factor_class": "test",
                "factors": {
                    "BROKEN": {
                        "module": "_missing_mining_module",
                        "execution_mode": "precomputed",
                        "parameters": {"window": {"type": "int", "low": 3, "high": 3}},
                    }
                },
                "evaluation": {"span": ["2024-01-01", "2024-01-15"]},
                "windows": {"lengths": [4], "steps": [4]},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    performance_path.write_text(
        yaml.safe_dump(
            {
                "runtime": {"backend": "serial", "workers": 1},
                "cache": {"enabled": False, "format": "npy-memmap"},
                "output": {"directory": str(output_dir), "window_results": "csv", "persist_full_nav": "none"},
                "logging": {"level": "INFO", "task_logs": True, "heartbeat_seconds": 0},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ModuleNotFoundError):
        mining.run_mining(parameter_path, performance_path)
    manifest = yaml.safe_load((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert "ModuleNotFoundError" in manifest["error"]
    assert "Traceback" in manifest["traceback"]
    audit = Path(manifest["audit_log"]).read_text(encoding="utf-8")
    assert "参数挖掘失败" in audit
    assert "ModuleNotFoundError" in audit
    assert not any(
        getattr(handler, "_betalens_mining_handler", False)
        for handler in mining._LOGGER.handlers
    )
