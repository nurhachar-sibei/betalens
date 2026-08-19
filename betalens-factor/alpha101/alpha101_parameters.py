"""Alpha101 公式参数目录与自动挖掘边界生成。

参数元数据来自 :mod:`alpha101_formulas` 中每个 ``alphaN`` 函数的关键字默认值。
自动参数空间是围绕论文默认值生成的启发式范围，不读取历史数据，也不根据回测
结果反推全局边界。默认最多放开公式签名中靠前的三个可搜索参数；其余参数仍会
进入配置，但 ``low == high == default``。

边界规则：

* ``window`` / ``lag``：约 ``[0.5d, d, 2d]``，最小值为 1；
* ``weight``：``[max(0, 0.5d), d, min(1, 1.5d)]``；
* ``threshold``：``d +/- max(0.5 * abs(d), 0.05)``；
* ``exponent``：``[0.5d, d, 2d]``；
* 其他种类固定为论文默认值 ``d``。

参考点最终只用于推导 ``low`` / ``high``。粗搜会在完整边界内采样，而不是只在
三个参考点上取值；``window`` / ``lag`` 使用对数尺度，其他参数使用线性尺度。
"""
from __future__ import annotations

import itertools
import json
import math
from typing import Any, Mapping, Sequence

from alpha101_formulas import AlphaParameter, default_compute_kwargs, get_definition


def parameter_catalog(alpha_id: str | int) -> dict[str, AlphaParameter]:
    """Return the ordered parameter catalog for one Alpha formula."""
    return dict(get_definition(alpha_id).parameters)


def _unique(values: Sequence[int | float]) -> list[int | float]:
    output: list[int | float] = []
    for value in values:
        if value not in output:
            output.append(value)
    return output


def candidate_values(spec: AlphaParameter) -> list[int | float]:
    """按参数种类返回论文默认值附近、用于推导边界的参考点。"""
    default = spec.default
    if spec.kind in {"window", "lag"}:
        if isinstance(default, float):
            center = float(default)
            return _unique([max(1.0, center * 0.5), center, max(1.0, center * 2.0)])
        center = max(1, int(default))
        return _unique([max(1, int(math.floor(center * 0.5 + 0.5))), center, max(1, center * 2)])
    if spec.kind == "weight":
        center = float(default)
        return _unique([max(0.0, center * 0.5), center, min(1.0, center * 1.5)])
    if spec.kind == "threshold":
        center = float(default)
        spread = max(abs(center) * 0.5, 0.05)
        return _unique([center - spread, center, center + spread])
    if spec.kind == "exponent":
        center = float(default)
        return _unique([center * 0.5, center, center * 2.0])
    return [default]


def default_search_space(alpha_id: str | int, max_dimensions: int = 3) -> dict[str, list[int | float]]:
    """按公式参数顺序放开至多 ``max_dimensions`` 个可搜索参数。"""
    if int(max_dimensions) < 0:
        raise ValueError("max_dimensions must be >= 0")
    remaining = int(max_dimensions)
    search_space: dict[str, list[int | float]] = {}
    for name, spec in parameter_catalog(alpha_id).items():
        values = [spec.default]
        if spec.searchable and remaining > 0:
            proposed = candidate_values(spec)
            if len(proposed) > 1:
                values = proposed
                remaining -= 1
        search_space[name] = values
    return search_space


def mining_parameter_specs(alpha_id: str | int, max_dimensions: int = 3) -> dict[str, dict[str, Any]]:
    """将参考点转换为 mining 使用的类型、边界、步长和尺度定义。

    整数参数使用 ``step=1``；窗口和滞后参数使用 ``scale=log``，其余参数使用
    ``scale=linear``。参考点本身不会作为 categorical 候选保留下来。
    """
    values_by_name = default_search_space(alpha_id, max_dimensions=max_dimensions)
    output = {}
    for name, values in values_by_name.items():
        catalog = parameter_catalog(alpha_id)[name]
        is_float = isinstance(catalog.default, float) or any(isinstance(value, float) for value in values)
        spec = {
            "type": "float" if is_float else "int",
            "low": min(values),
            "high": max(values),
            "scale": "log" if catalog.kind in {"window", "lag"} else "linear",
        }
        if not is_float:
            spec["step"] = 1
        output[name] = spec
    return validate_mining_parameter_specs(alpha_id, output)


def validate_mining_parameter_specs(
    alpha_id: str | int,
    parameter_specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Validate aggregate parameter definitions against one Alpha formula."""
    definition = get_definition(alpha_id)
    expected, supplied = set(definition.parameters), set(parameter_specs)
    if expected != supplied:
        missing, unknown = sorted(expected - supplied), sorted(supplied - expected)
        raise ValueError(f"{definition.name} parameter specs mismatch: missing={missing}; unknown={unknown}")
    from betalens.factor.mining import validate_parameter_specs

    output = {name: dict(parameter_specs[name]) for name in definition.parameters}
    validate_parameter_specs(output)
    return output


def mining_optuna_distributions(
    alpha_id: str | int,
    parameter_specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build Optuna distributions after formula-aware validation."""
    from betalens.factor.mining_optuna import to_optuna_distribution

    validated = validate_mining_parameter_specs(alpha_id, parameter_specs)
    return {name: to_optuna_distribution(spec) for name, spec in validated.items()}


def aggregate_mining_factors() -> dict[str, dict[str, Any]]:
    """将 ``factors: all`` 展开为 ALPHA1 至 ALPHA101 的自动参数空间。"""
    return {
        get_definition(number).name: {
            "module": "alpha101_mining",
            "execution_mode": "precomputed",
            "parameters": mining_parameter_specs(number),
        }
        for number in range(1, 102)
    }


def validate_search_space(
    alpha_id: str | int,
    search_space: Mapping[str, Sequence[Any]],
) -> dict[str, list[int | float]]:
    """Require one non-empty, numeric value list for every formula parameter."""
    definition = get_definition(alpha_id)
    expected = set(definition.parameters)
    supplied = set(search_space)
    missing = sorted(expected - supplied)
    unknown = sorted(supplied - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unknown:
            details.append(f"unknown={unknown}")
        raise ValueError(f"{definition.name} search_space mismatch: {'; '.join(details)}")

    validated: dict[str, list[int | float]] = {}
    for name, spec in definition.parameters.items():
        raw_values = search_space[name]
        if not isinstance(raw_values, (list, tuple)) or not raw_values:
            raise ValueError(f"{definition.name}.{name} must be a non-empty YAML list")
        values: list[int | float] = []
        for raw in raw_values:
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise TypeError(f"{definition.name}.{name} values must be numeric")
            value = float(raw) if isinstance(spec.default, float) or isinstance(raw, float) else int(raw)
            if not math.isfinite(float(value)):
                raise ValueError(f"{definition.name}.{name} values must be finite")
            if spec.kind in {"window", "lag"} and value < 1:
                raise ValueError(f"{definition.name}.{name} values must be >= 1")
            if spec.kind == "exponent" and value == 0:
                raise ValueError(f"{definition.name}.{name} values must be non-zero")
            if value not in values:
                values.append(value)
        validated[name] = values
    return validated


def grid_candidate_count(search_space: Mapping[str, Sequence[Any]]) -> int:
    return math.prod(len(values) for values in search_space.values())


def formula_param_candidates(
    alpha_id: str | int,
    search_space: Mapping[str, Sequence[Any]],
    *,
    max_grid_candidates: int = 256,
) -> list[dict[str, Any]]:
    """Expand the complete grid and fail before execution if it exceeds the cap."""
    validated = validate_search_space(alpha_id, search_space)
    count = grid_candidate_count(validated)
    if count > int(max_grid_candidates):
        raise ValueError(
            f"{get_definition(alpha_id).name} grid has {count} candidates, "
            f"exceeding max_grid_candidates={int(max_grid_candidates)}"
        )
    names = list(validated)
    return [
        dict(zip(names, values, strict=True))
        for values in itertools.product(*(validated[name] for name in names))
    ]


def formula_param_gid(alpha_id: str | int, params: Mapping[str, Any]) -> str:
    """Create a deterministic, filesystem-safe candidate identifier."""
    name = get_definition(alpha_id).name
    encoded = json.dumps(dict(params), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    import hashlib

    return f"{name}_{hashlib.sha1(encoded.encode('utf-8')).hexdigest()[:12]}"


def catalog_rows(alpha_id: str | int) -> list[dict[str, Any]]:
    definition = get_definition(alpha_id)
    return [
        {
            "alpha": definition.name,
            "parameter": spec.name,
            "kind": spec.kind,
            "default": spec.default,
            "searchable": spec.searchable,
            "source_line": spec.source_line,
        }
        for spec in definition.parameters.values()
    ]


__all__ = [
    "candidate_values",
    "catalog_rows",
    "default_compute_kwargs",
    "default_search_space",
    "formula_param_candidates",
    "formula_param_gid",
    "grid_candidate_count",
    "aggregate_mining_factors",
    "mining_optuna_distributions",
    "mining_parameter_specs",
    "parameter_catalog",
    "validate_mining_parameter_specs",
    "validate_search_space",
]
