"""Optuna adapter for the two-stage mining search.

This module intentionally has no worker, cache, audit, or storage concerns.
Those belong to :mod:`betalens.factor.mining`.
"""
from __future__ import annotations

import itertools
import json
import math
from bisect import bisect_left
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


def _optuna():
    try:
        import optuna  # type: ignore
        return optuna
    except ImportError as exc:
        raise ImportError(
            "parameter mining requires Optuna; install it with `pip install optuna`"
        ) from exc


def _choice_token(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return "__json__:" + json.dumps(value, ensure_ascii=False, sort_keys=True)


def _decode_choice(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("__json__:"):
        return json.loads(value[len("__json__:"):])
    return value


def to_optuna_distribution(spec: Mapping[str, Any]):
    kind = str(spec.get("type", "float")).lower()
    optuna = _optuna()
    if kind in {"categorical", "choice", "bool", "boolean"}:
        choices = list(spec.get("choices", [False, True] if kind in {"bool", "boolean"} else []))
        tokens = [_choice_token(value) for value in choices]
        return optuna.distributions.CategoricalDistribution(tokens)
    low, high = spec.get("low"), spec.get("high")
    if low is None or high is None or float(low) > float(high):
        raise ValueError("numeric parameter requires low <= high")
    log = str(spec.get("scale", "linear")).lower() == "log"
    if kind in {"int", "integer"}:
        if log and int(low) < 1:
            raise ValueError("log integer parameters must have low >= 1")
        return optuna.distributions.IntDistribution(
            int(low), int(high), step=int(spec.get("step") or 1), log=log
        )
    return optuna.distributions.FloatDistribution(
        float(low), float(high), step=spec.get("step"), log=log
    )


def suggest_params(trial, parameter_specs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Suggest one ordinary parameter dictionary from an Optuna trial."""
    output = {}
    for name, spec in parameter_specs.items():
        kind = str(spec.get("type", "float")).lower()
        if kind in {"categorical", "choice", "bool", "boolean"}:
            choices = list(
                spec.get("choices", [False, True] if kind in {"bool", "boolean"} else [])
            )
            value = trial.suggest_categorical(name, [_choice_token(item) for item in choices])
        elif kind in {"int", "integer"}:
            value = trial.suggest_int(
                name,
                int(spec["low"]),
                int(spec["high"]),
                step=int(spec.get("step") or 1),
                log=str(spec.get("scale", "linear")).lower() == "log",
            )
        else:
            value = trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                step=spec.get("step"),
                log=str(spec.get("scale", "linear")).lower() == "log",
            )
        output[name] = _decode_choice(value)
    return output


def create_coarse_study(config: Mapping[str, Any], *, direction: str = "maximize"):
    """Create the sampler-only study used by the coarse stage."""
    optuna = _optuna()
    seed = int(config.get("seed", 20260818))
    sampler_name = str(config.get("sampler", "random")).lower()
    if sampler_name == "random":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    elif sampler_name == "tpe":
        sampler = optuna.samplers.TPESampler(seed=seed)
    else:
        raise ValueError(f"unsupported coarse sampler: {sampler_name}")
    return optuna.create_study(sampler=sampler, direction=direction)


def create_fine_grid_study(
    search_space: Mapping[str, Sequence[Any]],
    *,
    direction: str = "maximize",
    seed: int = 20260818,
):
    """Create a GridSampler study while preserving composite categorical values."""
    optuna = _optuna()
    encoded = {
        name: [_choice_token(value) for value in values]
        for name, values in search_space.items()
    }
    if not encoded or any(not values for values in encoded.values()):
        raise ValueError("fine grid search space must contain non-empty dimensions")
    return optuna.create_study(
        sampler=optuna.samplers.GridSampler(encoded, seed=int(seed)),
        direction=direction,
    )


def tell_trial(study, trial, value: float | None) -> None:
    """Complete an externally evaluated trial, including GridSampler exhaustion."""
    try:
        study.tell(trial, value)
    except RuntimeError as exc:
        # GridSampler saves the final trial, then calls Study.stop(). Optuna only
        # permits that call from optimize(), not from the documented ask/tell API.
        if "Study.stop" not in str(exc):
            raise


def _values(spec: Mapping[str, Any], count: int) -> list[Any]:
    kind = str(spec.get("type", "float")).lower()
    if kind in {"categorical", "choice", "bool", "boolean"}:
        return list(spec.get("choices", [False, True] if kind in {"bool", "boolean"} else []))
    low, high = float(spec["low"]), float(spec["high"])
    count = max(1, int(count))
    if count == 1 or low == high:
        values = [low]
    elif str(spec.get("scale", "linear")).lower() == "log":
        if low <= 0:
            raise ValueError("log parameter requires low > 0")
        values = [math.exp(math.log(low) + (math.log(high) - math.log(low)) * i / (count - 1)) for i in range(count)]
    else:
        values = [low + (high - low) * i / (count - 1) for i in range(count)]
    if kind in {"int", "integer"}:
        step = int(spec.get("step") or 1)
        return list(
            dict.fromkeys(
                max(int(low), min(int(high), int(low + round((value - low) / step) * step)))
                for value in values
            )
        )
    step = spec.get("step")
    if step:
        decimals = max(0, len(str(step).split(".")[-1])) if "." in str(step) else 0
        values = [round(round((value - low) / float(step)) * float(step) + low, decimals) for value in values]
    return list(dict.fromkeys(values))


def _dedupe(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output, seen = [], set()
    for row in rows:
        key = json.dumps(dict(row), sort_keys=True, ensure_ascii=False, default=str)
        if key not in seen:
            seen.add(key)
            output.append(dict(row))
    return output


def _unique_values(values: Sequence[Any]) -> list[Any]:
    output, seen = [], set()
    for value in values:
        key = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            output.append(value)
    return output


def generate_coarse_candidates(parameters: Mapping[str, Mapping[str, Any]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Generate sparse candidates using an ask/tell Optuna study."""
    n_trials = max(1, int(config.get("n_trials", 32)))
    study = create_coarse_study(config)
    rows = []
    for _ in range(n_trials):
        trial = study.ask()
        rows.append(suggest_params(trial, parameters))
        study.tell(trial, 0.0)
    return _dedupe(rows)


@dataclass(frozen=True)
class FineGridPlan:
    candidates: list[dict[str, Any]]
    anchors: list[dict[str, Any]]
    dimensions: dict[str, list[Any]]
    local_bounds: dict[str, dict[str, Any]]


def generate_fine_candidates(
    parameters: Mapping[str, Mapping[str, Any]],
    anchors: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    coarse_candidates: Sequence[Mapping[str, Any]] | None = None,
) -> FineGridPlan:
    """Generate a bounded local grid around coarse winners."""
    if not anchors:
        return FineGridPlan([], [], {}, {})
    points = max(1, int(config.get("points_per_dimension", 7)))
    max_candidates = max(1, int(config.get("max_candidates", 256)))
    dimensions: dict[str, list[Any]] = {}
    local_bounds: dict[str, dict[str, Any]] = {}
    for name, spec in parameters.items():
        kind = str(spec.get("type", "float")).lower()
        if kind in {"categorical", "choice", "bool", "boolean"}:
            dimensions[name] = list(spec.get("choices", [False, True] if kind in {"bool", "boolean"} else []))
            local_bounds[name] = {"type": "categorical", "values": dimensions[name]}
            continue
        values = sorted(float(row[name]) for row in anchors if name in row)
        low, high = float(spec["low"]), float(spec["high"])
        sampled = sorted(
            set(
                float(row[name])
                for row in (coarse_candidates or [])
                if name in row
            )
        )
        local_lows, local_highs = [], []
        for anchor in values:
            position = bisect_left(sampled, anchor)
            if sampled:
                left = sampled[position - 1] if position > 0 else low
                next_position = position + 1 if position < len(sampled) and sampled[position] == anchor else position
                right = sampled[next_position] if next_position < len(sampled) else high
            else:
                gap = (high - low) / max(points - 1, 1)
                left, right = anchor - gap, anchor + gap
            local_lows.append(left)
            local_highs.append(right)
        local = dict(spec)
        local["low"] = max(low, min(local_lows))
        local["high"] = min(high, max(local_highs))
        dimensions[name] = _values(local, points)
        local_bounds[name] = {
            "type": kind,
            "global_low": spec["low"],
            "global_high": spec["high"],
            "local_low": local["low"],
            "local_high": local["high"],
            "values": dimensions[name],
        }
    names = list(dimensions)
    candidates = [dict(zip(names, values)) for values in itertools.product(*(dimensions[name] for name in names))]
    return FineGridPlan(
        candidates=_dedupe(candidates)[:max_candidates],
        anchors=[dict(value) for value in anchors],
        dimensions=dimensions,
        local_bounds=local_bounds,
    )


__all__ = [
    "create_coarse_study",
    "create_fine_grid_study",
    "FineGridPlan",
    "generate_coarse_candidates",
    "generate_fine_candidates",
    "suggest_params",
    "tell_trial",
    "to_optuna_distribution",
]
