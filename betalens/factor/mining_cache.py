"""Immutable memory-mapped cache primitives for factor mining workers."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


_LOGGER = logging.getLogger("betalens.factor.mining")


def _json_write(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_file(path: Path) -> None:
    with path.open("ab", buffering=0) as stream:
        stream.flush()
        os.fsync(stream.fileno())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(name: str) -> str:
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
    return f"{digest}_{''.join(ch if ch.isalnum() else '_' for ch in name)[:40]}"


def _write_axis(root: Path, kind: str, payload: bytes, suffix: str) -> Path:
    digest = hashlib.sha256(payload).hexdigest()
    target = root / "axes" / f"{kind}-{digest[:20]}{suffix}"
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".tmp-{os.getpid()}-{uuid.uuid4().hex}")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, target)
    except FileExistsError:
        temporary.unlink(missing_ok=True)
    return target


def _write_dates_axis(root: Path, dates: np.ndarray) -> Path:
    payload = dates.astype(np.int64, copy=False).tobytes(order="C")
    digest = hashlib.sha256(payload).hexdigest()
    target = root / "axes" / f"dates-{digest[:20]}.npy"
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".tmp-{os.getpid()}-{uuid.uuid4().hex}")
    with temporary.open("wb") as stream:
        np.save(stream, dates.astype(np.int64, copy=False), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    return target


def _write_frame(root: Path, name: str, frame: pd.DataFrame, *, dtype: str | None = None) -> dict[str, Any]:
    target = root / _safe_name(name)
    target.mkdir(parents=True, exist_ok=False)
    dates = pd.DatetimeIndex(frame.index).asi8
    dates_path = _write_dates_axis(root, dates)
    columns_payload = json.dumps(
        {"columns": [str(column) for column in frame.columns]},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    columns_path = _write_axis(root, "columns", columns_payload, ".json")

    numeric = dtype is not None or all(pd.api.types.is_numeric_dtype(value) for _key, value in frame.items())
    if numeric:
        array = frame.to_numpy(dtype=dtype or np.float64, copy=True)
        kind = "numeric"
        categories: list[str] = []
    else:
        values = frame.to_numpy(dtype=object, copy=False)
        flat = pd.Series(values.ravel(), dtype=object)
        normalized = flat.where(flat.isna(), flat.astype(str))
        codes, uniques = pd.factorize(normalized, sort=True, use_na_sentinel=True)
        categories = [str(value) for value in uniques.tolist()]
        array = codes.reshape(values.shape).astype(np.int32, copy=False)
        kind = "categorical"
        _json_write(target / "categories.json", {"categories": categories})
    values_path = target / "values.npy"
    np.save(values_path, array, allow_pickle=False)
    _fsync_file(values_path)
    return {
        "name": name,
        "path": target.name,
        "dates_path": str(dates_path.relative_to(root)),
        "columns_path": str(columns_path.relative_to(root)),
        "kind": kind,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "bytes": int(array.nbytes + dates.nbytes),
        "categories": len(categories),
        "dates_sha256": _file_sha256(dates_path),
        "columns_sha256": _file_sha256(columns_path),
        "values_sha256": _file_sha256(values_path),
    }


def _write_pit(root: Path, pit: Mapping[Any, set[str]] | None, universe: list[str]) -> dict[str, Any] | None:
    if not pit:
        return None
    dates = sorted(pd.Timestamp(day).normalize() for day in pit)
    code_index = {code: index for index, code in enumerate(universe)}
    mask = np.zeros((len(dates), len(universe)), dtype=np.bool_)
    for row, day in enumerate(dates):
        members = pit.get(day.date(), pit.get(day, set()))
        columns = [code_index[str(code)] for code in members if str(code) in code_index]
        if columns:
            mask[row, columns] = True
    frame = pd.DataFrame(mask, index=pd.DatetimeIndex(dates), columns=universe)
    return _write_frame(root, "__pit_mask__", frame, dtype="bool")


def publish(
    cache_dir: str | Path,
    signature: str,
    *,
    inputs: Mapping[str, pd.DataFrame],
    price: pd.DataFrame,
    execution_price: pd.DataFrame,
    trade_status: pd.DataFrame,
    industry_by_scheme: Mapping[str, pd.DataFrame],
    pit: Mapping[Any, set[str]] | None,
    universe: list[str],
    metadata: Mapping[str, Any],
) -> Path:
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = cache_dir / "input_manifest.json"
    final = cache_dir / "datasets"
    if manifest.exists() or final.exists():
        raise FileExistsError(f"task input cache already exists: {cache_dir}")
    staging = cache_dir / f".datasets.staging-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        staging.mkdir(parents=True, exist_ok=False)
        datasets = {
            "inputs": {name: _write_frame(staging, f"input:{name}", frame) for name, frame in inputs.items()},
            "price": _write_frame(staging, "price", price),
            "execution_price": _write_frame(staging, "execution_price", execution_price),
            "trade_status": _write_frame(staging, "trade_status", trade_status, dtype="int8"),
            "industry_by_scheme": {
                name: _write_frame(staging, f"industry:{name}", frame)
                for name, frame in industry_by_scheme.items()
            },
            "pit": _write_pit(staging, pit, universe),
        }
        for descriptor in _iter_descriptors(datasets):
            _validate_descriptor(staging, descriptor)
        os.replace(staging, final)
        temporary = cache_dir / f".input_manifest.{uuid.uuid4().hex}.tmp"
        _json_write(temporary, {
            "schema_version": 4,
            "cache_signature": signature,
            "created_at": time.time(),
            "universe": universe,
            "datasets": datasets,
            "metadata": dict(metadata),
        })
        os.replace(temporary, manifest)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _iter_descriptors(datasets: Mapping[str, Any]):
    yield from datasets.get("inputs", {}).values()
    yield from datasets.get("industry_by_scheme", {}).values()
    for name in ("price", "execution_price", "trade_status", "pit"):
        descriptor = datasets.get(name)
        if descriptor:
            yield descriptor


def _validate_descriptor(root: Path, descriptor: Mapping[str, Any]) -> None:
    target = root / str(descriptor["path"])
    dates_path = root / str(descriptor["dates_path"])
    columns_path = root / str(descriptor["columns_path"])
    values_path = target / "values.npy"
    for path, key in (
        (dates_path, "dates_sha256"),
        (columns_path, "columns_sha256"),
        (values_path, "values_sha256"),
    ):
        if not path.exists() or _file_sha256(path) != descriptor[key]:
            raise RuntimeError(f"mining cache verification failed: {path}")
    dates = np.load(dates_path, mmap_mode="r", allow_pickle=False)
    values = np.load(values_path, mmap_mode="r", allow_pickle=False)
    columns = json.loads(columns_path.read_text(encoding="utf-8"))["columns"]
    expected = tuple(int(value) for value in descriptor["shape"])
    if values.shape != expected or len(dates) != expected[0] or len(columns) != expected[1]:
        raise RuntimeError(
            f"mining cache shape verification failed: {descriptor['name']} "
            f"values={values.shape} dates={len(dates)} columns={len(columns)} expected={expected}"
        )


def _load_descriptor(root: Path, descriptor: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not descriptor:
        return None
    target = root / str(descriptor["path"])
    loaded = dict(descriptor)
    loaded["dates"] = np.load(root / str(descriptor["dates_path"]), mmap_mode="r", allow_pickle=False)
    loaded["values"] = np.load(target / "values.npy", mmap_mode="r", allow_pickle=False)
    loaded["columns"] = json.loads(
        (root / str(descriptor["columns_path"])).read_text(encoding="utf-8")
    )["columns"]
    if descriptor["kind"] == "categorical":
        loaded["category_values"] = json.loads(
            (target / "categories.json").read_text(encoding="utf-8")
        )["categories"]
    return loaded


def open_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 4:
        raise RuntimeError(f"unsupported mining cache schema: {payload.get('schema_version')}")
    root = path.parent / "datasets"
    datasets = payload["datasets"]
    return {
        "manifest": payload,
        "universe": list(payload.get("universe") or []),
        "inputs": {name: _load_descriptor(root, item) for name, item in datasets["inputs"].items()},
        "price": _load_descriptor(root, datasets["price"]),
        "execution_price": _load_descriptor(root, datasets["execution_price"]),
        "trade_status": _load_descriptor(root, datasets["trade_status"]),
        "industry_by_scheme": {
            name: _load_descriptor(root, item)
            for name, item in datasets["industry_by_scheme"].items()
        },
        "pit": _load_descriptor(root, datasets.get("pit")),
    }


def frame(descriptor: Mapping[str, Any], start: Any = None, end: Any = None, columns=None) -> pd.DataFrame:
    dates = np.asarray(descriptor["dates"])
    lo = 0 if start is None else int(np.searchsorted(dates, pd.Timestamp(start).value, side="left"))
    hi = len(dates) if end is None else int(np.searchsorted(dates, pd.Timestamp(end).value, side="right"))
    all_columns = list(descriptor["columns"])
    if columns is None:
        selected_columns = all_columns
        column_indexes = slice(None)
    else:
        selected_columns = [str(column) for column in columns if str(column) in all_columns]
        lookup = {column: index for index, column in enumerate(all_columns)}
        column_indexes = [lookup[column] for column in selected_columns]
    values = descriptor["values"][lo:hi, column_indexes]
    if descriptor["kind"] == "categorical":
        categories = np.asarray(descriptor["category_values"], dtype=object)
        codes = np.asarray(values, dtype=np.int32)
        decoded = np.full(codes.shape, np.nan, dtype=object)
        valid = codes >= 0
        decoded[valid] = categories[codes[valid]]
        values = decoded
    return pd.DataFrame(
        values,
        index=pd.DatetimeIndex(dates[lo:hi]),
        columns=selected_columns,
        copy=False,
    )


def estimated_resident_bytes(path: str | Path) -> int:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    total = 0
    datasets = payload.get("datasets", {})
    groups = [datasets.get("inputs", {}), datasets.get("industry_by_scheme", {})]
    descriptors = [
        datasets.get("price"), datasets.get("execution_price"),
        datasets.get("trade_status"), datasets.get("pit"),
    ]
    for group in groups:
        descriptors.extend(group.values())
    for descriptor in descriptors:
        if descriptor:
            total += int(descriptor.get("bytes", 0))
    # Only the active window is faulted into each worker. The fixed allowance
    # covers pandas objects, weights and formula intermediates.
    return min(total, 768 * 1024 * 1024)


@dataclass(frozen=True)
class CacheRequest:
    """Description of one task-local mining input cache."""

    directory: str | Path
    signature: str


class MiningCache:
    """Small public facade over the task-local memmap cache.

    One task writes one ``input_manifest.json`` and one ``datasets`` directory.
    Workers only load immutable slices from those files.
    """

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = open_manifest(self.manifest_path)

    @classmethod
    def open_or_build(cls, request: CacheRequest, builder=None) -> "MiningCache":
        started = time.perf_counter()
        directory = Path(request.directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        manifest = directory / "input_manifest.json"
        signature_short = request.signature[:12]
        _LOGGER.info(
            "开始检查任务输入缓存：目录=%s，签名=%s",
            directory,
            signature_short,
        )
        if manifest.exists():
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            if payload.get("cache_signature") != request.signature:
                raise RuntimeError("task input cache signature mismatch")
            return cls(manifest)
        if builder is None:
            raise FileNotFoundError(
                f"mining cache generation {request.signature} is missing; provide a builder"
            )
        _LOGGER.info("开始构建任务输入缓存：签名=%s", signature_short)
        payload = builder()
        _LOGGER.info("开始原子发布任务输入缓存：签名=%s", signature_short)
        manifest = publish(
            directory,
            request.signature,
            **payload,
        )
        _LOGGER.info(
            "任务输入缓存构建完成：签名=%s，清单=%s，总耗时=%.1f秒",
            signature_short,
            manifest,
            time.perf_counter() - started,
        )
        return cls(manifest)

    @property
    def universe(self) -> list[str]:
        return list(self.manifest.get("universe") or [])

    def load(self, name: str, start=None, end=None, columns=None) -> pd.DataFrame:
        groups = {
            **self.manifest.get("inputs", {}),
            **self.manifest.get("industry_by_scheme", {}),
            "price": self.manifest.get("price"),
            "execution_price": self.manifest.get("execution_price"),
            "trade_status": self.manifest.get("trade_status"),
            "pit": self.manifest.get("pit"),
        }
        descriptor = groups.get(name)
        if descriptor is None:
            raise KeyError(f"unknown mining cache dataset: {name}")
        return frame(descriptor, start=start, end=end, columns=columns)


__all__ = [
    "CacheRequest",
    "MiningCache",
    "estimated_resident_bytes",
    "frame",
    "open_manifest",
    "publish",
]
