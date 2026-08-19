"""Task-scoped storage and human-readable audit export for factor mining."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sqlite3
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


_TABLES = ("search_progress", "window_results", "candidate_summary", "winners", "errors")


def _json_default(value: Any) -> str:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _json_payload(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), ensure_ascii=False, default=_json_default, allow_nan=True)


def _safe_factor_id(value: str) -> str:
    result = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value))
    return result.strip("._") or "factor"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(path: Path) -> dict[str, Any]:
    try:
        root = subprocess.run(
            ["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", root, "status", "--short"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.splitlines()
        return {"root": root, "commit": commit, "dirty": bool(status), "changes": status}
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def _atomic_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    import yaml

    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(yaml.safe_dump(dict(payload), allow_unicode=True, sort_keys=False), encoding="utf-8")
    os.replace(temporary, path)


@dataclass
class ResultStore:
    path: Path
    connection: sqlite3.Connection = field(init=False)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        for table in _TABLES:
            self.connection.execute(
                f"CREATE TABLE IF NOT EXISTS {table} ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, event TEXT, stage TEXT, factor_id TEXT, "
                "candidate_id TEXT, window_id TEXT, payload_json TEXT NOT NULL)"
            )
            self.connection.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{table}_candidate ON {table}(stage, candidate_id)"
            )
        self.connection.commit()

    def append(self, table: str, rows: Iterable[Mapping[str, Any]]) -> int:
        if table not in _TABLES:
            raise ValueError(f"unknown mining result table: {table}")
        payloads = [(
            row.get("event"), row.get("stage"), row.get("factor_id"),
            row.get("candidate_id"), row.get("window_id"), _json_payload(row),
        ) for row in rows]
        if not payloads:
            return 0
        self.connection.executemany(
            f"INSERT INTO {table}(event, stage, factor_id, candidate_id, window_id, payload_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            payloads,
        )
        self.connection.commit()
        return len(payloads)

    def read(self, table: str) -> pd.DataFrame:
        if table not in _TABLES:
            raise ValueError(f"unknown mining result table: {table}")
        rows = self.connection.execute(f"SELECT payload_json FROM {table} ORDER BY id").fetchall()
        return pd.DataFrame([json.loads(row[0]) for row in rows])

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


def _empty_safe(frame: pd.DataFrame, fallback: str = "暂无记录") -> pd.DataFrame:
    if not frame.empty:
        return frame
    return pd.DataFrame({"说明": [fallback]})


_COLUMN_NAMES = {
    "event": "记录类型", "stage": "搜索阶段", "factor_id": "因子",
    "candidate_id": "候选编号", "candidate_order": "候选序号",
    "completed_order": "完成序号", "trial_number": "试验编号",
    "source": "候选来源", "params_json": "参数JSON", "anchors_json": "粗搜锚点",
    "local_bounds_json": "细搜局部边界", "window_id": "窗口编号",
    "window_start": "窗口开始", "window_end": "窗口结束", "ann_ret": "年化收益率",
    "ann_vol": "年化波动率", "sharpe": "夏普比率", "mdd": "最大回撤",
    "calmar": "卡玛比率", "turnover": "换手率", "rank_ic": "Rank IC",
    "ic_coverage": "IC覆盖率", "n_days": "有效天数", "error": "错误信息",
    "error_type": "错误类型", "valid_windows": "有效窗口数",
    "failed_windows": "失败窗口数", "candidate_elapsed_seconds": "候选耗时（秒）",
    "sharpe_median": "夏普中位数", "rank": "排名", "selection_status": "筛选状态",
    "selection_reason": "筛选说明", "objective": "目标值", "valid_window_ratio": "有效窗口比例",
    "valid_window_count": "有效窗口数", "window_count": "窗口总数", "max_mdd": "最大回撤",
}


def _humanize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    suffixes = {"mean": "均值", "median": "中位数", "p25": "25%分位数", "min": "最小值", "max": "最大值"}
    metric_names = {
        "ann_ret": "年化收益率", "ann_vol": "年化波动率", "sharpe": "夏普比率",
        "mdd": "最大回撤", "calmar": "卡玛比率", "turnover": "换手率",
        "rank_ic": "Rank IC", "ic_coverage": "IC覆盖率", "n_days": "有效天数",
    }
    names = {}
    for column in frame.columns:
        translated = _COLUMN_NAMES.get(str(column))
        if translated is None:
            for suffix, suffix_name in suffixes.items():
                marker = f"_{suffix}"
                if str(column).endswith(marker):
                    metric = str(column)[:-len(marker)]
                    translated = f"{metric_names.get(metric, metric)}{suffix_name}"
                    break
        names[column] = translated or str(column)
    return frame.rename(columns=names)


def _write_sheet(writer: pd.ExcelWriter, name: str, frame: pd.DataFrame, *, max_rows: int = 1_000_000) -> None:
    frame = _empty_safe(frame)
    if name == "窗口表现" and len(frame) > max_rows:
        for offset in range(0, len(frame), max_rows):
            frame.iloc[offset:offset + max_rows].to_excel(
                writer,
                sheet_name=f"窗口表现_{offset // max_rows + 1}",
                index=False,
            )
    else:
        frame.to_excel(writer, sheet_name=name[:31], index=False)


def _style_workbook(path: Path) -> None:
    from openpyxl import load_workbook
    from openpyxl.styles import Font, PatternFill

    workbook = load_workbook(path)
    header_fill = PatternFill("solid", fgColor="D9EAF7")
    for sheet in workbook.worksheets:
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.font = Font(bold=True)
            cell.fill = header_fill
        for column in sheet.columns:
            values = [str(cell.value or "") for cell in list(column)[:200]]
            width = min(42, max(10, max((len(value) for value in values), default=10) + 2))
            sheet.column_dimensions[column[0].column_letter].width = width
            header = str(column[0].value or "")
            if any(word in header for word in ("收益率", "波动率", "回撤", "换手率", "覆盖率", "比例")):
                for cell in list(column)[1:]:
                    cell.number_format = "0.00%"
    workbook.save(path)


@dataclass
class FactorMiningResult:
    factor_id: str
    run_id: str
    run_dir: Path
    status: str
    coarse_window_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    coarse_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    fine_window_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    fine_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    selected_candidates: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class MiningResult:
    launch_id: str
    factor_runs: tuple[FactorMiningResult, ...]


@dataclass
class MiningTask:
    factor_id: str
    run_id: str
    launch_id: str
    run_dir: Path
    cache_dir: Path
    audit_dir: Path
    metadata_path: Path
    workbook_path: Path
    log_path: Path
    store: ResultStore
    metadata: dict[str, Any]

    @classmethod
    def create(
        cls,
        output_root: str | Path,
        factor_id: str,
        launch_id: str,
        *,
        factor_class: str | None,
        parameter_path: Path,
        performance_path: Path,
        config: Mapping[str, Any],
    ) -> "MiningTask":
        now = datetime.now().astimezone()
        git = _git_state(parameter_path)
        run_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        factor_dir = Path(output_root).resolve() / _safe_factor_id(factor_id)
        run_dir = factor_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        cache_dir = run_dir / "cache"
        audit_dir = run_dir / "audit"
        cache_dir.mkdir()
        audit_dir.mkdir()
        metadata_path = run_dir / "metadata.yaml"
        task = cls(
            factor_id=str(factor_id),
            run_id=run_id,
            launch_id=launch_id,
            run_dir=run_dir,
            cache_dir=cache_dir,
            audit_dir=audit_dir,
            metadata_path=metadata_path,
            workbook_path=audit_dir / "挖掘审计.xlsx",
            log_path=audit_dir / "运行日志.log",
            store=ResultStore(cache_dir / "results.sqlite3"),
            metadata={
                "schema_version": 1,
                "status": "running",
                "launch_id": launch_id,
                "run_id": run_id,
                "factor_id": str(factor_id),
                "factor_class": factor_class,
                "started_at_local": now.isoformat(),
                "started_at_utc": now.astimezone(timezone.utc).isoformat(),
                "parameter_config": {"path": str(parameter_path), "sha256": _sha256(parameter_path)},
                "performance_config": {"path": str(performance_path), "sha256": _sha256(performance_path)},
                "configuration": dict(config),
                "environment": {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "hostname": platform.node(),
                    "pid": os.getpid(),
                },
                "git": git,
                "paths": {
                    "run_dir": str(run_dir),
                    "cache_dir": str(cache_dir),
                    "audit_dir": str(audit_dir),
                    "workbook": str(audit_dir / "挖掘审计.xlsx"),
                    "log": str(audit_dir / "运行日志.log"),
                    "results_db": str(cache_dir / "results.sqlite3"),
                },
            },
        )
        task.write_metadata()
        return task

    def write_metadata(self, **updates: Any) -> None:
        self.metadata.update(updates)
        _atomic_yaml(self.metadata_path, self.metadata)

    def export_workbook(self) -> Path:
        overview = pd.DataFrame(
            [
                {
                    "项目": key,
                    "内容": json.dumps(value, ensure_ascii=False, default=_json_default)
                    if isinstance(value, (dict, list)) else value,
                }
                for key, value in self.metadata.items()
                if key != "configuration"
            ]
        )
        config_rows = []
        for section, values in self.metadata.get("configuration", {}).items():
            if isinstance(values, Mapping):
                for key, value in values.items():
                    config_rows.append({"配置段": section, "配置项": key, "配置值": json.dumps(value, ensure_ascii=False, default=_json_default) if isinstance(value, (dict, list)) else value})
            else:
                config_rows.append({"配置段": "", "配置项": section, "配置值": values})
        parameter_rows = []
        parameter_config = self.metadata.get("configuration", {}).get("parameter_space", {})
        for factor_id, factor in (parameter_config.get("factors") or {}).items() if isinstance(parameter_config, Mapping) else []:
            if str(factor_id) != self.factor_id:
                continue
            for name, spec in (factor.get("parameters") or {}).items():
                row = {"因子": factor_id, "参数": name, **dict(spec)}
                parameter_rows.append(row)
        progress = self.store.read("search_progress")
        windows = self.store.read("window_results")
        summaries = self.store.read("candidate_summary")
        winners = self.store.read("winners")
        errors = self.store.read("errors")
        if not summaries.empty and not winners.empty and "candidate_id" in winners and "candidate_id" in summaries:
            winner_ids = set(winners["candidate_id"].astype(str))
            winner_summaries = summaries[
                summaries["candidate_id"].astype(str).isin(winner_ids)
                & summaries.get("stage", pd.Series(index=summaries.index, dtype=object)).eq("fine")
            ]
        else:
            winner_summaries = pd.DataFrame()
        winner_parameter_columns = [
            name for name in ("rank", "factor_id", "candidate_id", "params_json", "objective")
            if name in winners
        ]
        parameter_names = set()
        if "params_json" in winners:
            for payload in winners["params_json"].dropna():
                try:
                    parameter_names.update(json.loads(str(payload)))
                except (TypeError, ValueError):
                    pass
        winner_parameter_columns.extend(
            name for name in winners.columns
            if name in parameter_names and name not in winner_parameter_columns
        )
        winner_parameters = winners.loc[:, winner_parameter_columns] if winner_parameter_columns else winners
        self.workbook_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.workbook_path.with_name(f".{self.workbook_path.name}.{uuid.uuid4().hex}.tmp.xlsx")
        with pd.ExcelWriter(temporary, engine="openpyxl") as writer:
            _write_sheet(writer, "运行概览", _humanize_columns(overview))
            _write_sheet(writer, "运行配置", _humanize_columns(pd.DataFrame(config_rows)))
            _write_sheet(writer, "参数空间", _humanize_columns(pd.DataFrame(parameter_rows)))
            _write_sheet(writer, "搜索进度", _humanize_columns(progress))
            _write_sheet(writer, "窗口表现", _humanize_columns(windows))
            _write_sheet(writer, "候选汇总", _humanize_columns(summaries))
            _write_sheet(writer, "赢家参数", _humanize_columns(winner_parameters))
            _write_sheet(writer, "赢家汇总", _humanize_columns(winner_summaries))
            _write_sheet(writer, "错误", _humanize_columns(errors))
        _style_workbook(temporary)
        os.replace(temporary, self.workbook_path)
        return self.workbook_path

    def finish(self, result: FactorMiningResult, *, status: str, **updates: Any) -> None:
        windows = self.store.read("window_results")
        summaries = self.store.read("candidate_summary")
        winners = self.store.read("winners")
        errors = self.store.read("errors")
        window_stages = windows.get("stage", pd.Series(index=windows.index, dtype=object))
        summary_stages = summaries.get("stage", pd.Series(index=summaries.index, dtype=object))
        self.write_metadata(
            status=status,
            completed_at_local=datetime.now().astimezone().isoformat(),
            completed_at_utc=datetime.now().astimezone(timezone.utc).isoformat(),
            result_counts={
                "coarse_window_rows": int(window_stages.eq("coarse").sum()),
                "fine_window_rows": int(window_stages.eq("fine").sum()),
                "coarse_candidates": int(summary_stages.eq("coarse").sum()),
                "fine_candidates": int(summary_stages.eq("fine").sum()),
                "selected_candidates": len(winners),
                "errors": len(errors),
            },
            winners=winners.to_dict(orient="records"),
            **updates,
        )
        self.export_workbook()
        self.store.close()


__all__ = ["FactorMiningResult", "MiningResult", "MiningTask", "ResultStore"]
