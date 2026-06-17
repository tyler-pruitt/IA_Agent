"""原始数据快照工具，用于 UI 展示和分析审计。"""

from __future__ import annotations

from typing import Any


def summarize_raw_data(raw_data: dict[str, Any], sample_rows: int = 5) -> dict[str, Any]:
    """把 Agent 获取到的原始数据压缩成可展示摘要。"""
    summary: dict[str, Any] = {}
    errors = raw_data.get("errors", [])
    if errors:
        summary["data_quality"] = {
            "errors": errors[:20],
            "error_count": len(errors),
        }
    if raw_data.get("freshness"):
        summary["data_freshness"] = raw_data["freshness"]

    for key, value in raw_data.items():
        if key in {"errors", "freshness"}:
            continue
        summary[key] = _summarize_value(value, sample_rows)

    return summary


def _summarize_value(value: Any, sample_rows: int) -> Any:
    if isinstance(value, list):
        return _summarize_records(value, sample_rows)

    if isinstance(value, dict):
        nested = {"type": "dict", "keys": list(value.keys())[:30]}
        for key, item in value.items():
            if isinstance(item, list):
                nested[key] = _summarize_records(item, sample_rows)
            elif isinstance(item, dict):
                nested[key] = {
                    "type": "dict",
                    "keys": list(item.keys())[:30],
                    "sample": item,
                }
            elif key == "error" or key.endswith("_error"):
                nested[key] = item
            elif key.endswith(("_latest_date", "_note", "_warning")):
                nested[key] = item
            elif key in {
                "date",
                "preview_type",
                "note",
                "date_checked",
                "scope",
                "symbol",
                "period",
                "market",
                "source",
                "source_note",
                "source_warning",
                "sort_order",
                "query_scope",
                "primary_source_error",
            }:
                nested[key] = item
        return nested

    return value


def _summarize_records(records: list[Any], sample_rows: int) -> dict[str, Any]:
    fields = []
    first_record = next((item for item in records if isinstance(item, dict)), None)
    if first_record is not None:
        fields = list(first_record.keys())

    return {
        "type": "records",
        "rows": len(records),
        "fields": fields[:40],
        "sample": records[:sample_rows],
    }
