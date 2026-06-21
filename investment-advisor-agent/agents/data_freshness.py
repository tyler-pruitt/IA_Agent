"""数据时效校验工具。

Agent 拉取 MCP 数据后先经过这里判断最新日期。核心数据明显滞后时，
该 Agent 会被标记为不参与最终评分；滞后的子数据块也会从 prompt 数据中移除。
"""

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timedelta
import re
from typing import Any

import pandas as pd

from config.settings import DATA_FRESHNESS_CHECK_ENABLED, DATA_FRESHNESS_POLICIES

_EMPTY_STRINGS = {"", "nan", "none", "null", "NaN", "None", "--", "-"}
_IGNORED_DATE_KEYS = {"date_checked"}
_DATE_KEYWORDS = (
    "日期",
    "时间",
    "公告",
    "报告",
    "交易日",
    "上榜日",
    "date",
    "time",
    "day",
    "report",
    "notice",
    "update",
    "trade",
)

_DIMENSION_LABELS = {
    "fundamental": "基本面",
    "technical": "量价",
    "sentiment": "舆情",
    "macro": "宏观",
}


def apply_freshness_policy(agent_key: str, raw_data: dict[str, Any]) -> dict[str, Any]:
    """返回带 `freshness` 元数据的数据副本，并屏蔽滞后子块。"""
    data = deepcopy(raw_data)
    report = evaluate_freshness(agent_key, data)
    data["freshness"] = report

    if not DATA_FRESHNESS_CHECK_ENABLED:
        return data

    stale_raw = {}
    for check in report.get("checks", []):
        section = check.get("section")
        if check.get("status") == "stale" and section in data:
            stale_raw[section] = data[section]
            data[section] = _empty_like(data[section])
    if stale_raw:
        data["stale_raw"] = stale_raw

    notes = freshness_notes(report)
    if notes:
        data.setdefault("errors", []).extend(notes)
    return data


def evaluate_freshness(agent_key: str, raw_data: dict[str, Any]) -> dict[str, Any]:
    """根据配置评估某个 Agent 原始数据的时效性。"""
    policy = DATA_FRESHNESS_POLICIES.get(agent_key, {})
    sections = policy.get("sections", {})
    checks = []

    for section, max_age_days in sections.items():
        payload = raw_data.get(section)
        latest = latest_date(payload)
        is_empty = _is_empty(payload)

        if is_empty:
            status = "empty"
            age_days = None
        elif latest is None:
            status = "unknown_date"
            age_days = None
        else:
            age_days = (date.today() - latest).days
            status = "fresh" if age_days <= max_age_days else "stale"

        checks.append(
            {
                "section": section,
                "latest_date": latest.isoformat() if latest else "",
                "age_days": age_days,
                "max_age_days": max_age_days,
                "status": status,
                "required": _is_required(section, policy),
            }
        )

    usable, reason = _evaluate_agent_gate(policy, checks)
    if not DATA_FRESHNESS_CHECK_ENABLED:
        usable = True
        reason = "数据时效校验已关闭，所有已返回 Agent 默认参与评分。"

    return {
        "agent": agent_key,
        "agent_label": _DIMENSION_LABELS.get(agent_key, agent_key),
        "usable": usable,
        "reason": reason,
        "checks": checks,
        "stale_sections": [
            item["section"] for item in checks if item.get("status") == "stale"
        ],
        "missing_sections": [
            item["section"] for item in checks if item.get("status") == "empty"
        ],
        "unknown_date_sections": [
            item["section"] for item in checks if item.get("status") == "unknown_date"
        ],
    }


def latest_date(payload: Any) -> date | None:
    """递归提取 payload 中的最新有效日期。"""
    dates: list[date] = []
    _collect_dates(payload, dates)
    return max(dates) if dates else None


def freshness_notes(report: dict[str, Any]) -> list[str]:
    """把时效报告转换为可展示/传给 LLM 的数据质量说明。"""
    notes = []
    label = report.get("agent_label", report.get("agent", "Agent"))
    if not report.get("usable", True):
        notes.append(f"{label}.freshness_disabled: {report.get('reason', '')}")

    for check in report.get("checks", []):
        section = check.get("section")
        status = check.get("status")
        if status == "stale":
            notes.append(
                f"{label}.{section}.stale: 最新日期 {check.get('latest_date')}, "
                f"距今 {check.get('age_days')} 天，超过阈值 {check.get('max_age_days')} 天"
            )
        elif status == "unknown_date":
            notes.append(f"{label}.{section}.unknown_date: 未识别到可校验日期字段")
    return notes


def freshness_reason(report: dict[str, Any], fallback: str = "数据时效未通过") -> str:
    """提取给 UI/决策层使用的简短原因。"""
    return report.get("reason") or fallback


def result_is_fresh(result: Any) -> bool:
    """判断 Agent 结果是否可参与最终评分。"""
    if result is None:
        return False
    summary = getattr(result, "raw_data_summary", {}) or {}
    freshness = summary.get("data_freshness", {})
    if isinstance(freshness, dict) and freshness.get("usable") is False:
        return False
    return True


def result_freshness_reason(result: Any, label: str) -> str:
    """返回某个结果被禁用或未返回的原因。"""
    if result is None:
        return f"{label} Agent 未返回结果，暂不参与评分。"
    summary = getattr(result, "raw_data_summary", {}) or {}
    freshness = summary.get("data_freshness", {})
    if isinstance(freshness, dict) and freshness.get("usable") is False:
        return freshness_reason(freshness, f"{label}数据时效未通过，暂不参与评分。")
    return ""


def _evaluate_agent_gate(
    policy: dict[str, Any],
    checks: list[dict[str, Any]],
) -> tuple[bool, str]:
    check_by_section = {item["section"]: item for item in checks}
    required_all = policy.get("required_all", [])
    required_any = policy.get("required_any", [])

    failed_all = [
        section
        for section in required_all
        if check_by_section.get(section, {}).get("status") != "fresh"
    ]
    if failed_all:
        return False, f"核心数据 {', '.join(failed_all)} 未通过时效校验，Agent 暂不参与评分。"

    if required_any:
        has_fresh_required = any(
            check_by_section.get(section, {}).get("status") == "fresh"
            for section in required_any
        )
        if not has_fresh_required:
            return False, f"核心数据 {', '.join(required_any)} 均未通过时效校验，Agent 暂不参与评分。"

    stale_required = [
        item["section"]
        for item in checks
        if item.get("required") and item.get("status") == "stale"
    ]
    if stale_required:
        return False, f"核心数据 {', '.join(stale_required)} 已滞后，Agent 暂不参与评分。"

    return True, "数据时效校验通过。"


def _is_required(section: str, policy: dict[str, Any]) -> bool:
    return section in policy.get("required_all", []) or section in policy.get("required_any", [])


def _collect_dates(payload: Any, dates: list[date], key_hint: str = "") -> None:
    if payload is None:
        return

    if isinstance(payload, dict):
        for key, value in payload.items():
            key_text = str(key)
            if key_text in _IGNORED_DATE_KEYS:
                continue
            if _is_date_key(key_text):
                dates.extend(_parse_date_values(value))
            _collect_dates(value, dates, key_text)
        return

    if isinstance(payload, list):
        for item in payload[:500]:
            _collect_dates(item, dates, key_hint)
        return

    if key_hint and _is_date_key(key_hint):
        parsed = _parse_single_date(payload)
        if parsed is not None:
            dates.append(parsed)


def _parse_date_values(value: Any) -> list[date]:
    if isinstance(value, list):
        parsed = []
        for item in value:
            if isinstance(item, (dict, list)):
                continue
            date_value = _parse_single_date(item)
            if date_value is not None:
                parsed.append(date_value)
        return parsed
    if isinstance(value, dict):
        value = value.get("date") or value.get("日期") or value.get("TRADE_DATE")
    parsed = _parse_single_date(value)
    return [parsed] if parsed is not None else []


def _parse_single_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _valid_date(value.date())
    if isinstance(value, date):
        return _valid_date(value)

    text = str(value).strip()
    if not text or text in _EMPTY_STRINGS:
        return None

    text = (
        text.replace("年", "-")
        .replace("月", "-")
        .replace("日", "")
        .replace("/", "-")
        .strip()
    )
    if re.fullmatch(r"\d{8}", text):
        text = f"{text[:4]}-{text[4:6]}-{text[6:]}"

    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return _valid_date(parsed.date())


def _valid_date(value: date) -> date | None:
    # 宏观日历有时会包含未来公布日，不能用来证明数据已经可用。
    if value > date.today() + timedelta(days=7):
        return None
    if value.year < 1990:
        return None
    return value


def _is_date_key(key: str) -> bool:
    lowered = key.lower()
    return any(keyword in key or keyword in lowered for keyword in _DATE_KEYWORDS)


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, set)):
        return len(value) == 0
    if isinstance(value, dict):
        data_keys = [key for key in value if key not in {"scope", "symbol", "period", "market"}]
        if not data_keys:
            return True
        if "data" in value and value.get("data") == []:
            return True
    return False


def _empty_like(value: Any) -> Any:
    if isinstance(value, list):
        return []
    if isinstance(value, dict):
        return {}
    return None
