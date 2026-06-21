"""AI 分析页结构化报告构建器。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from agents.data_provider import DataProvider


def build_ai_analysis_report(
    symbol: str,
    dp: DataProvider,
    agent_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """汇总本地 RQData、实时量价和 Agent 摘要，生成前端可直接渲染的报告。"""
    agent_results = agent_results or {}
    fundamental = _safe_call(dp.get_rqdata_fundamental, symbol, "all", None, 32)
    scorecard = _safe_call(dp.get_rqdata_industry_scorecard, symbol)
    concepts = _safe_call(dp.get_rqdata_concepts, symbol, 20)
    consensus = _safe_call(getattr(dp, "get_rqdata_consensus", lambda *_args, **_kwargs: {}), symbol)

    return {
        "symbol": symbol,
        "name": _company_name(fundamental, symbol),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "fundamental": _build_fundamental_section(fundamental),
        "technical": _build_technical_section(symbol, dp, agent_results),
        "capital": _build_capital_section(symbol, dp, agent_results),
        "news": _build_news_section(concepts, agent_results),
        "valuation": _build_valuation_section(fundamental, scorecard, consensus),
        "source_note": "本页结合 RQData 离线面板、已运行 Agent 摘要和可用实时接口生成，仅供分析参考。",
    }


def _build_fundamental_section(payload: dict[str, Any]) -> dict[str, Any]:
    rows = _annual_rows(payload)
    if not rows:
        return {"available": False, "summary": ["本地 RQData 暂无可用年度财务数据。"], "tables": {}}

    revenue_values = [_format_money_cn(row.get("revenue"), "亿") for row in rows]
    profit_values = [_format_money_cn(row.get("net_profit"), "万") for row in rows]
    roe_values = [_format_percent(row.get("roe"), already_percent=True) for row in rows]
    years = [str(row["year"]) for row in rows]

    latest = rows[-1]
    prev = rows[-2] if len(rows) >= 2 else {}
    revenue_growth = _growth(latest.get("revenue"), prev.get("revenue"))
    profit_growth = _growth(latest.get("net_profit"), prev.get("net_profit"))
    roe_change = _delta(latest.get("roe"), prev.get("roe"))
    debt_change = _delta(latest.get("debt_ratio"), prev.get("debt_ratio"))
    cashflow_change = _growth(latest.get("cashflow_quality"), prev.get("cashflow_quality"))

    summary = [
        f"营业收入: {'-'.join([years[0], years[-1]])} 年分别为{'、'.join(revenue_values)}，最近一年同比{_format_percent(revenue_growth)}。",
        f"净利润: {'-'.join([years[0], years[-1]])} 年分别为{'、'.join(profit_values)}，最近一年同比{_format_percent(profit_growth)}。",
        f"盈利能力: ROE 分别为{'、'.join(roe_values)}，最近一年较上年变化{_format_percent(roe_change, already_percent=True)}。",
        f"资产负债率: 最近一年为{_format_percent(latest.get('debt_ratio'))}，较上年变化{_format_percent(debt_change)}；经营现金流/净利润为{_format_number(latest.get('cashflow_quality'))}。",
        f"现金流: 经营现金流质量最近一年同比{_format_percent(cashflow_change)}，可用于观察利润含金量变化。",
    ]

    financial_table = [
        {
            "年份": row["year"],
            "营业收入(亿元)": _round_or_blank(_scale(row.get("revenue"), 100_000_000)),
            "净利润(万元)": _round_or_blank(_scale(row.get("net_profit"), 10_000)),
            "ROE": _format_percent(row.get("roe"), already_percent=True),
        }
        for row in rows
    ]
    health_table = [
        {
            "指标": "资产负债率",
            **{str(row["year"]): _format_percent(row.get("debt_ratio")) for row in rows},
        },
        {
            "指标": "经营现金流/净利润",
            **{str(row["year"]): _format_number(row.get("cashflow_quality")) for row in rows},
        },
        {
            "指标": "现金及等价物(亿元)",
            **{str(row["year"]): _round_or_blank(_scale(row.get("cash"), 100_000_000)) for row in rows},
        },
    ]

    return {
        "available": True,
        "summary": summary,
        "financial_table": financial_table,
        "health_table": health_table,
    }


def _build_technical_section(
    symbol: str,
    dp: DataProvider,
    agent_results: dict[str, Any],
) -> dict[str, Any]:
    technical_payload = _safe_call(dp.get_rqdata_technical_indicators, symbol)
    rqdata_section = _build_rqdata_technical_section(technical_payload)
    if rqdata_section:
        timeseries_payload = _safe_call(dp.get_rqdata_technical_timeseries, symbol, 30)
        _append_technical_timeseries_summary(rqdata_section, timeseries_payload)
        return rqdata_section

    price_payload = _safe_call(dp.get_price_volume, symbol, "daily", "", "", "qfq")
    price_rows = _price_rows(price_payload)
    agent = (agent_results or {}).get("technical") or {}

    if price_rows:
        frame = pd.DataFrame(price_rows).sort_values("date")
        close = pd.to_numeric(frame["close"], errors="coerce")
        volume = pd.to_numeric(frame["volume"], errors="coerce")
        latest_close = close.dropna().iloc[-1] if close.notna().any() else None
        ma_values = {f"ma{window}": close.rolling(window).mean().iloc[-1] for window in [5, 10, 20, 30, 60] if len(close) >= window}
        support = close.tail(20).min() if close.notna().any() else None
        resistance = close.tail(20).max() if close.notna().any() else None
        latest_volume = volume.dropna().iloc[-1] if volume.notna().any() else None
        avg_volume_10 = volume.tail(10).mean() if volume.notna().any() else None
        volume_shape = "放量" if latest_volume and avg_volume_10 and latest_volume > avg_volume_10 * 1.15 else "缩量"
        trend_text = _ma_trend_text(latest_close, ma_values)

        summary = [
            f"均线系统: 当前股价{_format_price(latest_close)}，{trend_text}",
            f"支撑与压力: 近20日支撑位约{_format_price(support)}，压力位约{_format_price(resistance)}。",
            f"量价关系: 最新成交量相对近10日均量呈现{volume_shape}特征。",
        ]
        table = [{"指标": label.upper(), "数值": _format_price(value)} for label, value in ma_values.items()]
        table.extend(
            [
                {"指标": "近20日支撑", "数值": _format_price(support)},
                {"指标": "近20日压力", "数值": _format_price(resistance)},
                {"指标": "量价形态", "数值": volume_shape},
            ]
        )
        chart = frame.tail(80)[["date", "close"]].to_dict(orient="records")
        return {"available": True, "summary": summary, "table": table, "chart": chart}

    summary = _agent_dimension_summaries(
        agent,
        ["trend", "momentum", "capital_flow"],
        fallback="实时 K 线数据暂不可用，以下为量价 Agent 摘要。",
    )
    if isinstance(technical_payload, dict) and technical_payload.get("source_warning"):
        summary.insert(0, f"米筐技术指标: {technical_payload['source_warning']}")
    return {"available": bool(agent), "summary": summary, "table": _agent_score_table(agent), "chart": []}


def _build_rqdata_technical_section(payload: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    key_indicators = payload.get("key_indicators", {})
    if not isinstance(key_indicators, dict) or not key_indicators:
        return None
    if payload.get("non_null_indicator_count", 0) <= 0:
        return None

    ma_values = {
        key: _safe_number(key_indicators.get(key))
        for key in ["MA5", "MA10", "MA20", "MA30", "MA60"]
        if _safe_number(key_indicators.get(key)) is not None
    }
    macd_values = {
        key: _safe_number(key_indicators.get(key))
        for key in ["MACD_DIFF", "MACD_DEA", "MACD_HIST"]
        if _safe_number(key_indicators.get(key)) is not None
    }
    oscillator_values = {
        key: _safe_number(key_indicators.get(key))
        for key in ["KDJ_K", "KDJ_D", "KDJ_J", "RSI6", "RSI10", "WR"]
        if _safe_number(key_indicators.get(key)) is not None
    }
    boll_values = {
        key: _safe_number(key_indicators.get(key))
        for key in ["BOLL", "BOLL_UP", "BOLL_DOWN", "ATR"]
        if _safe_number(key_indicators.get(key)) is not None
    }
    volume_values = {
        key: _safe_number(key_indicators.get(key))
        for key in ["OBV", "VOL5", "VOL10", "VOL20"]
        if _safe_number(key_indicators.get(key)) is not None
    }

    summary = [
        (
            f"米筐指标: 已读取 {payload.get('date', '最新交易日')} 技术指标缓存，"
            f"{payload.get('indicator_count', 0)} 个指标中 {payload.get('non_null_indicator_count', 0)} 个非空。"
        )
    ]
    if ma_values:
        ma_text = "、".join(f"{key}={_format_number(value)}" for key, value in ma_values.items())
        summary.append(f"均线系统: {ma_text}，{_rq_ma_trend_text(ma_values)}")
    if macd_values:
        macd_text = "、".join(f"{key.replace('MACD_', '')}={_format_number(value)}" for key, value in macd_values.items())
        summary.append(f"MACD: {macd_text}，{_macd_signal_text(macd_values)}")
    if oscillator_values:
        oscillator_text = "、".join(f"{key}={_format_number(value)}" for key, value in oscillator_values.items())
        summary.append(f"摆动指标: {oscillator_text}，{_oscillator_signal_text(oscillator_values)}")
    if boll_values:
        boll_text = "、".join(f"{key}={_format_number(value)}" for key, value in boll_values.items())
        summary.append(f"波动通道: {boll_text}，布林带和 ATR 可用于观察短期波动区间。")
    if volume_values:
        volume_text = "、".join(f"{key}={_format_number(value)}" for key, value in volume_values.items())
        summary.append(f"量能指标: {volume_text}，可结合资金面分析判断量价配合度。")

    if payload.get("source_warning"):
        summary.append(f"数据质量: {payload['source_warning']}")

    table = _rq_indicator_table(
        [
            ("均线", ma_values),
            ("MACD", macd_values),
            ("KDJ/RSI", oscillator_values),
            ("BOLL/波动", boll_values),
            ("量能", volume_values),
        ]
    )
    return {"available": True, "summary": summary, "table": table, "chart": []}


def _append_technical_timeseries_summary(section: dict[str, Any], payload: dict[str, Any]):
    if not isinstance(payload, dict) or not payload.get("data"):
        return
    trend = payload.get("trend_summary") or {}
    lines = [
        (
            f"近月趋势: 基于 {trend.get('window_days', len(payload.get('data', [])))} 个交易日技术指标，"
            f"{trend.get('trend_signal', '趋势信号待确认')}。"
        ),
        trend.get("momentum_signal"),
        trend.get("risk_signal"),
        trend.get("operation_hint"),
    ]
    section.setdefault("summary", []).extend([line for line in lines if line])
    chart_rows = []
    for row in payload.get("data", [])[-30:]:
        chart_rows.append(
            {
                "date": row.get("date"),
                "MA5": row.get("MA5"),
                "MA20": row.get("MA20"),
                "MA60": row.get("MA60"),
                "MACD_HIST": row.get("MACD_HIST"),
                "RSI6": row.get("RSI6"),
            }
        )
    section["chart"] = chart_rows


def _build_capital_section(
    symbol: str,
    dp: DataProvider,
    agent_results: dict[str, Any],
) -> dict[str, Any]:
    agent = (agent_results or {}).get("technical") or {}
    capital_payload = _safe_call(dp.get_capital_flow, symbol, "individual")
    rows = _generic_records(capital_payload)
    latest = rows[0] if rows else {}

    summary = []
    flow_text = _find_first_value(latest, ["主力净流入", "主力净额", "净流入", "净额"])
    if flow_text:
        summary.append(f"主力资金动向: 最新可识别资金指标为 {flow_text}。")
    if agent.get("capital_flow", {}).get("summary"):
        summary.append(f"资金流 Agent: {agent['capital_flow']['summary']}")
    if not summary:
        summary.append("资金流实时明细暂不可用，可结合量价 Agent 与后续米筐资金接口补充。")

    table = []
    for idx, row in enumerate(rows[:5], start=1):
        flat = {str(key): value for key, value in row.items() if not isinstance(value, (dict, list))}
        flat.setdefault("序号", idx)
        table.append(flat)
    if not table:
        table = _agent_score_table(agent)
    return {"available": bool(rows or agent), "summary": summary, "table": table}


def _build_news_section(
    concepts: dict[str, Any],
    agent_results: dict[str, Any],
) -> dict[str, Any]:
    agent = (agent_results or {}).get("sentiment") or {}
    concept_names = concepts.get("concept_names", []) if isinstance(concepts, dict) else []
    events = agent.get("news_events", []) if isinstance(agent, dict) else []

    summary = []
    if concept_names:
        summary.append(f"主题标签: 公司关联概念包括{'、'.join(concept_names[:8])}。")
    summary.extend(
        event.get("description", "")
        for event in events[:3]
        if isinstance(event, dict) and event.get("description")
    )
    summary.extend(_agent_dimension_summaries(agent, ["market_emotion", "social_heat", "overall_market_emotion"], fallback=""))
    if not summary:
        summary.append("资讯与舆情摘要暂不可用，后续可接入米筐公告、研报和新闻 API 补强。")

    sources = []
    for item in events[:6]:
        if isinstance(item, dict):
            text = item.get("source") or item.get("type") or item.get("description")
            if text:
                sources.append(str(text)[:30])
    sources.extend(concept_names[:6])
    return {"available": bool(summary), "summary": summary[:6], "sources": sources[:8]}


def _build_valuation_section(
    payload: dict[str, Any],
    scorecard: dict[str, Any],
    consensus: dict[str, Any] | None = None,
) -> dict[str, Any]:
    combined = _combined_rows(payload)
    if not combined:
        return {"available": False, "summary": ["估值数据暂不可用。"], "table": []}

    latest = combined[0]
    historical = pd.DataFrame(combined)
    metrics = [
        ("pe_ratio_ttm", "市盈率(TTM)", "P/E"),
        ("pb_ratio", "市净率", "P/B"),
        ("pcf_ratio", "市现率(经营性现金流)", "P/CF"),
        ("ps_ratio_ttm", "市销率", "P/S"),
    ]
    score_metrics = {item.get("metric"): item for item in scorecard.get("metrics", [])} if isinstance(scorecard, dict) else {}

    summary = []
    table = []
    for column, label, short in metrics:
        current = _safe_number(latest.get(column))
        if column not in historical.columns:
            hist_avg = None
        else:
            values = pd.to_numeric(historical[column], errors="coerce")
            if column in {"pe_ratio_ttm", "pb_ratio", "ps_ratio_ttm", "pcf_ratio"}:
                values = values.where(values > 0)
            hist_avg = values.mean()
        peer = score_metrics.get(column, {})
        rank = peer.get("rank")
        peer_count = peer.get("peer_count")
        rank_text = f"{rank}/{peer_count}" if rank and peer_count else "—"
        summary.append(
            f"{label}: 当前值为{_format_number(current)}，行业排名{rank_text}，历史均值{_format_number(hist_avg)}。"
        )
        table.append(
            {
                "指标": label,
                "当前值": _format_number(current),
                "行业排名": rank_text,
                "历史均值": _format_number(hist_avg),
                "说明": short,
            }
        )

    consensus_summary = consensus.get("summary", {}) if isinstance(consensus, dict) else {}
    if consensus_summary:
        summary.append(
            "分析师一致预期: "
            f"覆盖{consensus_summary.get('institute_count', 0)}家机构，"
            f"平均目标价{_format_price(consensus_summary.get('target_price_mean'))}，"
            f"较当前价格隐含空间{_format_percent(consensus_summary.get('target_price_upside'))}，"
            f"平均评级为{consensus_summary.get('rating_label', '—')}。"
        )
        table.extend(
            [
                {
                    "指标": "分析师平均目标价",
                    "当前值": _format_price(consensus_summary.get("target_price_mean")),
                    "行业排名": "—",
                    "历史均值": "—",
                    "说明": f"覆盖{consensus_summary.get('institute_count', 0)}家机构",
                },
                {
                    "指标": "目标价隐含空间",
                    "当前值": _format_percent(consensus_summary.get("target_price_upside")),
                    "行业排名": "—",
                    "历史均值": "—",
                    "说明": "目标价/现价-1",
                },
                {
                    "指标": "平均评级系数",
                    "当前值": _format_number(consensus_summary.get("rating_coef_mean")),
                    "行业排名": "—",
                    "历史均值": "—",
                    "说明": consensus_summary.get("rating_label", "—"),
                },
            ]
        )
    return {"available": True, "summary": summary, "table": table}


def _annual_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in _combined_rows(payload):
        quarter = str(row.get("quarter", "")).lower()
        if not quarter.endswith("q4"):
            continue
        revenue = _pick(row, ["operating_revenue_ttm_0", "operating_revenue"])
        net_profit = _pick(row, ["net_profit_ttm_0", "net_profit", "net_profit_parent_company"])
        total_assets = _pick(row, ["total_assets"])
        total_liabilities = _pick(row, ["total_liabilities"])
        operating_cashflow = _pick(row, ["net_operate_cashflowTTM"])
        rows.append(
            {
                "year": quarter[:4],
                "revenue": revenue,
                "net_profit": net_profit,
                "roe": _pick(row, ["return_on_equity_weighted_average"]),
                "debt_ratio": _safe_divide(total_liabilities, total_assets),
                "cashflow_quality": _safe_divide(operating_cashflow, abs(net_profit) if net_profit else None),
                "cash": _pick(row, ["cash_equivalent", "pit_cash_equivalent"]),
            }
        )
    rows = sorted(rows, key=lambda item: item["year"])
    return rows[-5:]


def _combined_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    by_quarter: dict[str, dict[str, Any]] = {}
    for section in ["valuation", "financial_indicator", "balance_sheet", "profit_sheet", "cash_flow", "pit_disclosures"]:
        for row in payload.get(section, []) if isinstance(payload, dict) else []:
            if not isinstance(row, dict):
                continue
            quarter = str(row.get("quarter", "")).lower()
            if not quarter:
                continue
            by_quarter.setdefault(quarter, {}).update(row)

    rows = []
    for quarter, row in by_quarter.items():
        row = row.copy()
        row["quarter"] = quarter
        market_cap = _pick(row, ["market_cap", "total_market_cap"])
        operating_cashflow = _pick(row, ["net_operate_cashflowTTM"])
        row["pcf_ratio"] = _safe_divide(market_cap, operating_cashflow)
        rows.append(row)
    return sorted(rows, key=lambda item: _quarter_key(item.get("quarter")), reverse=True)


def _price_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = payload.get("data", []) if isinstance(payload, dict) else []
    rows = []
    for item in data:
        if not isinstance(item, dict):
            continue
        date_value = _pick_text(item, ["日期", "date", "trade_date", "交易日"])
        close = _pick(item, ["收盘", "close", "收盘价"])
        volume = _pick(item, ["成交量", "volume"])
        turnover_rate = _pick(item, ["换手率", "turnover_rate", "turnover"])
        if date_value and close is not None:
            rows.append(
                {
                    "date": str(date_value),
                    "close": close,
                    "volume": volume,
                    "turnover_rate": turnover_rate,
                }
            )
    return rows


def _agent_dimension_summaries(agent: dict[str, Any], keys: list[str], fallback: str) -> list[str]:
    summaries = []
    for key in keys:
        item = agent.get(key, {}) if isinstance(agent, dict) else {}
        if isinstance(item, dict) and item.get("summary"):
            summaries.append(f"{item.get('dimension', key)}: {item['summary']}")
    if not summaries and fallback:
        summaries.append(fallback)
    return summaries


def _agent_score_table(agent: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key, value in (agent or {}).items():
        if isinstance(value, dict) and "score" in value:
            rows.append({"指标": value.get("dimension", key), "数值": value.get("score"), "摘要": value.get("summary", "")})
    return rows


def _generic_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            return [item for item in payload["data"] if isinstance(item, dict)]
        for value in payload.values():
            if isinstance(value, list):
                records = [item for item in value if isinstance(item, dict)]
                if records:
                    return records
    return []


def _company_name(payload: dict[str, Any], fallback: str) -> str:
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    return metadata.get("symbol") or metadata.get("name") or fallback


def _safe_call(func, *args, **kwargs):
    try:
        result = func(*args, **kwargs)
        return result if isinstance(result, dict) else {}
    except Exception:
        return {}


def _pick(row: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = _safe_number(row.get(key))
        if value is not None:
            return value
    return None


def _pick_text(row: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value not in {None, ""}:
            return str(value)
    return ""


def _find_first_value(row: dict[str, Any], key_parts: list[str]) -> str:
    for key, value in row.items():
        text = str(key)
        if any(part in text for part in key_parts):
            return f"{text}={value}"
    return ""


def _safe_number(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_divide(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in {None, 0}:
        return None
    return numerator / denominator


def _growth(current: float | None, previous: float | None) -> float | None:
    if current is None or previous in {None, 0}:
        return None
    return (current - previous) / abs(previous)


def _delta(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None:
        return None
    return current - previous


def _scale(value: float | None, scale: float) -> float | None:
    return value / scale if value is not None else None


def _round_or_blank(value: float | None, digits: int = 2) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def _format_money_cn(value: float | None, unit: str) -> str:
    scale = 100_000_000 if unit == "亿" else 10_000
    return "—" if value is None else f"{value / scale:.2f}{unit}元"


def _format_percent(value: float | None, already_percent: bool = False) -> str:
    if value is None:
        return "—"
    number = value if already_percent else value * 100
    return f"{number:.2f}%"


def _format_number(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _format_price(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}元"


def _rq_indicator_table(groups: list[tuple[str, dict[str, float]]]) -> list[dict[str, Any]]:
    rows = []
    for group_name, values in groups:
        for key, value in values.items():
            rows.append({"类别": group_name, "指标": key, "数值": _format_number(value)})
    return rows


def _rq_ma_trend_text(ma_values: dict[str, float]) -> str:
    ordered_keys = [key for key in ["MA5", "MA10", "MA20", "MA30", "MA60"] if key in ma_values]
    if len(ordered_keys) < 3:
        return "均线样本较少，暂以数值展示为主。"

    values = [ma_values[key] for key in ordered_keys]
    if all(values[idx] >= values[idx + 1] for idx in range(len(values) - 1)):
        return "短期均线位于中长期均线之上，趋势结构偏强。"
    if all(values[idx] <= values[idx + 1] for idx in range(len(values) - 1)):
        return "短期均线位于中长期均线之下，趋势结构偏弱。"
    return "不同周期均线交织，趋势方向仍有分化。"


def _macd_signal_text(values: dict[str, float]) -> str:
    hist = values.get("MACD_HIST")
    diff = values.get("MACD_DIFF")
    dea = values.get("MACD_DEA")
    if hist is not None:
        if hist > 0:
            return "柱线为正，短线动能相对改善。"
        if hist < 0:
            return "柱线为负，短线动能仍偏弱。"
    if diff is not None and dea is not None:
        return "DIFF 高于 DEA，动能边际偏积极。" if diff >= dea else "DIFF 低于 DEA，动能边际偏谨慎。"
    return "MACD 信号不完整，暂以数值跟踪为主。"


def _oscillator_signal_text(values: dict[str, float]) -> str:
    rsi = values.get("RSI6") or values.get("RSI10")
    k = values.get("KDJ_K")
    d = values.get("KDJ_D")
    notes = []
    if rsi is not None:
        if rsi >= 70:
            notes.append("RSI 进入偏热区间")
        elif rsi <= 30:
            notes.append("RSI 进入偏冷区间")
        else:
            notes.append("RSI 处于中性区间")
    if k is not None and d is not None:
        notes.append("KDJ 短线偏强" if k >= d else "KDJ 短线偏弱")
    return "，".join(notes) + "。" if notes else "摆动指标信号不完整，暂以数值跟踪为主。"


def _ma_trend_text(latest_close: float | None, ma_values: dict[str, float]) -> str:
    if latest_close is None or not ma_values:
        return "均线数据不足。"
    below = [name.upper() for name, value in ma_values.items() if pd.notna(value) and latest_close < value]
    above = [name.upper() for name, value in ma_values.items() if pd.notna(value) and latest_close >= value]
    if below and not above:
        return f"已跌破{'/'.join(below)}，技术面偏弱。"
    if above and not below:
        return f"站上{'/'.join(above)}，趋势相对强势。"
    return f"站上{'/'.join(above)}，但低于{'/'.join(below)}，趋势分化。"


def _quarter_key(value: Any) -> tuple[int, int]:
    text = str(value or "").lower()
    if "q" not in text:
        return (0, 0)
    year, quarter = text.split("q", 1)
    try:
        return (int(year), int(quarter))
    except ValueError:
        return (0, 0)
