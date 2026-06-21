"""本地数据仓储。

这一层保存 MCP 工具返回的原始 payload，并用 SQLite 建索引。它和
akshare_mcp_server.cache.DataCache 的区别是: DataCache 是函数级 TTL 缓存，
这里是可按股票/市场/宏观/板块检索和审计的数据沉淀层。
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any

from agents.data_freshness import latest_date
from akshare_mcp_server.cache import TTL_CONFIG
from config.settings import (
    LOCAL_DATA_STORE_DIR,
    LOCAL_DATA_STORE_HISTORY_ENABLED,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "mcp_payload_v1"

TOOL_DATA_TYPE = {
    "get_stock_fundamental": "fundamental",
    "get_stock_profit_forecast": "fundamental",
    "get_stock_earnings_preview": "fundamental",
    "get_stock_price_volume": "quote",
    "get_stock_capital_flow": "quote",
    "get_stock_lhb": "quote",
    "get_stock_margin_detail": "quote",
    "get_stock_sentiment": "sentiment",
    "get_stock_news": "news",
    "get_stock_market_emotion": "sentiment",
    "get_macro_china_overview": "macro",
    "get_macro_global_interest": "macro",
    "get_stock_sector_analysis": "sector",
    "get_stock_market_valuation": "sector",
}


class LocalDataStore:
    """SQLite 索引 + JSON payload 的本地数据仓储。"""

    def __init__(self, root_dir: str | None = None, history_enabled: bool | None = None):
        self.root_dir = Path(root_dir or LOCAL_DATA_STORE_DIR).expanduser()
        self.payload_root = self.root_dir / "payloads"
        self.db_path = self.root_dir / "data_store.sqlite"
        self.history_enabled = (
            LOCAL_DATA_STORE_HISTORY_ENABLED
            if history_enabled is None
            else history_enabled
        )
        self._hit_count = 0
        self._miss_count = 0
        self._write_count = 0

        self.payload_root.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def get_fresh(self, tool: str, arguments: dict[str, Any] | None = None) -> dict | None:
        """读取未过期且无错误的本地 payload。"""
        arguments = arguments or {}
        cache_key = make_cache_key(tool, arguments)
        now = _utc_now_iso()

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT payload_path, expires_at, error_count
                FROM datasets
                WHERE cache_key = ?
                """,
                (cache_key,),
            ).fetchone()

        if row is None:
            self._miss_count += 1
            return None
        if row["error_count"] and row["error_count"] > 0:
            self._miss_count += 1
            return None
        if row["expires_at"] <= now:
            self._miss_count += 1
            return None

        payload_path = self.root_dir / row["payload_path"]
        if not payload_path.exists():
            self._miss_count += 1
            return None

        try:
            wrapper = json.loads(payload_path.read_text(encoding="utf-8"))
            payload = wrapper.get("payload")
        except Exception:
            logger.debug("读取本地 payload 失败: %s", payload_path, exc_info=True)
            self._miss_count += 1
            return None

        if not isinstance(payload, dict):
            self._miss_count += 1
            return None

        self._hit_count += 1
        return payload

    def save(
        self,
        tool: str,
        arguments: dict[str, Any] | None,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """保存 MCP payload，并返回写入元数据。"""
        arguments = arguments or {}
        if not isinstance(payload, dict):
            payload = {"payload": payload}

        now = _utc_now()
        fetched_at = _iso(now)
        data_type = TOOL_DATA_TYPE.get(tool, "quote")
        ttl = TTL_CONFIG.get(data_type, 300)
        expires_at = _iso_from_timestamp(now.timestamp() + ttl)
        cache_key = make_cache_key(tool, arguments)
        payload_hash = "sha256:" + _stable_hash(payload)
        location = classify_dataset(tool, arguments, payload)
        row_count, field_count, error_count = payload_stats(payload)
        latest = latest_date(payload)

        base_dir = self.payload_root.joinpath(*location["path_parts"])
        base_dir.mkdir(parents=True, exist_ok=True)
        safe_timestamp = fetched_at.replace(":", "-")
        history_name = f"{safe_timestamp}.json"
        latest_path = base_dir / "latest.json"
        history_path = base_dir / history_name

        meta = {
            "schema_version": SCHEMA_VERSION,
            "tool": tool,
            "arguments": arguments,
            "scope": location["scope"],
            "entity_type": location["entity_type"],
            "entity_id": location["entity_id"],
            "category": location["category"],
            "dataset": location["dataset"],
            "fetched_at": fetched_at,
            "expires_at": expires_at,
            "payload_hash": payload_hash,
            "row_count": row_count,
            "field_count": field_count,
            "latest_date": latest.isoformat() if latest else "",
            "freshness_status": "unknown",
            "error_count": error_count,
            "source": payload.get("source") or payload.get("scope") or "",
        }
        wrapper = {"schema_version": SCHEMA_VERSION, "meta": meta, "payload": payload}

        serialized = json.dumps(wrapper, ensure_ascii=False, indent=2, default=str)
        latest_path.write_text(serialized, encoding="utf-8")
        if self.history_enabled:
            history_path.write_text(serialized, encoding="utf-8")

        payload_rel_path = latest_path.relative_to(self.root_dir).as_posix()
        self._upsert_dataset(
            cache_key=cache_key,
            tool=tool,
            arguments=arguments,
            payload_path=payload_rel_path,
            payload_hash=payload_hash,
            meta=meta,
            fetched_at=fetched_at,
            expires_at=expires_at,
        )
        self._write_count += 1
        return meta | {"cache_key": cache_key, "payload_path": payload_rel_path}

    def clear_expired(self) -> int:
        """删除过期索引记录；payload 文件保留用于审计。"""
        now = _utc_now_iso()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM datasets WHERE expires_at <= ?", (now,))
            return cursor.rowcount

    @property
    def stats(self) -> dict[str, Any]:
        total = self._hit_count + self._miss_count
        hit_rate = self._hit_count / total if total else 0
        return {
            "hits": self._hit_count,
            "misses": self._miss_count,
            "writes": self._write_count,
            "hit_rate": f"{hit_rate:.1%}",
            "db_path": str(self.db_path),
        }

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  cache_key TEXT UNIQUE NOT NULL,
                  scope TEXT NOT NULL,
                  entity_type TEXT NOT NULL,
                  entity_id TEXT NOT NULL,
                  category TEXT NOT NULL,
                  dataset TEXT NOT NULL,
                  tool TEXT NOT NULL,
                  arguments_json TEXT NOT NULL,
                  payload_path TEXT NOT NULL,
                  payload_hash TEXT NOT NULL,
                  fetched_at TEXT NOT NULL,
                  expires_at TEXT NOT NULL,
                  latest_date TEXT,
                  freshness_status TEXT,
                  row_count INTEGER DEFAULT 0,
                  field_count INTEGER DEFAULT 0,
                  error_count INTEGER DEFAULT 0,
                  source TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_datasets_entity ON datasets(scope, entity_type, entity_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_datasets_tool ON datasets(tool)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_datasets_expires ON datasets(expires_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_datasets_latest_date ON datasets(latest_date)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dataset_sections (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  dataset_id INTEGER NOT NULL,
                  section TEXT NOT NULL,
                  row_count INTEGER DEFAULT 0,
                  field_count INTEGER DEFAULT 0,
                  latest_date TEXT,
                  freshness_status TEXT,
                  error TEXT,
                  fields_json TEXT,
                  FOREIGN KEY(dataset_id) REFERENCES datasets(id)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sections_dataset ON dataset_sections(dataset_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sections_section ON dataset_sections(section)")

    def _upsert_dataset(
        self,
        cache_key: str,
        tool: str,
        arguments: dict[str, Any],
        payload_path: str,
        payload_hash: str,
        meta: dict[str, Any],
        fetched_at: str,
        expires_at: str,
    ):
        now = _utc_now_iso()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO datasets (
                    cache_key, scope, entity_type, entity_id, category, dataset,
                    tool, arguments_json, payload_path, payload_hash, fetched_at,
                    expires_at, latest_date, freshness_status, row_count, field_count,
                    error_count, source, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    scope=excluded.scope,
                    entity_type=excluded.entity_type,
                    entity_id=excluded.entity_id,
                    category=excluded.category,
                    dataset=excluded.dataset,
                    tool=excluded.tool,
                    arguments_json=excluded.arguments_json,
                    payload_path=excluded.payload_path,
                    payload_hash=excluded.payload_hash,
                    fetched_at=excluded.fetched_at,
                    expires_at=excluded.expires_at,
                    latest_date=excluded.latest_date,
                    freshness_status=excluded.freshness_status,
                    row_count=excluded.row_count,
                    field_count=excluded.field_count,
                    error_count=excluded.error_count,
                    source=excluded.source,
                    updated_at=excluded.updated_at
                """,
                (
                    cache_key,
                    meta["scope"],
                    meta["entity_type"],
                    meta["entity_id"],
                    meta["category"],
                    meta["dataset"],
                    tool,
                    json.dumps(arguments, ensure_ascii=False, sort_keys=True),
                    payload_path,
                    payload_hash,
                    fetched_at,
                    expires_at,
                    meta["latest_date"],
                    meta["freshness_status"],
                    meta["row_count"],
                    meta["field_count"],
                    meta["error_count"],
                    meta["source"],
                    now,
                    now,
                ),
            )
            dataset_id = conn.execute(
                "SELECT id FROM datasets WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()[0]
            conn.execute("DELETE FROM dataset_sections WHERE dataset_id = ?", (dataset_id,))
            for section in payload_sections(meta, tool, arguments):
                conn.execute(
                    """
                    INSERT INTO dataset_sections (
                        dataset_id, section, row_count, field_count, latest_date,
                        freshness_status, error, fields_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        dataset_id,
                        section["section"],
                        section["row_count"],
                        section["field_count"],
                        section["latest_date"],
                        section["freshness_status"],
                        section["error"],
                        json.dumps(section["fields"], ensure_ascii=False),
                    ),
                )


def make_cache_key(tool: str, arguments: dict[str, Any] | None = None) -> str:
    """生成稳定 cache key。"""
    raw = json.dumps(
        {"tool": tool, "arguments": arguments or {}},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def classify_dataset(tool: str, arguments: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """根据工具名和参数映射本地存储分区。"""
    symbol = _plain_symbol(arguments.get("symbol") or payload.get("symbol") or "")
    scope_arg = arguments.get("scope") or payload.get("scope") or ""

    if tool == "get_stock_fundamental":
        report_type = arguments.get("report_type", "all")
        return _stock_location(symbol, "fundamental", report_type)
    if tool == "get_stock_profit_forecast":
        return _stock_location(symbol, "fundamental", "forecast")
    if tool == "get_stock_earnings_preview":
        preview_type = arguments.get("preview_type", "yjyg")
        date_arg = arguments.get("date", "")
        return _stock_location(symbol or "all", "fundamental", f"earnings/{preview_type}/{date_arg}")
    if tool == "get_stock_price_volume":
        period = arguments.get("period", "daily")
        adjust = arguments.get("adjust", "qfq") or "none"
        return _stock_location(symbol, "technical", f"price_volume/{period}_{adjust}")
    if tool == "get_stock_capital_flow" and scope_arg == "individual":
        return _stock_location(symbol, "technical", "capital_flow/individual")
    if tool == "get_stock_sentiment":
        return _stock_location(symbol, "sentiment", "sentiment")
    if tool == "get_stock_news" and scope_arg == "individual":
        return _stock_location(symbol, "sentiment", "news/individual")

    if tool == "get_stock_capital_flow":
        return _shared_location("market", "market", "all", "capital_flow", scope_arg or "market")
    if tool == "get_stock_lhb":
        detail_type = arguments.get("detail_type", "detail")
        return _shared_location("market", "market", "all", "lhb", detail_type)
    if tool == "get_stock_margin_detail":
        return _shared_location("market", "market", "all", "margin", "account")
    if tool == "get_stock_market_emotion":
        return _shared_location("market", "market", "all", "emotion", arguments.get("date") or "latest")
    if tool == "get_stock_market_valuation":
        return _shared_location("market", "market", "all", "valuation", arguments.get("index", "default"))
    if tool == "get_macro_china_overview":
        return _shared_location("macro", "macro", "global", "macro", "china_overview")
    if tool == "get_macro_global_interest":
        return _shared_location("macro", "macro", "global", "macro", "global_interest")
    if tool == "get_stock_sector_analysis":
        sector_type = arguments.get("sector_type", "industry")
        name = arguments.get("name", "default")
        return _shared_location("sector", sector_type, name, "sector", name)

    return _shared_location("misc", "misc", "global", "misc", tool)


def payload_stats(payload: dict[str, Any]) -> tuple[int, int, int]:
    """统计 payload 中的行数、字段数和错误数。"""
    rows = 0
    fields: set[str] = set()
    errors = 0

    def walk(value: Any):
        nonlocal rows, errors
        if isinstance(value, dict):
            for key, item in value.items():
                if (key == "error" or str(key).endswith("_error")) and item:
                    errors += 1
                walk(item)
        elif isinstance(value, list):
            rows += len(value)
            for item in value:
                if isinstance(item, dict):
                    fields.update(str(key) for key in item.keys())
                    break
            for item in value[:20]:
                walk(item)

    walk(payload)
    return rows, len(fields), errors


def payload_sections(meta: dict[str, Any], tool: str, arguments: dict[str, Any]) -> list[dict[str, Any]]:
    """根据已保存的 latest payload 生成 section 摘要。"""
    # 该函数在 _upsert_dataset 中调用时只需要表字段。为了避免重复读文件，
    # 先插入顶层汇总；后续若 UI 需要更细粒度，可扩展为传入 payload。
    return [
        {
            "section": meta["dataset"],
            "row_count": meta["row_count"],
            "field_count": meta["field_count"],
            "latest_date": meta["latest_date"],
            "freshness_status": meta["freshness_status"],
            "error": "" if meta["error_count"] == 0 else f"{meta['error_count']} errors",
            "fields": [],
        }
    ]


def _stock_location(symbol: str, category: str, dataset: str) -> dict[str, Any]:
    entity_id = _safe_part(symbol or "unknown")
    parts = ["stock", entity_id, category, *[_safe_part(item) for item in dataset.split("/") if item]]
    return {
        "scope": "stock",
        "entity_type": "stock",
        "entity_id": entity_id,
        "category": category,
        "dataset": dataset,
        "path_parts": parts,
    }


def _shared_location(
    scope: str,
    entity_type: str,
    entity_id: str,
    category: str,
    dataset: str,
) -> dict[str, Any]:
    safe_entity = _safe_part(entity_id)
    safe_dataset = _safe_part(dataset)
    parts = [scope, category, safe_dataset]
    if scope == "sector":
        parts = [scope, _safe_part(entity_type), safe_entity]
    return {
        "scope": scope,
        "entity_type": entity_type,
        "entity_id": safe_entity,
        "category": category,
        "dataset": dataset,
        "path_parts": parts,
    }


def _plain_symbol(symbol: str) -> str:
    raw = str(symbol).upper()
    match = re.search(r"(\d{6})", raw)
    return match.group(1) if match else raw


def _safe_part(value: Any) -> str:
    text = str(value or "default").strip()
    text = re.sub(r"[\\/:\s]+", "_", text)
    text = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", text)
    return text[:80] or "default"


def _stable_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _iso(_utc_now())


def _iso(value: datetime) -> str:
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def _iso_from_timestamp(timestamp: float) -> str:
    return _iso(datetime.fromtimestamp(timestamp, tz=timezone.utc))
