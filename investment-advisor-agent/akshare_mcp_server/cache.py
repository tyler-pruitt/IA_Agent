"""
数据缓存层 — 支持 SQLite (默认) 和 Redis (可选) 双后端

TTL 策略:
  - 行情数据: 5 分钟
  - 财务数据: 1 天
  - 舆情数据: 30 分钟
  - 宏观数据: 1 天
  - 行业数据: 30 分钟
"""

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# 默认缓存目录
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "akshare_mcp"

# 各数据类型默认 TTL (秒)
TTL_CONFIG = {
    "quote": 5 * 60,        # 行情: 5分钟
    "fundamental": 86400,   # 财务: 1天
    "sentiment": 30 * 60,   # 舆情: 30分钟
    "macro": 86400,         # 宏观: 1天
    "sector": 30 * 60,      # 行业: 30分钟
    "news": 15 * 60,        # 新闻: 15分钟
}


def _make_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """根据函数名+参数生成缓存 key"""
    raw = f"{func_name}:{args}:{sorted(kwargs.items())}"
    return hashlib.md5(raw.encode()).hexdigest()


class SQLiteCache:
    """基于 SQLite 的本地缓存，零依赖，适合单机部署"""

    def __init__(self, cache_dir: Optional[str] = None):
        cache_path = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        cache_path.mkdir(parents=True, exist_ok=True)
        self._db_path = cache_path / "cache.db"
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    expires_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires ON cache(expires_at)
            """)

    def get(self, key: str) -> Optional[str]:
        """获取缓存数据，过期返回 None"""
        now = time.time()
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT data FROM cache WHERE key = ? AND expires_at > ?",
                (key, now),
            ).fetchone()
            return row[0] if row else None

    def set(self, key: str, value: str, ttl: int):
        """写入缓存"""
        expires_at = time.time() + ttl
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, data, expires_at) VALUES (?, ?, ?)",
                (key, value, expires_at),
            )

    def delete(self, key: str):
        """删除缓存"""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))

    def clear_expired(self):
        """清理过期缓存"""
        now = time.time()
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("DELETE FROM cache WHERE expires_at <= ?", (now,))

    def clear_all(self):
        """清空所有缓存"""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("DELETE FROM cache")


class RedisCache:
    """基于 Redis 的缓存，适合多实例部署 (可选依赖)"""

    def __init__(self, url: str = "redis://localhost:6379/0", prefix: str = "akmcp:"):
        try:
            import redis
        except ImportError:
            raise ImportError(
                "Redis 缓存需要安装 redis: pip install redis"
            )
        self._client = redis.from_url(url)
        self._prefix = prefix

    def get(self, key: str) -> Optional[str]:
        data = self._client.get(f"{self._prefix}{key}")
        return data.decode() if data else None

    def set(self, key: str, value: str, ttl: int):
        self._client.setex(f"{self._prefix}{key}", ttl, value)

    def delete(self, key: str):
        self._client.delete(f"{self._prefix}{key}")

    def clear_all(self):
        for key in self._client.scan_iter(f"{self._prefix}*"):
            self._client.delete(key)


class DataCache:
    """
    统一缓存接口 — 自动序列化 DataFrame/JSON，按数据类型应用 TTL

    用法:
        cache = DataCache()
        cache.set("quote", "stock_zh_a_hist", ("000001",), {}, df)
        df = cache.get("quote", "stock_zh_a_hist", ("000001",), {})
    """

    def __init__(self, backend: str = "sqlite", cache_dir: Optional[str] = None, redis_url: Optional[str] = None):
        if backend == "redis":
            self._backend = RedisCache(url=redis_url or "redis://localhost:6379/0")
        else:
            self._backend = SQLiteCache(cache_dir=cache_dir)
        self._hit_count = 0
        self._miss_count = 0

    def get(
        self, data_type: str, func_name: str, args: tuple = (), kwargs: Optional[dict] = None
    ) -> Optional[pd.DataFrame]:
        """获取缓存，命中返回 DataFrame，未命中返回 None"""
        kwargs = kwargs or {}
        key = _make_cache_key(func_name, args, kwargs)
        raw = self._backend.get(key)
        if raw is None:
            self._miss_count += 1
            return None
        self._hit_count += 1
        try:
            parsed = json.loads(raw)
            return pd.DataFrame(parsed["data"], columns=parsed.get("columns"))
        except (json.JSONDecodeError, KeyError):
            return None

    def set(
        self,
        data_type: str,
        func_name: str,
        args: tuple = (),
        kwargs: Optional[dict] = None,
        data: pd.DataFrame = None,
    ):
        """写入缓存，自动按数据类型选择 TTL"""
        kwargs = kwargs or {}
        ttl = TTL_CONFIG.get(data_type, 300)
        key = _make_cache_key(func_name, args, kwargs)
        serialized = json.dumps({
            "data": data.values.tolist(),
            "columns": data.columns.tolist(),
        }, ensure_ascii=False, default=str)
        self._backend.set(key, serialized, ttl)

    def get_raw(
        self, data_type: str, func_name: str, args: tuple = (), kwargs: Optional[dict] = None
    ) -> Optional[Any]:
        """获取原始 JSON 缓存 (非 DataFrame 场景)"""
        kwargs = kwargs or {}
        key = _make_cache_key(func_name, args, kwargs)
        raw = self._backend.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw

    def set_raw(
        self,
        data_type: str,
        func_name: str,
        args: tuple = (),
        kwargs: Optional[dict] = None,
        data: Any = None,
    ):
        """写入原始 JSON 缓存"""
        kwargs = kwargs or {}
        ttl = TTL_CONFIG.get(data_type, 300)
        key = _make_cache_key(func_name, args, kwargs)
        serialized = json.dumps(data, ensure_ascii=False, default=str)
        self._backend.set(key, serialized, ttl)

    def clear_all(self):
        """清空所有缓存"""
        self._backend.clear_all()

    @property
    def stats(self) -> dict:
        """缓存统计"""
        total = self._hit_count + self._miss_count
        hit_rate = self._hit_count / total if total > 0 else 0
        return {
            "hits": self._hit_count,
            "misses": self._miss_count,
            "hit_rate": f"{hit_rate:.1%}",
        }
