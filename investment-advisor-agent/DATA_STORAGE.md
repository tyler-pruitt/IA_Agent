# 本地数据存储设计

本文档说明 `investment-advisor-agent` 的本地数据存储形式。目标不是替代现有 MCP 工具，而是在 MCP/API 返回数据之上增加一层可审计、可复用、可扩展的数据仓储。

当前 v1 已实现:

- `agents/local_data_store.py`: SQLite 索引 + JSON payload 文件。
- `agents/data_provider.py`: 请求前读本地仓储，未命中再调用 MCP；调用成功后写本地仓储。
- `config/settings.py`: 提供本地仓储开关和目录配置。

## 1. 当前已有缓存

项目当前已有 MCP 服务端函数级缓存:

| 层级 | 位置 | 作用 | 局限 |
|------|------|------|------|
| MCP DataCache | `akshare_mcp_server/cache.py` | 按 AKShare 函数 + 参数做 TTL 缓存，默认 SQLite | key 是 hash，不方便按股票/接口/日期检索 |
| Agent 数据快照 | `raw_data_summary` | UI 展示本次分析拿到的数据摘要 | 只跟随一次分析结果，不是长期数据仓储 |

因此项目新增了一层“规范化本地数据仓储”: 保留 MCP 返回 payload，建立可查询索引，并区分公共数据和个股数据。

## 2. 设计目标

1. 公共数据只拉一次，多只股票复用，例如北向资金、龙虎榜、融资融券、市场情绪、宏观、市场估值。
2. 个股数据按股票代码落盘，例如 K 线、估值、财报、资金流、新闻、舆情。
3. 所有数据保留 MCP 原始返回，避免二次清洗丢字段。
4. 用 SQLite 管理索引，用 JSON 文件保存 payload，方便人工查看和后续迁移。
5. 读缓存前必须检查 TTL 和数据时效，不能因为本地有旧数据就继续参与 Agent 分析。
6. 每次分析结果也可以落盘，便于复盘“当时用了哪些数据得到这个结论”。

## 3. 存储目录

默认目录:

```text
~/.cache/investment_advisor_agent/
  data_store.sqlite
  payloads/
    stock/
      000001/
        fundamental/
          valuation/
            latest.json
            2026-06-07T10-30-00Z.json
          balance/
          profit/
          cashflow/
        technical/
          price_volume/daily_qfq/latest.json
          capital_flow/individual/latest.json
        sentiment/
          sentiment/latest.json
          news/individual/latest.json
    market/
      capital_flow/north/latest.json
      lhb/stock_statistic/latest.json
      margin/latest.json
      emotion/latest.json
      valuation/latest.json
    macro/
      china_overview/latest.json
      global_interest/latest.json
    sector/
      industry/银行/latest.json
      concept/人工智能/latest.json
    analysis/
      stock/000001/2026-06-07T10-35-00Z.json
```

说明:

- `latest.json` 用于快速读取最新缓存。
- 时间戳文件用于审计和复盘，可以按保留策略定期清理。
- 文件名使用 ISO 时间，但把 `:` 替换成 `-`，避免跨平台路径问题。
- 股票代码统一用 6 位代码，例如 `000001`；市场信息放入 payload metadata。

## 4. Payload 文件格式

每个 JSON 文件保留统一外壳:

```json
{
  "schema_version": "mcp_payload_v1",
  "meta": {
    "tool": "get_stock_price_volume",
    "arguments": {
      "symbol": "000001",
      "period": "daily",
      "adjust": "qfq"
    },
    "scope": "stock",
    "entity_type": "stock",
    "entity_id": "000001",
    "category": "technical",
    "dataset": "price_volume",
    "fetched_at": "2026-06-07T10:30:00Z",
    "expires_at": "2026-06-07T10:35:00Z",
    "payload_hash": "sha256:...",
    "row_count": 300,
    "field_count": 12,
    "latest_date": "2026-06-05",
    "freshness_status": "fresh",
    "source": "eastmoney.stock_zh_a_hist"
  },
  "payload": {
    "symbol": "000001",
    "period": "daily",
    "sort_order": "date_desc",
    "data": []
  }
}
```

原则:

- `payload` 必须是 MCP 工具原始返回，不做破坏性改写。
- `meta.latest_date` 来自 `agents.data_freshness.latest_date(payload)`。
- `meta.payload_hash` 用于去重，同一份 payload 不重复写审计文件。
- `freshness_status` 当前 v1 先记录为 `unknown`；读取后 Agent 层仍会重新执行数据时效闸门。

## 5. SQLite 索引表

### 5.1 `datasets`

记录每个可复用数据集的最新版本。

```sql
CREATE TABLE datasets (
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
);

CREATE INDEX idx_datasets_entity ON datasets(scope, entity_type, entity_id);
CREATE INDEX idx_datasets_tool ON datasets(tool);
CREATE INDEX idx_datasets_expires ON datasets(expires_at);
CREATE INDEX idx_datasets_latest_date ON datasets(latest_date);
```

### 5.2 `dataset_sections`

记录 payload 内部子块摘要，方便 UI 展示和数据质量排查。

```sql
CREATE TABLE dataset_sections (
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
);

CREATE INDEX idx_sections_dataset ON dataset_sections(dataset_id);
CREATE INDEX idx_sections_section ON dataset_sections(section);
```

### 5.3 `analysis_runs`

记录一次完整分析使用了哪些数据和最终结果。

```sql
CREATE TABLE analysis_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT UNIQUE NOT NULL,
  symbol TEXT NOT NULL,
  query TEXT,
  profile_json TEXT,
  recommendation_path TEXT NOT NULL,
  total_score REAL,
  advice TEXT,
  active_dimensions_json TEXT,
  skipped_dimensions_json TEXT,
  started_at TEXT NOT NULL,
  finished_at TEXT NOT NULL
);
```

### 5.4 `analysis_run_datasets`

关联一次分析和实际使用的数据集。

```sql
CREATE TABLE analysis_run_datasets (
  run_id TEXT NOT NULL,
  dataset_id INTEGER NOT NULL,
  agent TEXT NOT NULL,
  used_in_scoring INTEGER NOT NULL DEFAULT 1,
  skip_reason TEXT,
  PRIMARY KEY (run_id, dataset_id),
  FOREIGN KEY(dataset_id) REFERENCES datasets(id)
);
```

## 6. 接口到存储分区映射

### 6.1 个股数据

| MCP 工具 | 参数 | scope | 路径分区 | 复用策略 |
|----------|------|-------|----------|----------|
| `get_stock_fundamental` | `symbol`, `report_type=valuation` | stock | `stock/{symbol}/fundamental/valuation` | 同一股票复用 |
| `get_stock_fundamental` | `report_type=indicator` | stock | `stock/{symbol}/fundamental/indicator` | 同一股票复用 |
| `get_stock_fundamental` | `report_type=balance/profit/cashflow` | stock | `stock/{symbol}/fundamental/{report_type}` | 同一股票复用 |
| `get_stock_profit_forecast` | `symbol` | stock | `stock/{symbol}/fundamental/forecast` | 同一股票复用；底层可复用全市场数据 |
| `get_stock_earnings_preview` | `symbol`, `date`, `preview_type` | stock | `stock/{symbol}/fundamental/earnings/{preview_type}` | 同一股票同报告期复用 |
| `get_stock_price_volume` | `symbol`, `period`, `adjust` | stock | `stock/{symbol}/technical/price_volume/{period}_{adjust}` | 同一股票同周期复用 |
| `get_stock_capital_flow` | `symbol`, `scope=individual` | stock | `stock/{symbol}/technical/capital_flow/individual` | 同一股票复用 |
| `get_stock_sentiment` | `symbol` | stock | `stock/{symbol}/sentiment/sentiment` | 同一股票复用 |
| `get_stock_news` | `symbol`, `scope=individual` | stock | `stock/{symbol}/sentiment/news/individual` | 同一股票复用 |

### 6.2 公共市场数据

| MCP 工具 | 参数 | scope | 路径分区 | 复用策略 |
|----------|------|-------|----------|----------|
| `get_stock_capital_flow` | `scope=north` | market | `market/capital_flow/north` | 所有股票复用 |
| `get_stock_lhb` | `detail_type=stock_statistic` | market | `market/lhb/stock_statistic` | 所有股票复用 |
| `get_stock_margin_detail` | 无 | market | `market/margin` | 所有股票复用 |
| `get_stock_market_emotion` | `date` | market | `market/emotion` | 所有股票复用 |
| `get_stock_market_valuation` | `index` | market | `market/valuation/{index}` | 所有股票复用 |

### 6.3 宏观和板块数据

| MCP 工具 | 参数 | scope | 路径分区 | 复用策略 |
|----------|------|-------|----------|----------|
| `get_macro_china_overview` | 无 | macro | `macro/china_overview` | 全局复用 |
| `get_macro_global_interest` | 无 | macro | `macro/global_interest` | 全局复用 |
| `get_stock_sector_analysis` | `sector_type`, `name` | sector | `sector/{sector_type}/{name}` | 同板块复用 |

## 7. TTL 与时效策略

建议沿用当前 MCP `TTL_CONFIG`，但本地数据仓储需要额外记录 `latest_date` 和 `freshness_status`。

| 数据类型 | 建议 TTL | 读取缓存条件 |
|----------|----------|--------------|
| K 线/行情 | 5 分钟 | `expires_at > now` 且最新日期未超过时效阈值 |
| 个股资金流 | 5 分钟 | 同上 |
| 北向/融资融券/龙虎榜 | 5-30 分钟 | 公共数据可被多股复用 |
| 新闻 | 15 分钟 | 新闻时效阈值 14 天 |
| 舆情 | 30 分钟 | 舆情时效阈值 30 天 |
| 财报/估值 | 1 天 | 财报可长期保留，但读取时仍做 freshness gate |
| 宏观 | 1 天 | 宏观低频，但超过 120 天不参与评分 |
| 板块 | 30 分钟 | 热点和成分股变化较快 |

缓存读取必须满足:

```text
exists(payload)
AND expires_at > now
AND 用户没有 force_refresh
```

如果 TTL 命中但 Agent 层 freshness 不通过，可以展示本地旧数据，但不能让对应 Agent 参与评分。

## 8. 读写流程

### 8.1 写入流程

```text
DataProvider._call_tool
  -> MCPToolClient.call_tool
  -> 得到 MCP payload
  -> LocalDataStore.save(tool, args, payload)
     - 计算 cache_key
     - 提取 symbol/scope/category/dataset
     - 提取 row_count/field_count/latest_date/error_count
     - 写 payloads/.../timestamp.json
     - 更新 latest.json
     - upsert datasets
     - upsert dataset_sections
  -> 返回 payload 给 Agent
```

### 8.2 读取流程

```text
DataProvider._call_tool
  -> LocalDataStore.get_fresh(tool, args)
     - 根据 cache_key 查 datasets
     - 检查 expires_at
     - 读取 latest.json
     - 重新执行 data_freshness
  -> 命中则返回本地 payload
  -> 未命中则调用 MCP
```

### 8.3 强制刷新

后续可以在 UI 增加按钮:

```text
刷新实时数据
```

它对应后续可扩展的 `force_refresh=True`，跳过本地读取，但仍写入本地仓储。当前 v1 尚未暴露 UI 强制刷新入口。

## 9. 与现有 Agent 的关系

本地数据仓储只负责“拿数据”和“存数据”，不负责评分。

| 模块 | 职责 |
|------|------|
| `DataProvider` | 统一请求入口，优先读本地存储，必要时调用 MCP |
| `LocalDataStore` | 保存 MCP payload、索引、元数据、读取最新有效缓存 |
| `data_freshness.py` | 判断数据是否可参与 Agent 分析 |
| 子 Agent | 从 DataProvider 获取数据，时效通过才调用 LLM |
| `decision_agent.py` | 只对有效维度加权 |

## 10. 当前配置

`config/settings.py` 中已支持:

```python
LOCAL_DATA_STORE_ENABLED = True
LOCAL_DATA_STORE_READ_THROUGH = True
LOCAL_DATA_STORE_WRITE_THROUGH = True
LOCAL_DATA_STORE_HISTORY_ENABLED = True
LOCAL_DATA_STORE_DIR = "~/.cache/investment_advisor_agent"
```

环境变量示例:

```bash
export LOCAL_DATA_STORE_ENABLED=true
export LOCAL_DATA_STORE_DIR="$HOME/.cache/investment_advisor_agent"
```

## 11. 后续增强

1. UI 数据快照增加“本地缓存命中/刷新时间/来源文件”展示。
2. 增加 `force_refresh=True` 和 UI 强制刷新按钮。
3. 增加清理/查询命令:

```bash
python -m scripts.cache_admin --clear-expired
python -m scripts.cache_admin --show 000001
```

## 12. 示例: `000001` 单股分析会落哪些数据

```text
stock/000001/fundamental/valuation/latest.json
stock/000001/fundamental/indicator/latest.json
stock/000001/fundamental/balance/latest.json
stock/000001/fundamental/profit/latest.json
stock/000001/fundamental/cashflow/latest.json
stock/000001/fundamental/forecast/latest.json
stock/000001/technical/price_volume/daily_qfq/latest.json
stock/000001/technical/capital_flow/individual/latest.json
stock/000001/sentiment/sentiment/latest.json
stock/000001/sentiment/news/individual/latest.json

market/capital_flow/north/latest.json
market/lhb/stock_statistic/latest.json
market/margin/latest.json
market/emotion/latest.json

analysis/stock/000001/{run_id}.json
```

其中 `market/*` 下的数据可以被下一只股票直接复用，不需要重复拉取。
