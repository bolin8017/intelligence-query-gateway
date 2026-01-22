# Intelligence Query Gateway - Original Design Document (ARCHIVED)

> **Status**: ARCHIVED - Historical reference only
> **Date**: 2026-01-21
> **Superseded by**: [docs/architecture.md](../architecture.md)
> **Language**: Traditional Chinese (original design language)

This is the original system design document preserved for historical reference. For current architectural information, see [architecture.md](../architecture.md).

---

# Intelligence Query Gateway - 系統設計文件

> 設計日期：2026-01-21
> 設計原則：參考 Google 開源風格、Python 社群共識、業界成熟穩定系統設計模式

---

## 1. 系統總覽

### 1.1 核心功能

接收使用者查詢，透過語義分類器判斷查詢複雜度，路由至：
- **Fast Path (Label 0)**：簡單任務（分類、摘要）
- **Slow Path (Label 1)**：複雜任務（創意寫作、一般問答）

### 1.2 技術決策總覽

| 面向 | 決策 |
|------|------|
| 語言/框架 | Python + FastAPI + PyTorch/transformers |
| 分類模型 | Fine-tune DistilBERT |
| 批次處理 | 自建 Batching Layer（asyncio.Queue + 時間窗口） |
| 架構風格 | 分層式架構（Layered Architecture） |
| 依賴注入 | 建構函數注入（Constructor Injection） |
| 配置管理 | Pydantic Settings |
| 快取策略 | 雙層快取（L1 LRU + L2 Redis 可選） |
| 錯誤處理 | Google 風格結構化錯誤 |
| 可觀測性 | structlog + Prometheus metrics |
| 測試策略 | 完整測試金字塔 |
| API 風格 | Google API 風格回應格式 |

### 1.3 架構圖

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Server                         │
│                     (uvicorn + async)                       │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                     API Layer (api/)                        │
│              POST /v1/query-classify endpoint               │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                  Service Layer (services/)                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ CacheService│→ │BatchService │→ │ ClassifierService   │  │
│  │ (L1+L2)     │  │ (Queue+Timer)│  │ (Model Inference)  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                   Model Layer (models/)                     │
│              SemanticRouter (DistilBERT)                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 專案結構

```
intelligence-query-gateway/
├── src/
│   ├── __init__.py
│   ├── main.py                    # FastAPI 應用入口 + lifespan
│   │
│   ├── api/                       # API 層
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── classify.py        # POST /v1/query-classify
│   │   │   └── health.py          # GET /health/live, /health/ready
│   │   ├── dependencies.py        # FastAPI Depends 注入
│   │   └── schemas.py             # Pydantic Request/Response 模型
│   │
│   ├── services/                  # 業務邏輯層
│   │   ├── __init__.py
│   │   ├── classifier.py          # ClassifierService
│   │   ├── batcher.py             # BatchingService
│   │   └── cache.py               # CacheService (L1 + L2)
│   │
│   ├── models/                    # ML 模型層
│   │   ├── __init__.py
│   │   └── semantic_router.py     # SemanticRouter 封裝
│   │
│   ├── core/                      # 核心元件
│   │   ├── __init__.py
│   │   ├── config.py              # Pydantic Settings
│   │   ├── exceptions.py          # 自定義例外類別
│   │   ├── logging.py             # structlog 配置
│   │   └── metrics.py             # Prometheus 指標
│   │
│   └── utils/                     # 工具函數
│       ├── __init__.py
│       └── hashing.py             # 快取 key 生成
│
├── tests/                         # 測試
│   ├── unit/                      # 單元測試
│   ├── integration/               # 整合測試
│   ├── load/                      # 負載測試 (locust)
│   └── fixtures/                  # 測試資料
│
├── scripts/                       # 腳本
│   └── train_router.py            # 訓練語義路由模型
│
├── Dockerfile
├── docker compose.yml             # 含 Redis (可選)
├── pyproject.toml                 # Poetry 依賴管理 (PEP 518)
├── .env.example
└── README.md
```

### 命名規範（Google Python Style Guide）

- 模組名：`snake_case`（如 `semantic_router.py`）
- 類別名：`PascalCase`（如 `ClassifierService`）
- 函數/變數：`snake_case`（如 `classify_query`）
- 常數：`UPPER_SNAKE_CASE`（如 `MAX_BATCH_SIZE`）

---

## 3. 核心元件設計

### 3.1 BatchingService（最關鍵的元件）

**設計原理：**
```
請求進入 → 放入 Queue → 等待觸發條件 → 批次推論 → 分發結果
```

**觸發條件（兩者取其先）：**
- 達到 `max_batch_size`（如 32）
- 達到 `max_wait_time`（≤10ms）

**核心機制：**
- 使用 `asyncio.Queue` 收集請求
- 使用 `asyncio.Future` 讓每個請求能等待自己的結果
- 背景迴圈獨立執行，與 HTTP 請求處理分離
- 非阻塞設計：不影響其他請求的處理

### 3.2 CacheService（雙層快取）

**運作流程：**
1. 請求進入 → 先查 L1 本地 LRU Cache
2. L1 miss → 查 L2 Redis（若啟用）
3. L2 miss → 進入 BatchingService 做推論
4. 推論完成 → 結果寫回 L1 和 L2

**快取 Key 生成：**
- 對查詢文字做正規化（去除多餘空白、統一小寫）
- 用 SHA256 hash 產生固定長度的 key
- 避免特殊字元造成 Redis key 問題

**快取失效策略：**
- L1：LRU 淘汰 + TTL（如 5 分鐘）
- L2：Redis TTL（如 1 小時）
- 不做主動失效，因為相同查詢的分類結果不會改變

### 3.3 ClassifierService

**職責：**
- 封裝 SemanticRouter 模型的推論邏輯
- 處理 tokenization、模型推論、後處理
- 支援批次輸入，回傳批次結果

**推論流程：**
1. 接收查詢列表
2. Tokenizer 批次編碼
3. 模型前向傳播（單次 GPU/CPU 運算）
4. Softmax 取得機率分佈
5. 回傳標籤 + 信心分數

### 3.4 SemanticRouter（模型層）

**模型選擇：DistilBERT**
- 比 BERT 小 40%，速度快 60%
- 保留 97% 的效能
- 非常適合低延遲場景

**訓練策略：**
- 使用 Databricks Dolly 15k 資料集
- 4 個類別映射到 2 個標籤：
  - Fast Path (0)：`classification`, `summarization`
  - Slow Path (1)：`creative_writing`, `open_qa`
- Fine-tune 最後幾層 + 分類頭

---

## 4. API 規格（參考 Google Cloud APIs 風格）

### 4.1 端點設計

| 方法 | 路徑 | 說明 |
|------|------|------|
| POST | `/v1/query-classify` | 分類查詢 |
| GET | `/health/live` | 存活檢查（Kubernetes liveness） |
| GET | `/health/ready` | 就緒檢查（Kubernetes readiness） |
| GET | `/metrics` | Prometheus 指標端點 |

### 4.2 請求/回應格式

**POST /v1/query-classify**

**Request：**
- `query`（必填）：查詢文字，長度 1-2048 字元
- `request_id`（選填）：客戶端提供的追蹤 ID，若未提供則由伺服器生成

**Response（成功）：**
```json
{
  "data": {
    "label": 0,
    "confidence": 0.92,
    "category": "classification"
  },
  "metadata": {
    "request_id": "abc-123",
    "latency_ms": 12.5,
    "cache_hit": false,
    "batch_size": 4
  }
}
```

### 4.3 健康檢查設計（Kubernetes 最佳實踐）

**Liveness（存活）：**
- 只檢查進程是否正常運行
- 回傳快，不做外部依賴檢查
- 失敗 → Kubernetes 重啟容器

**Readiness（就緒）：**
- 檢查模型是否載入完成
- 檢查必要依賴是否可用（如 Redis 連線，若啟用）
- 失敗 → Kubernetes 不將流量導入此實例

### 4.4 冪等性設計（參考 Stripe）

- 相同 `request_id` 的請求，若在快取 TTL 內重複發送，直接回傳快取結果
- 這能防止網路重試造成重複計算
- 對於分類服務，這也天然符合「相同輸入、相同輸出」的特性

---

## 5. 錯誤處理（參考 Google Cloud API 錯誤模型）

### 5.1 錯誤回應格式

```json
{
  "error": {
    "code": 400,
    "message": "Query cannot be empty",
    "status": "INVALID_ARGUMENT",
    "details": []
  }
}
```

### 5.2 例外類別層級

```
ServiceError (基礎類別)
├── ValidationError      # 請求驗證失敗（400）
├── ResourceError        # 資源相關錯誤
│   ├── ModelNotReady    # 模型未載入完成（503）
│   └── CacheError       # 快取操作失敗（內部處理，不暴露）
├── RateLimitError       # 超過速率限制（429）
└── InternalError        # 內部錯誤（500）
```

### 5.3 錯誤處理策略

| 錯誤類型 | 處理方式 |
|---------|---------|
| 請求驗證失敗 | 回傳 400，明確告知錯誤原因 |
| 模型未就緒 | 回傳 503，提示稍後重試 |
| 速率限制 | 回傳 429，附帶 `Retry-After` header |
| 快取失敗 | **不暴露**，降級為無快取模式，記錄日誌 |
| 推論失敗 | 回傳 500，不暴露內部細節 |

### 5.4 降級策略（Graceful Degradation）

- Redis 連線失敗 → 降級為純 L1 快取
- 批次處理逾時 → 單筆處理該請求
- 這是 Google SRE 強調的「優雅降級」原則

### 5.5 日誌記錄原則

**結構化日誌（JSON 格式），包含：**
- `timestamp`：ISO 8601 格式
- `level`：INFO / WARNING / ERROR
- `request_id`：請求追蹤 ID
- `message`：錯誤描述
- `error_type`：例外類別名稱
- `stack_trace`（僅 ERROR 級別）：堆疊追蹤

**敏感資訊處理：**
- 不記錄完整查詢內容（可能包含敏感資訊）
- 只記錄查詢的 hash 值或長度

---

## 6. 測試策略（參考 Google Testing Blog 最佳實踐）

### 6.1 測試金字塔配比

**Google 推薦的比例：70% 單元 / 20% 整合 / 10% 端對端**

| 層級 | 佔比 | 執行速度 | 涵蓋範圍 |
|------|------|---------|---------|
| 單元測試 | 70% | 毫秒級 | 單一函數/類別 |
| 整合測試 | 20% | 秒級 | 元件間協作 |
| 負載測試 | 10% | 分鐘級 | 系統整體效能 |

### 6.2 單元測試策略

| 元件 | 測試重點 |
|------|---------|
| BatchingService | 批次收集邏輯、超時觸發、滿批觸發、空批處理 |
| CacheService | L1 命中/未命中、L2 降級、TTL 過期、key 生成一致性 |
| ClassifierService | 輸入正規化、批次推論結果對應、信心分數範圍 |
| 請求驗證 | 邊界條件（空字串、超長輸入、特殊字元） |

**Mock 策略（Google 的 Test Doubles 原則）：**
- 模型推論 → Mock，避免載入真實模型
- Redis → 使用 `fakeredis` 套件
- 時間相關 → Mock `asyncio.get_event_loop().time()`

### 6.3 整合測試策略

| 場景 | 驗證項目 |
|------|---------|
| 完整請求流程 | API → Cache → Batcher → Classifier → Response |
| 快取命中 | 相同請求第二次應命中快取，延遲顯著降低 |
| 批次處理 | 同時發送多個請求，應被批次處理 |
| 優雅關閉 | 關閉時 Queue 中的請求應被處理完成 |
| 健康檢查 | 模型載入前 readiness 應回傳 503 |

**測試工具：**
- FastAPI 的 `TestClient`（同步測試）
- `httpx.AsyncClient`（非同步測試）
- `pytest-asyncio` 處理 async 測試

### 6.4 負載測試策略（參考 Google SRE 的 SLO 概念）

**測試工具：Locust**（Python 生態系標準）

| 場景 | 配置 | 目標 |
|------|------|------|
| 基準測試 | 100 併發，持續 60 秒 | 建立效能基線 |
| 壓力測試 | 逐步增加到 500 併發 | 找到系統瓶頸 |
| 峰值測試 | 突發 1000 請求 | 驗證批次處理效果 |

**需收集的指標（符合規格要求）：**
- 延遲：P50 / P95 / P99
- 吞吐量：RPS（Requests Per Second）
- 錯誤率：5xx 比例
- 快取命中率

### 6.5 測試資料管理

**Fixture 設計原則（Google 風格）：**
- 使用 `pytest.fixture` 管理測試資料
- 測試資料與測試程式碼分離，放在 `tests/fixtures/` 目錄
- 每個測試應該獨立，不依賴其他測試的執行順序

**模型測試：**
- 單元測試用 Mock 模型
- 整合測試可用小型測試模型（如只訓練 100 個樣本的快速版本）

---

## 7. 部署配置（參考 Kubernetes 與 Docker 最佳實踐）

### 7.1 Docker 設計（Multi-stage Build）

| 階段 | 用途 | 說明 |
|------|------|------|
| builder | 建置環境 | 安裝依賴、下載模型 |
| runtime | 執行環境 | 只包含必要檔案，映像更小 |

**映像優化原則：**
- 使用 `python:3.11-slim` 作為基礎映像（非 alpine，因 PyTorch 相容性）
- 依賴分層：先複製 `pyproject.toml`，再複製程式碼（利用 Docker 快取）
- 非 root 使用者執行（安全性）
- 設定 `PYTHONUNBUFFERED=1` 確保日誌即時輸出

**模型處理策略：**
- 訓練好的模型放在獨立目錄 `/app/models/`
- 可透過 Volume 掛載外部模型，方便更新

### 7.2 配置管理（12-Factor App 原則）

| 變數 | 說明 | 預設值 |
|------|------|--------|
| `APP_ENV` | 環境（dev/staging/prod） | dev |
| `APP_HOST` | 監聽地址 | 0.0.0.0 |
| `APP_PORT` | 監聽埠 | 8000 |
| `MODEL_PATH` | 模型路徑 | /app/models/router |
| `BATCH_MAX_SIZE` | 批次上限 | 32 |
| `BATCH_MAX_WAIT_MS` | 批次等待時間 | 10 |
| `CACHE_L1_SIZE` | L1 快取大小 | 10000 |
| `CACHE_L1_TTL_SEC` | L1 TTL | 300 |
| `REDIS_URL` | Redis 連線字串（可選） | None |
| `LOG_LEVEL` | 日誌級別 | INFO |

**配置驗證：**
- 啟動時用 Pydantic Settings 驗證所有配置
- 缺少必要配置時 fail fast，不進入半工作狀態

### 7.3 容器生命週期管理

**啟動順序（Lifespan 管理）：**
1. 載入配置
2. 初始化日誌系統
3. 載入 ML 模型到記憶體
4. 初始化快取連線（Redis 可選）
5. 啟動 BatchingService 背景迴圈
6. 標記 readiness = True
7. 開始接收請求

**關閉順序（Graceful Shutdown）：**
1. 收到 SIGTERM
2. 標記 readiness = False（停止接收新請求）
3. 等待 BatchingService 處理完 Queue 中的請求
4. 關閉 Redis 連線
5. 釋放模型資源
6. 結束進程

**超時保護：**
- 關閉等待上限（如 30 秒），超時強制結束
- 避免無限等待造成部署卡住

### 7.4 docker compose 設計（本地開發與測試）

| 服務 | 說明 |
|------|------|
| gateway | 主應用服務 |
| redis | 快取服務（可選） |
| prometheus | 指標收集（開發用） |
| grafana | 指標視覺化（開發用） |

**開發模式：**
- 程式碼目錄掛載為 Volume，支援熱重載
- `uvicorn --reload` 自動偵測檔案變更

### 7.5 Kubernetes 就緒（選配，但設計時考慮）

雖然作業只要求 Docker，但設計時已考慮 K8s 部署：
- 健康檢查端點符合 K8s probe 規範
- 無狀態設計，可水平擴展
- 日誌輸出到 stdout/stderr（K8s 日誌收集標準）
- 配置透過環境變數注入（ConfigMap/Secret 友好）

---

## 8. 額外考量

### 8.1 優雅關閉（Graceful Shutdown）

- 處理 SIGTERM 信號
- 停止接收新請求
- 處理完 Queue 中的剩餘請求
- 正確釋放模型資源

### 8.2 模型載入策略

- 應用啟動時載入模型（非首次請求時）
- 使用 FastAPI 的 `lifespan` 管理生命週期
- 避免冷啟動影響首批請求

### 8.3 速率限制（Rate Limiting）

- 輸入驗證（查詢長度上限、非空檢查）
- 可選的速率限制機制（防止濫用）

---

## 9. 實作優先順序建議

1. **Phase 1 - 核心功能**
   - 專案骨架與配置
   - SemanticRouter 模型訓練
   - ClassifierService 基本推論
   - API 端點

2. **Phase 2 - 效能優化**
   - BatchingService 實現
   - CacheService（L1）實現

3. **Phase 3 - 生產就緒**
   - 錯誤處理完善
   - 健康檢查端點
   - Prometheus 指標
   - Docker 配置

4. **Phase 4 - 測試與文件**
   - 單元測試
   - 整合測試
   - 負載測試
   - README 文件

5. **Phase 5 - 進階功能（Bonus）**
   - L2 Redis 快取
   - 優雅關閉
   - 速率限制
