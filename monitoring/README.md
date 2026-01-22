# Intelligence Query Gateway - 監控系統使用指南

本目錄包含 Intelligence Query Gateway 的完整監控配置，包括 Prometheus metrics 收集和 Grafana 視覺化儀表板。

## 📋 目錄

- [快速開始](#快速開始)
- [架構總覽](#架構總覽)
- [Prometheus 設定](#prometheus-設定)
- [Grafana 儀表板](#grafana-儀表板)
- [告警規則](#告警規則)
- [故障排除](#故障排除)

---

## 快速開始

### 啟動完整監控堆疊

```bash
# 從專案根目錄執行
docker compose up -d

# 驗證所有服務都在運行
docker compose ps
```

預期輸出應顯示以下服務都處於 `Up` 狀態:
- `query-gateway` (Port 8000)
- `query-gateway-prometheus` (Port 9090)
- `query-gateway-grafana` (Port 3000)

### 訪問監控介面

| 服務 | URL | 認證 |
|------|-----|------|
| **Gateway API** | http://localhost:8000 | 無 |
| **Prometheus** | http://localhost:9090 | 無 |
| **Grafana** | http://localhost:3000 | admin / admin |
| **Metrics 端點** | http://localhost:8000/metrics | 無 |

### 驗證 Metrics 收集

```bash
# 1. 檢查 Gateway metrics 端點
curl http://localhost:8000/metrics

# 2. 檢查 Prometheus 是否成功抓取 metrics
curl 'http://localhost:9090/api/v1/query?query=up{job="query-gateway"}'

# 3. 測試發送請求以產生 metrics
curl -X POST http://localhost:8000/v1/query-classify \
  -H "Content-Type: application/json" \
  -d '{"text": "What is machine learning?"}'
```

---

## 架構總覽

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Compose Stack                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    scrapes     ┌──────────────┐          │
│  │   Gateway    │ ───────────▶   │  Prometheus  │          │
│  │  (Port 8000) │   /metrics     │  (Port 9090) │          │
│  │              │                 │              │          │
│  │ - API        │                 │ - Metrics DB │          │
│  │ - /metrics   │                 │ - Alerting   │          │
│  └──────────────┘                 └──────┬───────┘          │
│                                           │                   │
│                                           │ datasource        │
│                                           ▼                   │
│                                  ┌──────────────┐            │
│                                  │   Grafana    │            │
│                                  │  (Port 3000) │            │
│                                  │              │            │
│                                  │ - Dashboards │            │
│                                  │ - Visualization │         │
│                                  └──────────────┘            │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 資料流程

1. **Gateway** 暴露 `/metrics` 端點，提供 Prometheus 格式的 metrics
2. **Prometheus** 每 10 秒抓取一次 Gateway 的 metrics
3. **Grafana** 從 Prometheus 查詢資料並視覺化在儀表板上
4. **告警規則** 在 Prometheus 中評估,觸發時可發送通知

---

## Prometheus 設定

### 配置檔案

- **[prometheus.yml](prometheus/prometheus.yml)** - 主要配置
- **[alerts.yml](prometheus/alerts.yml)** - 告警規則定義

### 抓取配置

```yaml
scrape_configs:
  - job_name: 'query-gateway'
    scrape_interval: 10s      # 每 10 秒抓取一次
    scrape_timeout: 5s        # 5 秒逾時
    metrics_path: '/metrics'
    static_configs:
      - targets: ['gateway:8000']
```

### 關鍵 Metrics 說明

| Metric 名稱 | 類型 | 說明 |
|------------|------|------|
| `query_gateway_requests_total` | Counter | 總請求數 (按 status, cache_hit 分組) |
| `query_gateway_request_latency_seconds` | Histogram | 請求延遲分佈 |
| `query_gateway_inference_latency_seconds` | Histogram | 模型推論延遲 |
| `query_gateway_inference_batch_size` | Histogram | 批次大小分佈 |
| `query_gateway_cache_hits_total` | Counter | 快取命中次數 |
| `query_gateway_cache_misses_total` | Counter | 快取未命中次數 |
| `query_gateway_cache_size` | Gauge | 當前快取大小 |
| `query_gateway_batch_queue_size` | Gauge | 批次佇列深度 |
| `query_gateway_active_requests` | Gauge | 當前處理中的請求數 |
| `query_gateway_model_loaded` | Gauge | 模型是否已載入 (1=是, 0=否) |
| `query_gateway_classifications_total` | Counter | 分類次數 (按 label 分組) |
| `query_gateway_confidence_score` | Histogram | 信心分數分佈 |

### PromQL 查詢範例

```promql
# 每秒請求率 (RPS)
rate(query_gateway_requests_total[1m])

# P99 延遲
histogram_quantile(0.99, rate(query_gateway_request_latency_seconds_bucket[5m]))

# 快取命中率
sum(rate(query_gateway_cache_hits_total[5m]))
/
(sum(rate(query_gateway_cache_hits_total[5m])) + sum(rate(query_gateway_cache_misses_total[5m])))

# 錯誤率
sum(rate(query_gateway_requests_total{status="error"}[5m]))
/
sum(rate(query_gateway_requests_total[5m]))

# 平均批次大小
histogram_quantile(0.50, rate(query_gateway_inference_batch_size_bucket[5m]))
```

---

## Grafana 儀表板

### 自動配置

Grafana 啟動時會自動載入以下配置:

1. **Datasource**: Prometheus 連線 (已自動配置)
2. **Dashboards**: 從 [dashboards/](grafana/dashboards/) 目錄載入

### 可用儀表板

#### 1. Query Gateway - Overview

**檔案**: `query-gateway-overview.json`

**包含面板**:
- ✅ 模型狀態 (Model Status)
- 📊 請求率 (RPS)
- ⚠️ 錯誤率 (Error Rate)
- ⏱️ P99 延遲
- 💾 快取命中率
- 🔄 活躍請求數
- 📈 延遲分佈 (P50/P95/P99)
- 📉 快取命中/未命中趨勢
- 📦 批次大小分佈
- 🧠 模型推論延遲
- 🏷️ 分類結果分佈
- 💯 信心分數分佈

**訪問方式**:
1. 開啟 http://localhost:3000
2. 登入 (admin / admin)
3. 點選左側選單 → Dashboards → Query Gateway - Overview

---

## 告警規則

### SLO 相關告警

根據 Phase 4 效能測試結果設定的 SLO:

| 指標 | SLO | 告警閾值 | 持續時間 | 嚴重性 |
|------|-----|----------|----------|--------|
| **P99 延遲** | < 100ms | > 100ms | 5 分鐘 | Warning |
| **錯誤率** | < 0.1% | > 0.1% | 2 分鐘 | Critical |
| **快取命中率** | > 30% | < 30% | 10 分鐘 | Warning |
| **服務可用性** | 100% | Down | 1 分鐘 | Critical |

### 告警組別

**1. SLO 告警** (`query_gateway_slo_alerts`)
- `HighP99Latency` - P99 延遲過高
- `HighErrorRate` - 錯誤率過高
- `LowCacheHitRate` - 快取命中率過低

**2. 可用性告警** (`query_gateway_availability_alerts`)
- `ServiceDown` - 服務不可用
- `ModelNotReady` - 模型未載入
- `HighConcurrentRequests` - 並發請求數過高

**3. 效能告警** (`query_gateway_performance_alerts`)
- `HighBatchQueueDepth` - 批次佇列過深
- `SlowModelInference` - 模型推論變慢
- `IneffecientBatching` - 批次效率低

**4. 快取告警** (`query_gateway_cache_alerts`)
- `CacheNearCapacity` - 快取接近容量上限
- `CacheFull` - 快取已滿

**5. 品質告警** (`query_gateway_quality_alerts`)
- `HighLowConfidenceRate` - 低信心分類比例過高

### 查看告警狀態

```bash
# 檢查當前告警
curl http://localhost:9090/api/v1/alerts | jq '.data.alerts'

# 檢查告警規則配置
curl http://localhost:9090/api/v1/rules | jq '.data.groups[].rules[].name'
```

---

## 故障排除

### 1. Prometheus 無法抓取 metrics

**症狀**: Prometheus UI 中 Targets 顯示 Down

**檢查步驟**:

```bash
# 1. 確認 Gateway 的 metrics 端點可訪問
curl http://localhost:8000/metrics

# 2. 檢查 Prometheus 配置是否正確
docker exec query-gateway-prometheus cat /etc/prometheus/prometheus.yml

# 3. 檢查 Prometheus 是否能連接到 Gateway
docker exec query-gateway-prometheus wget -O- http://gateway:8000/metrics

# 4. 查看 Prometheus 日誌
docker logs query-gateway-prometheus
```

**常見原因**:
- Gateway 服務未啟動
- 網路配置問題 (確保都在 `monitoring` 網路中)
- 埠號配置錯誤

### 2. Grafana 顯示 "No Data"

**症狀**: Dashboard 面板顯示沒有資料

**檢查步驟**:

```bash
# 1. 確認 Grafana 可以連接到 Prometheus
curl -u admin:admin http://localhost:3000/api/datasources

# 2. 直接查詢 Prometheus 確認有資料
curl 'http://localhost:9090/api/v1/query?query=query_gateway_requests_total'

# 3. 檢查 Grafana 日誌
docker logs query-gateway-grafana

# 4. 在 Grafana 中測試 datasource 連線
# 開啟 Configuration → Data Sources → Prometheus → Save & Test
```

**常見原因**:
- Prometheus 還沒有收集到資料 (需要等待一個 scrape interval)
- Datasource 配置錯誤
- 時間範圍選擇問題 (選擇 "Last 15 minutes")

### 3. 告警不觸發

**症狀**: 即使條件滿足,告警也不觸發

**檢查步驟**:

```bash
# 1. 驗證告警規則已載入
curl http://localhost:9090/api/v1/rules

# 2. 手動測試告警條件
curl 'http://localhost:9090/api/v1/query?query=<your_alert_expression>'

# 3. 檢查告警評估日誌
docker logs query-gateway-prometheus | grep -i "alert"

# 4. 重新載入 Prometheus 配置
curl -X POST http://localhost:9090/-/reload
```

**常見原因**:
- `for` 持續時間還未達到
- 告警規則語法錯誤
- 資料不足以評估告警條件

### 4. 重新載入配置

```bash
# Prometheus 配置重新載入 (不需要重啟)
curl -X POST http://localhost:9090/-/reload

# Grafana 重啟 (需要重新載入 provisioning)
docker restart query-gateway-grafana

# 完整重啟監控堆疊
docker compose restart prometheus grafana
```

### 5. 清理並重新開始

```bash
# 停止所有服務
docker compose down

# 刪除所有資料 (包括 Prometheus 歷史資料和 Grafana 設定)
docker compose down -v

# 重新啟動
docker compose up -d
```

---

## 進階配置

### 資料保留設定

Prometheus 預設保留 30 天的資料。修改 [docker compose.yml](../docker compose.yml):

```yaml
prometheus:
  command:
    - '--storage.tsdb.retention.time=30d'  # 修改這個值
```

### AlertManager 整合 (可選)

如需發送告警通知 (Email, Slack, PagerDuty 等),可以配置 AlertManager:

1. 取消註解 [prometheus.yml](prometheus/prometheus.yml) 中的 `alerting` 區塊
2. 建立 `alertmanager.yml` 配置檔案
3. 在 [docker compose.yml](../docker compose.yml) 中新增 AlertManager 服務

詳細設定請參考 [Prometheus AlertManager 文件](https://prometheus.io/docs/alerting/latest/alertmanager/)。

---

## 參考資源

### 官方文件
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [PromQL Basics](https://prometheus.io/docs/prometheus/latest/querying/basics/)

### 專案文件
- [Phase 5 Implementation Prompt](../docs/phase5-prompt.md)
- [Phase 4 Performance Report](../docs/PHASE4_PERFORMANCE_REPORT.md)
- [Design Document](../docs/plans/2026-01-21-semantic-router-gateway-design.md)

### 最佳實踐
- [Google SRE Book - Monitoring](https://sre.google/sre-book/monitoring-distributed-systems/)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
- [Grafana Dashboard Best Practices](https://grafana.com/docs/grafana/latest/dashboards/build-dashboards/best-practices/)
