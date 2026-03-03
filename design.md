实验设计

### 1) 数据集（满足“至少两套数据” + 空间要求）

- **数据集 A：银行/ETF 市场数据（`yfinance`）**
  - 标的：`KRE` + 一组区域性银行（如 `ZION`, `PACW`, `WAL` 等）
  - 频率建议：先用**日频**（更稳、算力友好）

- **数据集 B：新闻/监管文本数据（`GDELT` / `NewsAPI` / `EDGAR`）**
  - 最推荐：`GDELT`（公开、可复现、粒度细）
  - 抓取关键词：`FDIC` / `SEC` / `Fed` / `regulation` / `bank run` / `regional bank` + `ticker/name`
  - 保留字段：时间、来源、标题/摘要/正文片段（文章级）

- **数据集 C（用于空间）：FDIC 银行总部地理信息 + 美国州界 shapefile**
  - 用 bank HQ 的州/经纬度把压力聚合到州层面，做空间可视化

### 2) 关键变量定义（把“CTR/CVR”变成你题目的标签）

- **冲击标签（CTR 类）**
  - 定义（以 “次日异常波动” 为 `1`）：

$$
\text{spike}_{i,t+1} = \mathbf{1}\left[\text{RV}_{i,t+1} > q_{0.9}(\text{RV}_i)\right]
$$

  - 说明：对每家银行使用自身分位数阈值 \(q_{0.9}(\text{RV}_i)\)，避免不同银行“基准波动水平”差异带来的不可比性。

- **强度标签（CVR 类）**
  - 可选定义：

$$
\text{severity}_{i,t+1} = \text{RV}_{i,t+1}
$$

  - 或者使用绝对收益：

$$
\text{severity}_{i,t+1} = |r_{i,t+1}|
$$

- **文本特征（“LLM sentiment”思想；按日聚合到 bank-date 粒度）**
  - 情绪均值、负面占比、情绪离散度（disagreement）
  - 文章数（attention proxy）
  - “监管/政策”关键词占比（policy intensity）

> 情绪模型实现上：如果你不想依赖 API key，主线用开源金融情绪模型（如 FinBERT）或 `GDELT` 自带 `tone` 作为 fallback；写进 `README.md`，保证可复现。

### 3) 两阶段模型（AuroraBid 风格的核心）

- **Stage 1（CTR / Wide&Deep 思路）**：预测 \(p(\text{spike})\)
  - Baseline：Logit / XGBoost
  - AuroraBid 借鉴版：Wide&Deep（wide 记忆规则 + deep 泛化）

- **Stage 2（CVR / Deep-only 思路）**：在 \(\text{spike}=1\) 的样本上预测 `severity`
  - 输出：

$$
\mathbb{E}(\text{severity}\mid \text{spike})
$$

- **组合得到每日每家银行的压力评分（决策输入）**

$$
\text{StressScore}_{i,t}
=
p(\text{spike}_{i,t+1}=1\mid x_{i,t})
\times
\mathbb{E}(\text{severity}_{i,t+1}\mid \text{spike}_{i,t+1}=1, x_{i,t})
\times
w_i
$$

其中 \(w_i\) 可选（例如市值、地区重要性、监管关注权重）。

### 4) 决策层（把“出价 + 预算”迁移成“监管资源分配”）

- **预算约束**：每天只能选 \(K\) 家银行进入“重点监测名单”（或每州最多 \(k_s\) 家）。

- **策略对比（适合写进 final project）**
  - Rule-based：按昨日波动/成交量选 top-\(K\)
  - Supervised score：按 `StressScore` 选 top-\(K\)（对应 AuroraBid 的 eCPM 类）
  - LinUCB：用 contextual bandit 在线选（探索-利用），奖励定义为“次日真实 `severity`”（或 \(\text{spike}=1\) 的奖励 = 1）

- **离线评估指标建议**
  - Recall@\(K\)：top-\(K\) 覆盖了多少真实 `spike`
  - Captured severity@\(K\)：top-\(K\) 覆盖的 `severity` 总量（更贴近“稳定监测收益”）

### 5) 你要交的图和 Streamlit（最稳拿分）

- **静态图 1（Altair）**：某银行/ETF 的波动（RV）时间序列 + 情绪指数叠加（可选滚动均值），并标出 `spike` 日
- **静态图 2（空间，geopandas）**：按州聚合的平均 `StressScore` 或 `spike` 发生频率（满足 spatial 要求）
- **Streamlit 动态 App（至少一个动态组件/图）**：
  - 选银行/ETF、日期范围、情绪平滑窗口
  - 动态展示：时间序列、散点（情绪 vs 次日 RV 或 `severity`）、交互地图（可把空间图也放进来）
