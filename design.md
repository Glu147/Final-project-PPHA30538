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

### 2) 关键变量定义（把“CTR/CVR”变成题目标签）

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

ETF数据: 美股宽基ETF 平均报价和现实差价数据

### 3) 两阶段模型

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

---

### 6) 与三个研究问题的对应关系

#### Q1：LLM 情绪能否预测短期波动？

> 1. Can sentiment scores derived from LLMs applied to financial news predict short-term volatility in regional bank stocks?

- **标签设计**
  - \(\text{spike}_{i,t+1}\)：是否发生异常波动（见上文定义）
  - \(\text{severity}_{i,t+1}\)：波动强度，使用 \(\text{RV}_{i,t+1}\) 或 \(|r_{i,t+1}|\)

- **特征集合**
  - 传统市场变量：滞后收益、成交量、历史 RV 等
  - 文本变量：LLM 情绪特征（均值、负面占比、分歧度、policy 关键词强度等）

- **模型对比**
  - Baseline：只用市场变量（Logit / XGBoost）预测 `spike` / `severity`
  - AuroraBid 版：市场变量 + 文本情绪 → Wide&Deep（`spike`）+ Deep-only（`severity`）

- **检验方式**
  - 对比 AUC / PR-AUC / Brier score / \(R^2\) 等指标，看加情绪后预测短期波动是否显著提升
  - 做增量解释力对比：
    - 模型 1：No-sentiment（仅市场特征）
    - 模型 2：+sentiment（市场 + 情绪）
  - 报告性能提升幅度 + 若干特征重要性结果（如 permutation importance）

这可以直接回答：**LLM 情绪是否有预测能力、提升了多少（覆盖 Q1）**。

#### Q2：LLM 指标是否比传统指标更早/更强？

> 2. Do LLM-based sentiment indicators provide earlier or stronger signals than traditional metrics such as historical volatility or trading volume?

- **“更强”的部分**
  - 已在 Q1 的模型对比中体现：检查加情绪后，预测性能是否超过只用历史波动/成交量的 baseline。
  - 可以显式写成嵌套模型（nested models）：

    - Base 模型：

      $$
      \text{RV}_{i,t+1} \leftarrow \text{past RV/vol}
      $$

    - Base + Sent 模型：

      $$
      \text{RV}_{i,t+1} \leftarrow \text{past RV/vol} + \text{sentiment}
      $$

- **“更早”的部分（lead-lag 分析）**
  - **Lead 设定**：用 \(t\) 期的情绪预测 \(t+1\)、\(t+2\) 期的 `spike` / `severity`，比较不同 horizon 下：
    - 情绪特征的系数/重要性是否仍然显著
    - 纯市场变量的信息含量是否衰减更快
  - **事件窗口图**：
    - 选若干典型监管/负面新闻事件，画：
      - 事件附近几天的情绪走向
      - 与 realized volatility 的时序对比，看情绪是否在波动“抬头”前就先动
  - 在 Streamlit 时间序列图里，提供直观的 “情绪领先 vs 波动” 可视化，让“更早的信号”不只停留在回归系数上。

这样可以说明：**不仅情绪增强了预测能力，而且在更早的 horizon 上仍然有信息（回答 Q2 的 “earlier or stronger”）**。

#### Q3：市场反应是否具有地理集聚？

> 3. Are market reactions to financial news and regulation geographically concentrated, affecting certain regions more than others?

- **已有设计**
  - 数据集 C：FDIC HQ + 州 shapefile
  - 标签/得分：`spike` 频率、平均 `severity`、\(\text{StressScore}_{i,t}\)

- **构造州级指标**

  例如定义州 \(s\) 的异常波动发生率：

  $$
  \text{SpikeRate}_s
  =
  \frac{\#\{\text{spike 银行-日 属于州 } s\}}
       {\#\{\text{所有 银行-日 属于州 } s\}}
  $$

  或者在危机窗口（如 2023 regional banking turmoil）内求各州平均 \(\text{StressScore}\)。

- **空间可视化与统计检验**
  - 使用 `geopandas` + choropleth map（这就是静态图 2），展示州级 `SpikeRate` 或平均 `StressScore`
  - 简单空间/面板分析（可选加分）：
    - 比较不同州间的均值差异并给出置信区间 / t-test
    - 如有精力，可计算 Moran’s I / LISA，检验是否存在空间自相关

由此可以较稳地说：**某些地区在情绪冲击/波动冲击上的暴露度更高，从而回答 Q3**。

#### 风格决策层在论文问题中的作用

- 论文的主问题是“能否量化和提前识别市场反应”，而 LinUCB/Bandit + 预算约束这一层，把前面的指标**操作化**为“有限监管资源下的监测策略”：
  - 假设监管者每天只能重点盯 \(K\) 家银行
  - 策略对比：
    - 传统规则：按历史波动排序选 top-\(K\)
    - LLM-sentiment 驱动的 `StressScore`：按 `StressScore` 排序选 top-\(K\)
    - LinUCB：用 contextual bandit 在线学习选择 top-\(K\)
  - 比较在离线评估中，哪种策略捕获更多真实 `spike` / `severity`。

这不会改变 Q1–Q3 的答案，但可以让结论从“LLM 有信息”升级为：  
**在有限注意力条件下，用 LLM-sentiment 驱动的策略更能识别高风险银行**，非常贴合 “regulatory monitoring” 的 policy 解释。

#### 小建议（让故事更紧、更好写）

- 论文结构可以按下面顺序组织：
  1. 描述数据与情绪构造（对应 proposal 的 Datasets + Sentiment）
  2. Q1 + Q2：预测实验（`spike` / `severity` + baseline vs +sentiment，含 lead-lag）
  3. Q3：地理分布与空间图 / 简单检验
  4. Extension：AuroraBid 风格的监管资源分配策略（LinUCB vs rule-based）
- 课程角度：只要把上述实验做出来，已经明显超出“最低要求”；Bandit/RL 属于“锦上添花”，时间不够可以做减法，但这套设计本身是可执行且足以回答你原文里的三个研究问题的。
