实验设计

### 1) 数据集

- **A：新闻/监管文本情绪数据 **
  - 来源：`GDELT` / Kaggle 金融新闻 / `raw_partner_headlines.csv` 等
  - 观测单位：**州–年** 或 **州–季度** \((s,t)\)
  - 处理思路：将新闻根据时间戳和实体映射到州 \(s\)，在每个年份/季度 \(t\) 聚合得到：
    - 平均情绪 `sent_mean_{s,t}`
    - 负面情绪占比 `sent_neg_share_{s,t}`
    - “监管/政策”相关新闻占比 `policy_news_share_{s,t}`
    - 新闻数量 `news_count_{s,t}`（attention proxy）

- **B：FDIC 州级银行财务/监管数据**
  - 示例来源：`Summary_data_states.csv`、或 `Financial_3_2_2026.csv` 按州–年汇总
  - 关键字段：`ASSET`, `BANKS`, `DEP`, `NETINC`, `NIM`, `STNAME`, `YEAR` 等
  - 核心衍生指标：
    - 资产回报率
      $$
      ROA_{s,t} = \frac{NETINC_{s,t}}{ASSET_{s,t}}
      $$
    - ROA 变化
      $$
      \Delta ROA_{s,t} = ROA_{s,t} - ROA_{s,t-1}
      $$
    - 银行数量、总资产、总存款的增速，作为传统“基本面/监管压力”特征

- **C（用于空间）：美国州界 shapefile + 银行总部州信息**
  - 使用州级 shapefile（如 `cb_2024_us_state_20m`）与数据集 B 通过 `STNAME`/州缩写连接
  - 如需更细粒度，可利用 FDIC 银行 HQ 数据中的 `state` 先在机构层面汇总到州级，再与 shapefile join
  - 最终得到州级 \((s,t)\) 压力指标，可直接用于 choropleth 空间图

### 2) 变量（把“CTR/CVR”变成州级压力标签）

- **冲击标签（CTR 类：是否进入“坏年份”）**
  - 以州 \(s\) 在年份 \(t+1\) 的 ROA 变化是否“足够差”作为二元标签：

$$
bad\_year_{s,t+1}
=
\mathbf{1}\left[\Delta ROA_{s,t+1} < q_{0.2}(\Delta ROA_s)\right]
$$

  - 其中 \(q_{0.2}(\Delta ROA_s)\) 为该州历史 ROA 变化的第 20 个百分位，用州内分位数保证不同州的标尺可比。

- **强度标签（CVR 类：坏年份的严重程度）**

$$
severity_{s,t+1}
=
 - \min(\Delta ROA_{s,t+1}, 0)
$$

  - 若 \(\Delta ROA_{s,t+1} \ge 0\)，说明并不“变差”，则 `severity=0`；越负代表盈利恶化越严重。

- **文本特征（“LLM sentiment”思想；按州–期聚合）**
  - 对落在 \((s,t)\) 的新闻聚合得到：
    - 情绪均值、负面占比、情绪离散度（disagreement）
    - 文章数（attention proxy）
    - “监管/政策”关键词占比（policy intensity）

> 情绪模型：如果不想依赖 API key，可优先使用开源金融情绪模型（如 FinBERT）或 `GDELT` 自带 `V2Tone` 作为情绪分数；在 `README.md` 中说明具体选择，保证可复现。

### 3) 两阶段模型

- **Stage 1（CTR / Wide&Deep 思路）**：预测州级“坏年份”概率 \(p(bad\_year)\)
  - 观测单位：州–期 \((s,t)\)，特征包括：
    - 文本情绪特征：`sent_mean_{s,t}`, `sent_neg_share_{s,t}`, `policy_news_share_{s,t}`, `news_count_{s,t}` 等
    - 传统财务特征：上一期的 \(ROA_{s,t}\)、\(\Delta ROA_{s,t}\)、`NIM`, 资产/存款增速、银行数量等
  - 目标：\(bad\_year_{s,t+1}\)
  - 模型：
    - Baseline：只用 FDIC 财务特征的 Logit / XGBoost
    - 财务 + 文本情绪 → Wide&Deep（wide 记忆规则 + deep 泛化）

- **Stage 2（CVR / Deep-only 思路）**：在 `bad_year=1` 的州–期上预测恶化强度 `severity`
  - 输出：

$$
\mathbb{E}\bigl(severity_{s,t+1}\mid bad\_year_{s,t+1}=1, x_{s,t}\bigr)
$$

- **组合得到州级压力评分（决策输入）**

$$
\text{StressScore}_{s,t}
=
p(bad\_year_{s,t+1}=1\mid x_{s,t})
\times
\mathbb{E}\bigl(severity_{s,t+1}\mid bad\_year_{s,t+1}=1, x_{s,t}\bigr)
\times
w_s
$$

其中 \(w_s\) 可选（例如该州银行体系资产规模、在全国中的系统重要性、监管关注权重等）。

### 4) 决策层（把“出价 + 预算”迁移成“监管资源分配”）

- **预算约束**：监管者在每个报告期 \(t\) 只能对 \(K\) 个州（或州内若干重点银行）投入“额外监管注意力”（现场检查、压力测试、深度监控）。

- **策略对比（适合写进 final project）**
  - Rule-based：按传统 FDIC 财务指标（例如上一期 \(\Delta ROA\) 最差、亏损州）选 top-\(K\)
  - Supervised score：按 `StressScore_{s,t}` 选 top-\(K\)（对应 AuroraBid 的 eCPM 类）
  - LinUCB：把州–期的文本+财务特征作为上下文，用 contextual bandit 在线学习“选择哪些州进行重点监测”，奖励可以设置为：
    - 若后一期 \(bad\_year_{s,t+1}=1\) 或 `severity_{s,t+1}` 较大 → 奖励高
    - 否则奖励低

- **离线评估指标建议**
  - Recall@\(K\)：top-\(K\) 州覆盖了多少真实坏年份（`bad_year=1`）
  - Captured severity@\(K\)：top-\(K\) 州覆盖的 `severity` 总量（更贴近“稳定监测收益”）

### 5) 图和 Streamlit

- **静态图 1（Altair）**：横轴为某州或若干州的情绪指标（如 `sent_mean_{s,t}`），纵轴为下一期 \(\Delta ROA_{s,t+1}\) 或坏年份概率，画散点 + 回归线，展示“情绪 vs 基本面恶化”的关系
- **静态图 2（空间，geopandas）**：按州聚合的平均 `StressScore_{s,t}` 或坏年份发生率 `Pr(bad_year=1)`（满足 spatial 要求）
- **Streamlit 动态 App（至少一个动态组件/图）**：
  - 选择年份/季度、情绪平滑窗口、可选情绪指标
  - 动态展示：
    - 州级 choropleth 地图（`StressScore_{s,t}` 或坏年份概率）
    - 情绪 vs ROA 变化的可交互散点/时间序列（例如点击某州后展示该州的时间路径）

---

### 6) 研究问题

#### Q1：LLM 情绪能否预测银行体系基本面恶化？

> 1. Can sentiment scores derived from LLMs applied to financial news predict deteriorations in state-level banking fundamentals (e.g., ROA drops) captured by FDIC data?

- **标签设计**
  - \(bad\_year_{s,t+1}\)：州 \(s\) 是否进入“坏年份”（见上文基于 \(\Delta ROA\) 的定义）
  - \(severity_{s,t+1}\)：坏年份的严重程度，使用 \(-\min(\Delta ROA_{s,t+1},0)\)

- **特征集合**
  - 传统 FDIC 财务变量：上一期 ROA、ROA 变化、NIM、资产/存款增速、银行数量等
  - 文本变量：LLM 情绪特征（均值、负面占比、分歧度、policy 关键词强度等），按州–期聚合

- **模型对比**
  - Baseline：只用 FDIC 财务变量（Logit / XGBoost）预测 `bad_year` / `severity`
  - AuroraBid 版：财务变量 + 文本情绪 → Wide&Deep（`bad_year`）+ Deep-only（`severity`）

- **检验方式**
  - 对比 AUC / PR-AUC / Brier score / \(R^2\) 等指标，看加情绪后预测“坏年份”是否显著提升
  - 做增量解释力对比：
    - 模型 1：No-sentiment（仅 FDIC 特征）
    - 模型 2：+sentiment（FDIC + 情绪）
  - 报告性能提升幅度 + 若干特征重要性结果（如 permutation importance）

直接回答：**LLM 情绪是否在 FDIC 度量的银行体系压力上具有预测能力、提升了多少（覆盖 Q1）**。

#### Q2：LLM 指标是否比传统 FDIC 指标更早/更强？

> 2. Do LLM-based sentiment indicators provide earlier or stronger signals than traditional FDIC fundamentals (e.g., ROA, NIM, growth rates) about future stress?

- **“更强”的部分**
  - 已在 Q1 的模型对比中体现：检查加情绪后，预测 `bad_year` / `severity` 的性能是否超过只用 ROA / NIM / 资产增速等传统指标的 baseline。
  - 可以显式写成嵌套模型（nested models）：
    - Base 模型：

      $$
      \Delta ROA_{s,t+1} \leftarrow \text{past FDIC fundamentals}
      $$

    - Base + Sent 模型：

      $$
      \Delta ROA_{s,t+1} \leftarrow \text{past FDIC fundamentals} + \text{sentiment}_{s,t}
      $$

- **“更早”的部分（lead-lag 分析）**
  - **Lead 设定**：用 \(t\) 期的情绪预测 \(t+1\)、\(t+2\) 期的 `bad_year` / `severity`，比较不同 horizon 下：
    - 情绪特征的系数/重要性是否仍然显著
    - 纯 FDIC 财务变量的信息含量是否衰减更快
  - **事件窗口图（如 2008、2023 等压力期）**：
    - 选若干典型监管/负面新闻事件或已知压力年份，画：
      - 事件附近几期的州级情绪轨迹
      - 与 ROA/NIM 变化的时序对比，看情绪是否在基本面恶化“抬头”前就先动
  - 在 Streamlit 时间序列/散点图里，提供直观的 “情绪领先 vs FDIC 指标” 可视化，让“更早的信号”不只停留在回归系数上。

**不仅情绪增强了预测能力，而且在更早的 horizon 上仍然有信息**。

#### Q3：银行体系压力是否具有地理集聚？

> 3. Are stress reactions measured by FDIC fundamentals and sentiment-driven StressScore geographically concentrated, affecting certain regions more than others?

- **已有设计**
  - 数据集 C：州 shapefile +（可选）银行 HQ 信息
  - 标签/得分：州级坏年份频率、平均 `severity_{s,t}`、\(\text{StressScore}_{s,t}\)

- **构造州级指标**

  例如定义州 \(s\) 的坏年份发生率：

  $$
  \text{BadYearRate}_s
  =
  \frac{\#\{(s,t): bad\_year_{s,t}=1\}}
       {\#\{(s,t): \text{有 FDIC 数据}\}}
  $$

  或者在特定危机窗口（如 2008–2010 或 2023 regional banking turmoil）内求各州平均 \(\text{StressScore}_{s,t}\)。

- **空间可视化与统计检验**
  - 使用 `geopandas` + choropleth map（这就是静态图 2），展示州级 `BadYearRate_s` 或平均 `StressScore_{s,\cdot}`
  - 简单空间/面板分析（可选加分）：
    - 比较不同州间的均值差异并给出置信区间 / t-test
    - 如有精力，可计算 Moran’s I / LISA，检验是否存在空间自相关

**某些地区在情绪冲击/基本面恶化上的暴露度更高**。

#### 风格决策层在论文问题中的作用

- 论文的主问题是“能否量化和提前识别银行体系压力”，而 LinUCB/Bandit + 预算约束这一层，把前面的指标**操作化**为“有限监管资源下的空间分配策略”：
  - 假设监管者在每个报告期只能重点盯 \(K\) 个州（或州内重点银行组合）
  - 策略对比：
    - 传统规则：按上一期 ROA 下降幅度、亏损程度排序选 top-\(K\)
    - LLM-sentiment 驱动的 `StressScore_{s,t}`：按 `StressScore_{s,t}` 排序选 top-\(K\)
    - LinUCB：用 contextual bandit 在线学习选择 top-\(K\) 州
  - 比较在离线评估中，哪种策略捕获更多真实 `bad_year` / `severity`。

让结论从“LLM 有信息”升级为：  
**在有限注意力条件下，用 LLM-sentiment 驱动的策略更能识别高风险地区/州**，贴合 “regulatory monitoring” 的 policy 解释。

#### 建议

- 论文结构可以按下面顺序组织：
  1. 描述数据与情绪构造（对应 proposal 的 Datasets + Sentiment）
  2. Q1 + Q2：预测实验（`bad_year` / `severity` + baseline vs +sentiment，含 lead-lag）
  3. Q3：地理分布与空间图 / 简单检验