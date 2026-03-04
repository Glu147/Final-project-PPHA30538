**Can LLMs Quantify Market Reactions to Financial News and Regulation? Evidence from Regional Bank Volatility**

### **Group Members**

SongyiHuang  JiayuZhao  YichenLiu

### **Motivating Research Question**

Financial regulators such as the SEC, FDIC, and the Federal Reserve closely monitor market stability, yet it remains difficult to measure how quickly and intensely financial markets react to regulatory announcements and news events.

This project asks:

1. **Can sentiment scores derived from Large Language Models (LLMs) applied to financial news predict short-term volatility in regional bank stocks?**
2. **Do LLM-based sentiment indicators provide earlier or stronger signals of market instability than traditional market metrics such as historical volatility or trading volume?**
3. **Are market reactions to financial news and regulation geographically concentrated, affecting certain regions more than others?**

By answering these questions, the project explores how AI-based text analysis could support **financial stability monitoring and regulatory decision-making**, making it directly relevant to public policy.

### **Potential Datasets**

#### **Dataset 1 — Financial Market Data (Disaggregated)**

- **Source:** Yahoo Finance (yfinance API)
- **Content:** Daily (or hourly) OHLCV stock data for U.S. regional banks (e.g., constituents of the KRE ETF or individual banks such as ZION, PACW, WAL).
- **Why it works:** Data is disaggregated by **company and date**, allowing firm-level volatility analysis. From this we will construct rolling volatility, daily returns, and market instability measures.



#### **Dataset 2 — Financial News & Regulatory Text Data**

- **Source:** NewsAPI, GDELT, or SEC EDGAR (8-K filings and regulatory announcements)
- **Content:** News headlines and/or short articles related to banking, financial regulation, and market conditions.
- **Processing:** We will use an LLM or NLP sentiment model to convert raw text into **daily sentiment scores** and **sentiment dispersion measures** (e.g., disagreement across news).
- **Why it works:** This transforms unstructured policy-relevant text into structured numerical indicators.



#### **Dataset 3 — Geographic Data for Banks **

- **Source:** FDIC bank headquarters database or SEC company location data; U.S. Census shapefiles for mapping
- **Content:** Geographic locations (state or county) of bank headquarters
- **Purpose:** Allows us to aggregate stock performance by region and create spatial visualizations of financial stress.



### **Proposed Static Visualizations**

#### 1. Time Series with Sentiment Overlay (Altair)

A dual-axis line chart showing the stock price (or volatility) of a representative regional bank or ETF over time, overlaid with the moving average of LLM-derived sentiment scores.

**Purpose:** To visually assess whether changes in news sentiment coincide with or precede periods of high market instability.



#### **2. Scatter Plot of Sentiment Disagreement vs. Market Volatility (Altair)**

Each point represents a trading day. The x-axis measures the dispersion or disagreement in daily news sentiment, and the y-axis measures market volatility (e.g., rolling standard deviation or True Range).

**Purpose:** To explore whether “confusing” or conflicting news environments are associated with higher financial instability.

#### 3. Spatial Map of Regional Financial Stress (Geopandas)

A U.S. map where states (or counties) are colored based on the average stock performance or volatility of regional banks headquartered there during a selected period (e.g., a banking crisis window).

**Purpose:** To show that financial stress may be geographically concentrated, which is relevant for regionally targeted regulatory monitoring.



#### **4. Heatmap of Sector-Level Sentiment Over Time (Altair)**

A heatmap where rows represent sectors (e.g., banking, technology, energy) and columns represent dates, with color showing the average LLM-derived sentiment score.

**Purpose:** To compare whether banking sentiment behaves differently from other sectors during regulatory or financial stress events.



### **Planned Streamlit Application**

We will build an interactive dashboard titled:

> **“Financial News Sentiment & Market Stress Monitor”**


The app will allow users to explore how financial news sentiment and stock market volatility evolve over time and across regions.


Key interactive features will include:

- Selecting individual banks or ETFs

- Adjusting the time window
- Viewing sentiment and volatility together
- Exploring geographic patterns of financial stress

This application will make our analysis more accessible and demonstrate how AI-driven sentiment measures could be used as a real-time policy monitoring tool.



# **Streamlit **Wireframe

**Financial News Sentiment & Market Stress Monitor**

### section 1 - Controls Panel

User inputs:

Select bank or ETF(KRE ETF, ZION, PACW, WAL,etc)

Date range:date slider

Volatility Metric:

Radio buttons: Rolling Std Dev / True Range / Daily Return

Sentiment Smoothing Window: 3-day / 7-day / 14-day moving average



### **Section 2 — Time Series & Sentiment**

Interactive Line Chart:

Stock price or volatility,LLM sentiment score,Hover tooltips showing exact date, price, sentiment

### **Section 3 — Sentiment vs Volatility Relationship**

**Interactive Scatter Plot**

X-axis: Sentiment dispersion or sentiment level

Y-axis: Volatility

### **Section 4 — Geographic Financial Stress Map** 

Color scale: Average bank stock decline or volatility by state

Hover: State name + average bank performance

Dropdown to switch: Crisis period,Full sample,Custom date range



### **Section 5 — Sector Sentiment Heatmap**