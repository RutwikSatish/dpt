import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ---------------- Page config / branding ----------------
st.set_page_config(
    page_title="PlanPulse — Demand Planning Auto-Analyst",
    page_icon="📦",
    layout="wide",
)

# Minimal CSS to feel like a product
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; }
    .pp-subtle { 
    color: #B74134; 
    font-size: 0.95rem; 
    font-weight: 500;
}

    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📦 PlanPulse")
st.markdown(
    '<div class="pp-subtle">Upload demand planning data → get instant diagnostics, small visuals, Top-10 risk SKUs, and an explainable next-best decision.</div>',
    unsafe_allow_html=True
)

with st.expander("What files does this work with?"):
    st.write(
        """
**Works with most demand planning exports** (CSV/XLSX) as long as you have:
- A date/week/month column (ex: `date`, `week`, `period`, `YYYY-WW`)
- A demand/actual column (ex: `demand`, `sales`, `consumption`, `actual_qty`)

Optional but recommended:
- SKU/Item
- Region/Channel
- Forecast
- On-hand inventory
- Lead time
- Unit cost
        """
    )

st.divider()

# ---------------- Helpers ----------------
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def pick_col(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None

def coerce_date(series: pd.Series) -> pd.Series:
    s = series.copy()
    dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    if dt.notna().mean() > 0.6:
        return dt

    # Try ISO week-like strings
    s2 = s.astype(str).str.strip().str.replace("W", "", regex=False)
    parts = s2.str.split("-", expand=True)
    if parts.shape[1] >= 2:
        y = pd.to_numeric(parts[0], errors="coerce")
        w = pd.to_numeric(parts[1], errors="coerce")
        mask = y.notna() & w.notna()
        out = pd.Series(pd.NaT, index=s.index)
        try:
            out.loc[mask] = pd.to_datetime(
                y[mask].astype(int).astype(str) + "-W" + w[mask].astype(int).astype(str) + "-1",
                format="%G-W%V-%u",
                errors="coerce",
            )
            if out.notna().mean() > 0.6:
                return out
        except Exception:
            pass

    return dt

def safe_num(series):
    return pd.to_numeric(series, errors="coerce")

def wape(actual, forecast):
    denom = np.sum(np.abs(actual))
    if denom == 0:
        return np.nan
    return np.sum(np.abs(actual - forecast)) / denom

def bias_ratio(actual, forecast):
    denom = np.sum(actual)
    if denom == 0:
        return np.nan
    return np.sum(forecast - actual) / denom

def cv(series):
    s = np.array(series, dtype=float)
    mu = np.mean(s)
    if mu == 0:
        return np.nan
    sd = np.std(s, ddof=1) if len(s) > 1 else 0.0
    return sd / mu

def make_demo_dataset():
    """Small, realistic-ish weekly demand planning dataset."""
    rng = pd.date_range("2024-01-01", periods=78, freq="W-MON")
    skus = ["ACC_CABLE", "ACC_MATS", "ACC_CHARGER", "ACC_WIPER", "ACC_FILTER",
            "ACC_CASE", "ACC_RACK", "ACC_SENSOR", "ACC_BADGE", "ACC_KIT", "ACC_COVER", "ACC_ADAPTER"]
    regions = ["US", "EU", "CN"]

    rows = []
    np.random.seed(7)
    for r in regions:
        for sku in skus:
            base = np.random.randint(80, 600)
            season = (np.sin(np.linspace(0, 6*np.pi, len(rng))) + 1.2) * np.random.uniform(0.05, 0.22)
            noise = np.random.normal(0, base * np.random.uniform(0.05, 0.25), len(rng))
            demand = np.maximum(0, base * (1 + season) + noise)

            # some SKUs are volatile / have bias
            if sku in ["ACC_CHARGER", "ACC_SENSOR"]:
                demand = demand * np.random.uniform(0.9, 1.4, len(rng))
            if sku in ["ACC_MATS"]:
                demand[-10:] = demand[-10:] * 1.7  # structural upshift

            forecast = demand * np.random.uniform(0.85, 1.15)  # baseline
            if sku in ["ACC_RACK"]:
                forecast = demand * 1.25  # overforecast bias
            if sku in ["ACC_FILTER"]:
                forecast = demand * 0.78  # underforecast bias

            on_hand = np.random.randint(200, 3500)
            lead_time_days = np.random.choice([10, 14, 21, 28, 35, 45], p=[0.15,0.15,0.25,0.25,0.15,0.05])
            unit_cost = np.random.uniform(8, 180)

            for i, d in enumerate(rng):
                rows.append({
                    "date": d,
                    "sku": sku,
                    "region": r,
                    "demand": float(demand[i]),
                    "forecast": float(forecast[i]),
                    "on_hand": float(on_hand + np.random.randint(-120, 120)),
                    "lead_time_days": float(lead_time_days),
                    "unit_cost": float(unit_cost),
                })
    return pd.DataFrame(rows)

# ---------------- Demo toggle ----------------
if "use_demo" not in st.session_state:
    st.session_state.use_demo = False

left, right = st.columns([1, 2])
with left:
    if st.button("Use Demo Dataset"):
        st.session_state.use_demo = True
with right:
    if st.session_state.use_demo:
        st.success("Demo dataset loaded. You can still upload your own file anytime (refresh page to reset).")

# ---------------- Upload ----------------
df = None
if st.session_state.use_demo:
    df = make_demo_dataset()
else:
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"], accept_multiple_files=False)
    if not uploaded:
        st.stop()

    if uploaded.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

df = normalize_cols(df)

st.subheader("1) Preview")
st.dataframe(df.head(20), use_container_width=True)

# ---------------- Column mapping ----------------
st.subheader("2) Map columns (auto-detect → confirm)")

cols = list(df.columns)

auto_date = pick_col(cols, ["date", "week", "period", "ds", "time", "month"])
auto_sku = pick_col(cols, ["sku", "item", "material", "part", "part_number", "product", "item_id"])
auto_region = pick_col(cols, ["region", "market", "country", "geo", "channel", "site", "dc"])
auto_demand = pick_col(cols, ["demand", "actual", "sales", "consumption", "qty", "units", "actual_qty"])
auto_fcst = pick_col(cols, ["forecast", "fcst", "plan", "prediction", "forecast_qty"])
auto_onhand = pick_col(cols, ["on_hand", "onhand", "inventory", "stock", "oh"])
auto_lt = pick_col(cols, ["lead_time", "leadtime", "lead_time_days", "leadtime_days"])
auto_cost = pick_col(cols, ["unit_cost", "cost", "unit_cost_usd", "cogs", "price"])

c1, c2, c3 = st.columns(3)
with c1:
    date_col = st.selectbox("Date / Week / Period (required)", cols, index=cols.index(auto_date) if auto_date in cols else 0)
    demand_col = st.selectbox("Demand / Actual (required)", cols, index=cols.index(auto_demand) if auto_demand in cols else 0)
    sku_col = st.selectbox("SKU / Item (recommended)", ["(none)"] + cols, index=(["(none)"] + cols).index(auto_sku) if auto_sku in cols else 0)
with c2:
    region_col = st.selectbox("Region / Channel (optional)", ["(none)"] + cols, index=(["(none)"] + cols).index(auto_region) if auto_region in cols else 0)
    forecast_col = st.selectbox("Forecast (optional)", ["(none)"] + cols, index=(["(none)"] + cols).index(auto_fcst) if auto_fcst in cols else 0)
    onhand_col = st.selectbox("On-hand inventory (optional)", ["(none)"] + cols, index=(["(none)"] + cols).index(auto_onhand) if auto_onhand in cols else 0)
with c3:
    lt_col = st.selectbox("Lead time days (optional)", ["(none)"] + cols, index=(["(none)"] + cols).index(auto_lt) if auto_lt in cols else 0)
    cost_col = st.selectbox("Unit cost (optional)", ["(none)"] + cols, index=(["(none)"] + cols).index(auto_cost) if auto_cost in cols else 0)

# ---------------- Standardize ----------------
work = df.copy()
work["__date"] = coerce_date(work[date_col])
work["__demand"] = safe_num(work[demand_col])

if work["__date"].notna().mean() < 0.5:
    st.error("Could not parse your date column reliably. Try selecting a different date/week/period column.")
    st.stop()

if work["__demand"].notna().mean() < 0.5:
    st.error("Could not parse demand/actual as numbers. Try selecting a different demand column.")
    st.stop()

work["__sku"] = work[sku_col] if sku_col != "(none)" else "ALL_SKU"
work["__region"] = work[region_col] if region_col != "(none)" else "ALL_REGION"
work["__forecast"] = safe_num(work[forecast_col]) if forecast_col != "(none)" else np.nan
work["__onhand"] = safe_num(work[onhand_col]) if onhand_col != "(none)" else np.nan
work["__lt"] = safe_num(work[lt_col]) if lt_col != "(none)" else np.nan
work["__cost"] = safe_num(work[cost_col]) if cost_col != "(none)" else np.nan

work = work.dropna(subset=["__date", "__demand"]).copy()
work = work.sort_values("__date")

# ---------------- Controls ----------------
st.subheader("3) Controls")
a, b, c = st.columns(3)

regions = sorted(pd.Series(work["__region"]).dropna().unique().tolist())
skus_all = sorted(pd.Series(work["__sku"]).dropna().unique().tolist())

with a:
    region_sel = st.selectbox("Region", regions[:500] if regions else ["ALL_REGION"])
with b:
    window = st.selectbox("Recent window (weeks)", ["8", "13", "26", "52"], index=0)
with c:
    show_top10 = st.toggle("Show Top 10 problem SKUs", value=True)

weeks = int(window)

# helper: resample a slice to weekly
def to_weekly(slice_df: pd.DataFrame) -> pd.DataFrame:
    slice_df = slice_df.set_index("__date").sort_index()
    weekly = slice_df.resample("W-MON").sum(numeric_only=True).reset_index().rename(columns={"__date": "date"})
    weekly["demand"] = weekly["__demand"]
    weekly["forecast"] = weekly["__forecast"] if forecast_col != "(none)" else np.nan
    return weekly

# ---------------- Top 10 risk SKUs ----------------
if show_top10 and sku_col != "(none)":
    st.subheader("4) Top 10 problem SKUs (auto-ranked)")
    region_df = work[work["__region"] == region_sel].copy()

    # safety: if too many SKUs, sample top by total demand
    sku_counts = region_df.groupby("__sku")["__demand"].sum().sort_values(ascending=False)
    candidate_skus = sku_counts.head(300).index.tolist()  # keep compute fast

    rows = []
    for sku in candidate_skus:
        s = region_df[region_df["__sku"] == sku].copy()
        wk = to_weekly(s).tail(weeks)

        if wk.empty:
            continue

        dmean = float(wk["demand"].mean())
        d_cv = cv(wk["demand"].values)

        has_fcst = forecast_col != "(none)" and wk["forecast"].notna().any()
        if has_fcst:
            w_fc = wk.dropna(subset=["forecast"])
            fc_wape = wape(w_fc["demand"].values, w_fc["forecast"].values) if len(w_fc) else np.nan
            fc_bias = bias_ratio(w_fc["demand"].values, w_fc["forecast"].values) if len(w_fc) else np.nan
        else:
            fc_wape, fc_bias = np.nan, np.nan

        # weeks cover (if onhand exists)
        weeks_cover = np.nan
        if onhand_col != "(none)":
            onhand = safe_num(s["__onhand"]).dropna()
            if len(onhand) and dmean > 0:
                weeks_cover = float(onhand.iloc[-1]) / dmean

        # scoring (explainable)
        score = 0
        reasons = []

        if not np.isnan(d_cv) and d_cv >= 0.60:
            score += 2; reasons.append("High volatility")
        elif not np.isnan(d_cv) and d_cv >= 0.35:
            score += 1; reasons.append("Moderate volatility")

        if has_fcst and not np.isnan(fc_wape):
            if fc_wape >= 0.35:
                score += 2; reasons.append("High forecast error")
            elif fc_wape >= 0.20:
                score += 1; reasons.append("Elevated forecast error")

            if not np.isnan(fc_bias) and abs(fc_bias) >= 0.10:
                score += 1; reasons.append("Bias risk")

        if not np.isnan(weeks_cover):
            if weeks_cover < 1.0:
                score += 2; reasons.append("Low cover (stockout risk)")
            elif weeks_cover > 6.0:
                score += 1; reasons.append("Excess cover")

        top_reason = reasons[0] if reasons else "Stable"

        rows.append({
            "SKU": sku,
            "RiskScore": score,
            "Volatility_CV": d_cv,
            "Forecast_WAPE": fc_wape,
            "Forecast_Bias": fc_bias,
            "Weeks_Cover": weeks_cover,
            "TopReason": top_reason,
        })

    rank = pd.DataFrame(rows).sort_values(["RiskScore", "Volatility_CV"], ascending=False).head(10)

    if rank.empty:
        st.info("Not enough data to rank SKUs. Try increasing the window or check your SKU mapping.")
    else:
        st.dataframe(rank, use_container_width=True)

        # small visual: bar chart of risk scores
        fig = plt.figure()
        plt.barh(rank["SKU"][::-1], rank["RiskScore"][::-1])
        plt.xlabel("Risk Score")
        plt.tight_layout()
        st.pyplot(fig)

    st.divider()

# ---------------- Single SKU deep dive ----------------
st.subheader("5) SKU Deep Dive")

# SKU picker (if available)
if sku_col != "(none)":
    skus_region = sorted(pd.Series(work.loc[work["__region"] == region_sel, "__sku"]).unique().tolist())
    sku_sel = st.selectbox("Select SKU", skus_region[:800] if skus_region else skus_all[:800])
else:
    sku_sel = "ALL_SKU"
    st.info("No SKU column selected — showing an aggregated view (ALL_SKU).")

view = work[(work["__region"] == region_sel) & (work["__sku"] == sku_sel)].copy()
if view.empty:
    st.warning("No data found for this selection.")
    st.stop()

weekly = to_weekly(view)
weekly_recent = weekly.tail(weeks).copy()

# ---------------- Diagnostics ----------------
dmean = float(weekly_recent["demand"].mean())
dstd = float(weekly_recent["demand"].std(ddof=1)) if len(weekly_recent) > 1 else 0.0
d_cv = cv(weekly_recent["demand"].values)

has_fcst = forecast_col != "(none)" and weekly_recent["forecast"].notna().any()
if has_fcst:
    w_fc = weekly_recent.dropna(subset=["forecast"])
    fc_wape = wape(w_fc["demand"].values, w_fc["forecast"].values) if len(w_fc) else np.nan
    fc_bias = bias_ratio(w_fc["demand"].values, w_fc["forecast"].values) if len(w_fc) else np.nan
else:
    fc_wape, fc_bias = np.nan, np.nan

# weeks cover if inventory exists
weeks_cover = np.nan
if onhand_col != "(none)":
    onhand = view["__onhand"].dropna()
    if len(onhand) and dmean > 0:
        weeks_cover = float(onhand.iloc[-1]) / dmean

# ---------------- Situation Overview ----------------
st.subheader("6) Situation Overview")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Avg demand", f"{dmean:,.0f}")
k2.metric("Volatility (CV)", "N/A" if np.isnan(d_cv) else f"{d_cv:.2f}")
k3.metric("Forecast WAPE", "N/A" if np.isnan(fc_wape) else f"{fc_wape:.1%}")
k4.metric("Forecast bias", "N/A" if np.isnan(fc_bias) else f"{fc_bias:+.1%}")

with st.container(border=True):
    st.markdown("**Demand trend (recent window)**")
    fig = plt.figure()
    plt.plot(weekly_recent["date"], weekly_recent["demand"])
    if has_fcst:
        plt.plot(weekly_recent["date"], weekly_recent["forecast"])
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

# ---------------- Problem detection + decision ----------------
problems = []
score = 0

# Volatility
if not np.isnan(d_cv) and d_cv >= 0.60:
    problems.append("High demand volatility — forecast risk elevated.")
    score += 2
elif not np.isnan(d_cv) and d_cv >= 0.35:
    problems.append("Moderate volatility — monitor and refine assumptions.")
    score += 1

# Forecast accuracy/bias
if has_fcst and not np.isnan(fc_wape):
    if fc_wape >= 0.35:
        problems.append("Forecast error is high (WAPE ≥ 35%) — needs driver/method refresh.")
        score += 2
    elif fc_wape >= 0.20:
        problems.append("Forecast error elevated (WAPE 20–35%) — review drivers and bias.")
        score += 1
    if not np.isnan(fc_bias) and abs(fc_bias) >= 0.10:
        problems.append("Systematic bias detected (|bias| ≥ 10%) — adjust assumptions/overrides.")
        score += 1

# Inventory cover
if not np.isnan(weeks_cover):
    if weeks_cover < 1.0:
        problems.append("Low weeks-of-cover (< 1.0) — stockout/service risk.")
        score += 2
    elif weeks_cover > 6.0:
        problems.append("High weeks-of-cover (> 6.0) — excess/obsolescence risk.")
        score += 1

# Decision mode
decision_mode = "Balanced"
tradeoff = "Take targeted actions while monitoring signals."

if (not np.isnan(weeks_cover) and weeks_cover < 1.0) or (has_fcst and not np.isnan(fc_wape) and fc_wape >= 0.35) or (not np.isnan(d_cv) and d_cv >= 0.60):
    decision_mode = "Service"
    tradeoff = "Accept higher short-term cost/effort to protect availability while stabilizing the plan."
elif (not np.isnan(weeks_cover) and weeks_cover > 6.0) and (not np.isnan(d_cv) and d_cv < 0.35):
    decision_mode = "Cost/Cash"
    tradeoff = "Reduce excess exposure; accept slower service improvement to avoid write-offs."

confidence = "High"
if len(weekly_recent) < 6:
    confidence = "Low"
elif not has_fcst:
    confidence = "Medium"

# Owners
if decision_mode == "Service":
    owners = [
        "Demand Planning: refresh near-term assumptions; adjust overrides on top SKUs",
        "Inventory/Material Planning: rebalance allocation; increase review cadence",
        "Supplier/Procurement: confirm lead-time / expedite options",
        "Operations: prioritize constrained SKUs if needed",
    ]
elif decision_mode == "Cost/Cash":
    owners = [
        "Demand Planning: reduce optimistic bias; tune overrides",
        "Inventory: slow replenishment; manage excess exposure",
        "Leadership: align on risk tolerance and target cover",
    ]
else:
    owners = [
        "Demand Planning: refine assumptions; monitor top drivers weekly",
        "Cross-functional: short weekly check on high-impact SKUs",
    ]

# ---------------- Outputs ----------------
st.subheader("7) Recommendation")

with st.container(border=True):
    x, y, z = st.columns(3)
    x.metric("Decision Mode", decision_mode)
    y.metric("Decision Score", score)
    z.metric("Confidence", confidence)
    st.write("**Accepted trade-off:**", tradeoff)

    tags = []
    tags.append("📈 Volatile" if (not np.isnan(d_cv) and d_cv >= 0.35) else "📉 Stable")
    tags.append("🎯 Forecast risk" if (has_fcst and not np.isnan(fc_wape) and fc_wape >= 0.20) else "✅ Forecast OK / N/A")
    if not np.isnan(weeks_cover):
        tags.append("📦 Low cover" if weeks_cover < 1.0 else "📦 Excess" if weeks_cover > 6.0 else "📦 Healthy cover")
    st.caption(" | ".join(tags))

with st.container(border=True):
    st.markdown("**Problem overview (auto-detected)**")
    if problems:
        for p in problems[:6]:
            st.write("•", p)
    else:
        st.write("No major issues detected in the selected window.")

with st.container(border=True):
    st.markdown("**Who should act next**")
    for o in owners:
        st.write("•", o)

# ---------------- Export ----------------
export = pd.DataFrame([{
    "region": region_sel,
    "sku": sku_sel,
    "window_weeks": weeks,
    "demand_mean": dmean,
    "demand_cv": d_cv,
    "forecast_wape": fc_wape,
    "forecast_bias": fc_bias,
    "weeks_cover": weeks_cover,
    "decision_mode": decision_mode,
    "decision_score": score,
    "confidence": confidence,
    "tradeoff": tradeoff,
    "top_problems": " | ".join(problems[:6]),
}])

st.download_button(
    "Download summary (CSV)",
    data=export.to_csv(index=False).encode("utf-8"),
    file_name="planpulse_summary.csv",
    mime="text/csv",
)
