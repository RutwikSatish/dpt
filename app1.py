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
    .pp-pill {
        display:inline-block; padding: .25rem .6rem; border-radius: 999px;
        border: 1px solid rgba(255,255,255,.12);
        margin-right: .35rem; margin-bottom: .35rem;
        font-size: .85rem;
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

# =======================
# Helpers
# =======================
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

    # Try ISO week-like strings: "YYYY-WW" or "YYYY-WWW"
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

def clamp(x, lo, hi):
    if np.isnan(x):
        return np.nan
    return max(lo, min(hi, x))

def pattern_classification(demand_series: np.ndarray):
    """
    Lightweight demand pattern classification to boost planner credibility.
    Uses CV + zero-frequency as a proxy for: smooth/intermittent/erratic/lumpy.
    """
    d = np.array(demand_series, dtype=float)
    if len(d) == 0:
        return "Unknown", np.nan, np.nan

    zero_rate = float(np.mean(d <= 0))
    d_cv = cv(d)

    # Simple, explainable rules:
    if zero_rate >= 0.35 and (np.isnan(d_cv) or d_cv >= 0.80):
        return "Lumpy", d_cv, zero_rate
    if zero_rate >= 0.35 and (np.isnan(d_cv) or d_cv < 0.80):
        return "Intermittent", d_cv, zero_rate
    if zero_rate < 0.35 and (not np.isnan(d_cv) and d_cv >= 0.60):
        return "Erratic", d_cv, zero_rate
    return "Smooth", d_cv, zero_rate

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

            # make some intermittent/lumpy behavior for realism
            if sku in ["ACC_BADGE", "ACC_KIT"]:
                mask = np.random.rand(len(rng)) < 0.25
                demand[mask] = 0

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

# =======================
# Demo toggle
# =======================
if "use_demo" not in st.session_state:
    st.session_state.use_demo = False

left, right = st.columns([1, 2])
with left:
    if st.button("Use Demo Dataset"):
        st.session_state.use_demo = True
with right:
    if st.session_state.use_demo:
        st.success("Demo dataset loaded. You can still upload your own file anytime (refresh page to reset).")

# =======================
# Upload
# =======================
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

# =======================
# Column mapping
# =======================
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

# =======================
# Standardize
# =======================
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

# =======================
# Controls (time-horizon awareness)
# =======================
st.subheader("3) Controls")
a, b, c, d = st.columns([1, 1, 1, 1])

regions = sorted(pd.Series(work["__region"]).dropna().unique().tolist())
skus_all = sorted(pd.Series(work["__sku"]).dropna().unique().tolist())

with a:
    region_sel = st.selectbox("Region", regions[:500] if regions else ["ALL_REGION"])
with b:
    exec_window = st.selectbox("Execution horizon (weeks)", ["8", "13"], index=0)
with c:
    plan_window = st.selectbox("Planning horizon (weeks)", ["26", "52"], index=0)
with d:
    show_top10 = st.toggle("Show Top 10 problem SKUs", value=True)

exec_weeks = int(exec_window)
plan_weeks = int(plan_window)

# helper: resample a slice to weekly
def to_weekly(slice_df: pd.DataFrame) -> pd.DataFrame:
    slice_df = slice_df.set_index("__date").sort_index()
    weekly = slice_df.resample("W-MON").sum(numeric_only=True).reset_index().rename(columns={"__date": "date"})
    weekly["demand"] = weekly["__demand"]
    weekly["forecast"] = weekly["__forecast"] if forecast_col != "(none)" else np.nan
    return weekly

# =======================
# Risk scoring (explainable)
# =======================
def risk_decomposition(d_cv, fc_wape, fc_bias, weeks_cover):
    contrib = {"Volatility": 0.0, "ForecastError": 0.0, "Bias": 0.0, "Cover": 0.0}
    reasons = []

    if not np.isnan(d_cv):
        if d_cv >= 0.60:
            contrib["Volatility"] = 2.0; reasons.append("High volatility")
        elif d_cv >= 0.35:
            contrib["Volatility"] = 1.0; reasons.append("Moderate volatility")

    if not np.isnan(fc_wape):
        if fc_wape >= 0.35:
            contrib["ForecastError"] = 2.0; reasons.append("High forecast error")
        elif fc_wape >= 0.20:
            contrib["ForecastError"] = 1.0; reasons.append("Elevated forecast error")

    if not np.isnan(fc_bias) and abs(fc_bias) >= 0.10:
        contrib["Bias"] = 1.0; reasons.append("Bias risk")

    if not np.isnan(weeks_cover):
        if weeks_cover < 1.0:
            contrib["Cover"] = 2.0; reasons.append("Low cover (stockout risk)")
        elif weeks_cover > 6.0:
            contrib["Cover"] = 1.0; reasons.append("Excess cover")

    total = float(sum(contrib.values()))
    top_reason = reasons[0] if reasons else "Stable"
    return total, contrib, top_reason, reasons

def normalize_risk_score(contrib: dict, has_fcst: bool, has_onhand: bool):
    """
    Normalizes risk to 0–100 and adds a light penalty for missing signals so rankings are fair.
    """
    max_points = 2.0 + 2.0 + 1.0 + 2.0  # Vol + FcErr + Bias + Cover = 7
    raw = float(sum(contrib.values()))

    missing_penalty = 0.0
    if not has_fcst:
        missing_penalty += 0.5
    if not has_onhand:
        missing_penalty += 0.5

    adj = raw + missing_penalty
    score_0_100 = 100.0 * adj / (max_points + 1.0)  # +1 headroom for penalties
    score_0_100 = float(np.clip(score_0_100, 0, 100))

    signals = []
    signals.append("Forecast" if has_fcst else "NoForecast")
    signals.append("Inventory" if has_onhand else "NoInventory")
    return score_0_100, " | ".join(signals), missing_penalty

def show_contrib_chart(contrib: dict, title: str = "Risk decomposition"):
    labels = list(contrib.keys())
    vals = [contrib[k] for k in labels]

   fig, ax = make_fig(5.5, 2.4)   # smaller
    ax.bar(labels, vals)
    ax.set_ylabel("Risk points")
    ax.set_title(title)
    ax.set_ylim(0, max(3, max(vals) + 0.5))
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()

    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

# =======================
# Top 10 risk SKUs
# =======================
rank = pd.DataFrame()  # for exception log export
if show_top10 and sku_col != "(none)":
    st.subheader("4) Top 10 problem SKUs (auto-ranked)")
    region_df = work[work["__region"] == region_sel].copy()

    sku_counts = region_df.groupby("__sku")["__demand"].sum().sort_values(ascending=False)
    candidate_skus = sku_counts.head(300).index.tolist()

    rows = []
    for sku in candidate_skus:
        s = region_df[region_df["__sku"] == sku].copy()
        wk = to_weekly(s)

        exec_slice = wk.tail(exec_weeks)
        if exec_slice.empty:
            continue

        dmean = float(exec_slice["demand"].mean())
        d_cv = cv(exec_slice["demand"].values)

        has_fcst = forecast_col != "(none)" and exec_slice["forecast"].notna().any()
        if has_fcst:
            w_fc = exec_slice.dropna(subset=["forecast"])
            fc_wape = wape(w_fc["demand"].values, w_fc["forecast"].values) if len(w_fc) else np.nan
            fc_bias = bias_ratio(w_fc["demand"].values, w_fc["forecast"].values) if len(w_fc) else np.nan
        else:
            fc_wape, fc_bias = np.nan, np.nan

        weeks_cover = np.nan
        if onhand_col != "(none)":
            onhand = safe_num(s["__onhand"]).dropna()
            if len(onhand) and dmean > 0:
                weeks_cover = float(onhand.iloc[-1]) / dmean

        total_score, contrib, top_reason, _reasons = risk_decomposition(d_cv, fc_wape, fc_bias, weeks_cover)

        has_onhand = (onhand_col != "(none)") and (not np.isnan(weeks_cover))
        risk_0_100, signals_present, miss_pen = normalize_risk_score(contrib, has_fcst, has_onhand)

        rows.append({
            "SKU": sku,
            "RiskScore": total_score,
            "RiskScore_0_100": risk_0_100,
            "Signals": signals_present,
            "MissingPenalty": miss_pen,
            "Volatility_CV": d_cv,
            "Forecast_WAPE": fc_wape,
            "Forecast_Bias": fc_bias,
            "Weeks_Cover": weeks_cover,
            "TopReason": top_reason,
            "Pts_Vol": contrib["Volatility"],
            "Pts_FcErr": contrib["ForecastError"],
            "Pts_Bias": contrib["Bias"],
            "Pts_Cover": contrib["Cover"],
        })

    rank = pd.DataFrame(rows).sort_values(
        ["RiskScore_0_100", "RiskScore", "Volatility_CV"], ascending=False
    ).head(10)

    if rank.empty:
        st.info("Not enough data to rank SKUs. Try increasing the horizon or check your SKU mapping.")
    else:
        st.dataframe(rank, use_container_width=True)

        fig, ax = plt.subplots()
        ax.barh(rank["SKU"][::-1], rank["RiskScore_0_100"][::-1])
        ax.set_xlabel("Risk Score (0–100)")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

        st.markdown("**Explain a ranked SKU**")
        sku_explain = st.selectbox("Pick from Top 10", rank["SKU"].tolist(), index=0)
        row = rank[rank["SKU"] == sku_explain].iloc[0].to_dict()
        contrib_explain = {
            "Volatility": float(row["Pts_Vol"]),
            "ForecastError": float(row["Pts_FcErr"]),
            "Bias": float(row["Pts_Bias"]),
            "Cover": float(row["Pts_Cover"]),
        }
        show_contrib_chart(contrib_explain, title=f"Risk decomposition — {sku_explain}")

        st.markdown("**Download exception log (for weekly planning cadence)**")
        exception_log = rank.copy()
        exception_log["Owner"] = np.where(
            exception_log["TopReason"].str.contains("cover|stockout|excess", case=False, na=False),
            "Inventory",
            np.where(
                exception_log["TopReason"].str.contains("forecast|bias|error", case=False, na=False),
                "Demand Planning",
                "Demand Planning"
            )
        )
        exception_log["SuggestedAction"] = np.where(
            exception_log["TopReason"].str.contains("Low cover", case=False, na=False),
            "Expedite / reallocate / raise safety stock (short-term)",
            np.where(
                exception_log["TopReason"].str.contains("Excess", case=False, na=False),
                "Reduce POs / run-down plan / redeploy / markdown",
                np.where(
                    exception_log["TopReason"].str.contains("Bias", case=False, na=False),
                    "Calibrate overrides / fix baseline assumptions",
                    "Review drivers / segmentation / event calendar"
                )
            )
        )

        st.download_button(
            "Download exception log (CSV)",
            data=exception_log.to_csv(index=False).encode("utf-8"),
            file_name="planpulse_exception_log.csv",
            mime="text/csv",
        )

    st.divider()

# =======================
# Single SKU deep dive
# =======================
st.subheader("5) SKU Deep Dive")

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
weekly_exec = weekly.tail(exec_weeks).copy()
weekly_plan = weekly.tail(plan_weeks).copy()

def compute_kpis(wk: pd.DataFrame):
    dmean = float(wk["demand"].mean()) if len(wk) else np.nan
    d_cv = cv(wk["demand"].values) if len(wk) else np.nan

    has_fcst = forecast_col != "(none)" and wk["forecast"].notna().any()
    if has_fcst:
        w_fc = wk.dropna(subset=["forecast"])
        fc_wape = wape(w_fc["demand"].values, w_fc["forecast"].values) if len(w_fc) else np.nan
        fc_bias = bias_ratio(w_fc["demand"].values, w_fc["forecast"].values) if len(w_fc) else np.nan
    else:
        fc_wape, fc_bias = np.nan, np.nan

    weeks_cover = np.nan
    if onhand_col != "(none)":
        onhand = view["__onhand"].dropna()
        if len(onhand) and dmean and dmean > 0:
            weeks_cover = float(onhand.iloc[-1]) / dmean

    pattern, _pattern_cv, zero_rate = pattern_classification(wk["demand"].values)

    return {
        "dmean": dmean, "d_cv": d_cv,
        "fc_wape": fc_wape, "fc_bias": fc_bias,
        "weeks_cover": weeks_cover,
        "has_fcst": has_fcst,
        "pattern": pattern,
        "zero_rate": zero_rate,
    }

k_exec = compute_kpis(weekly_exec)
k_plan = compute_kpis(weekly_plan)

# =======================
# Situation Overview
# =======================
st.subheader("6) Situation Overview")

topL, topR = st.columns([1, 1])

with topL:
    st.markdown("**Execution horizon (near-term)**")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Avg demand", "N/A" if np.isnan(k_exec["dmean"]) else f"{k_exec['dmean']:,.0f}")
    e2.metric("Volatility (CV)", "N/A" if np.isnan(k_exec["d_cv"]) else f"{k_exec['d_cv']:.2f}")
    e3.metric("Forecast WAPE", "N/A" if np.isnan(k_exec["fc_wape"]) else f"{k_exec['fc_wape']:.1%}")
    e4.metric("Forecast bias", "N/A" if np.isnan(k_exec["fc_bias"]) else f"{k_exec['fc_bias']:+.1%}")

with topR:
    st.markdown("**Planning horizon (mid-term)**")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Avg demand", "N/A" if np.isnan(k_plan["dmean"]) else f"{k_plan['dmean']:,.0f}")
    p2.metric("Volatility (CV)", "N/A" if np.isnan(k_plan["d_cv"]) else f"{k_plan['d_cv']:.2f}")
    p3.metric("Forecast WAPE", "N/A" if np.isnan(k_plan["fc_wape"]) else f"{k_plan['fc_wape']:.1%}")
    p4.metric("Forecast bias", "N/A" if np.isnan(k_plan["fc_bias"]) else f"{k_plan['fc_bias']:+.1%}")

pattern = k_exec["pattern"]
zero_rate = k_exec["zero_rate"]
st.markdown(
    f"""
    <span class="pp-pill">📌 Pattern: <b>{pattern}</b></span>
    <span class="pp-pill">🧊 Zero-demand rate: <b>{'N/A' if np.isnan(zero_rate) else f'{zero_rate:.0%}'}</b></span>
    """,
    unsafe_allow_html=True
)

with st.container(border=True):
    st.markdown("**Demand & forecast trend (Execution horizon)**")
    fig, ax = plt.subplots()
    ax.plot(weekly_exec["date"], weekly_exec["demand"], label="Demand")
    if k_exec["has_fcst"]:
        ax.plot(weekly_exec["date"], weekly_exec["forecast"], label="Forecast")
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

# =======================
# Planner Override Simulation (What-if)
# =======================
st.subheader("6.5) Planner Override Simulation (What-if)")

with st.container(border=True):
    colA, colB, colC, colD = st.columns([1, 1, 1, 1])

    if forecast_col != "(none)":
        with colA:
            fc_mult = st.slider("Forecast override (multiplier)", 0.70, 1.30, 1.00, 0.01)
    else:
        fc_mult = 1.0
        with colA:
            st.info("No forecast column → override simulation disabled.")

    with colB:
        target_cover = st.slider("Target weeks of cover", 0.5, 12.0, 4.0, 0.5)

    with colC:
        preference = st.selectbox("Decision preference", ["Balanced", "Service-first", "Cash-first"], index=0)

    with colD:
        override_scope = st.selectbox(
            "Apply override to",
            ["Execution horizon only", "Next 4 weeks only"],
            index=0
        )

    wk_sim = weekly_exec.copy()
    if forecast_col != "(none)" and wk_sim["forecast"].notna().any():
        wk_sim["forecast_sim"] = wk_sim["forecast"].copy()

        if override_scope == "Next 4 weeks only" and len(wk_sim) >= 4:
            wk_sim.loc[wk_sim.index[-4:], "forecast_sim"] = wk_sim.loc[wk_sim.index[-4:], "forecast"] * fc_mult
        else:
            wk_sim["forecast_sim"] = wk_sim["forecast"] * fc_mult

        w_fc = wk_sim.dropna(subset=["forecast_sim"])
        sim_wape = wape(w_fc["demand"].values, w_fc["forecast_sim"].values) if len(w_fc) else np.nan
        sim_bias = bias_ratio(w_fc["demand"].values, w_fc["forecast_sim"].values) if len(w_fc) else np.nan
    else:
        sim_wape, sim_bias = np.nan, np.nan

# =======================
# Decision engine (fixed: no hidden global dependencies)
# =======================
def decision_engine(kpis_exec, weekly_exec_df, sim_wape, sim_bias, target_cover, preference="Balanced"):
    problems = []
    owners = []

    d_cv = kpis_exec["d_cv"]
    fc_wape = sim_wape if not np.isnan(sim_wape) else kpis_exec["fc_wape"]
    fc_bias = sim_bias if not np.isnan(sim_bias) else kpis_exec["fc_bias"]
    weeks_cover = kpis_exec["weeks_cover"]

    total_score, contrib, top_reason, reasons = risk_decomposition(d_cv, fc_wape, fc_bias, weeks_cover)

    pattern = kpis_exec["pattern"]
    if pattern in ["Intermittent", "Lumpy"]:
        problems.append(
            f"Demand pattern is {pattern} — error will be structurally higher; consider intermittent methods (e.g., Croston/TSB) or order-based planning."
        )
        contrib["Volatility"] += 0.5

    if not np.isnan(d_cv) and d_cv >= 0.60:
        problems.append("High demand volatility — near-term execution risk elevated (allocation/review cadence).")
    elif not np.isnan(d_cv) and d_cv >= 0.35:
        problems.append("Moderate volatility — monitor; tighten assumptions and event calendar.")

    if not np.isnan(fc_wape):
        if fc_wape >= 0.35:
            problems.append("Forecast error is high (WAPE ≥ 35%) — refresh drivers/model; review promo/events, NPI, substitution.")
        elif fc_wape >= 0.20:
            problems.append("Forecast error elevated (WAPE 20–35%) — review key drivers; reduce manual noise.")
    if not np.isnan(fc_bias) and abs(fc_bias) >= 0.10:
        problems.append("Systematic bias detected (|bias| ≥ 10%) — calibrate overrides / safety stock assumptions.")

    cover_gap = np.nan
    if not np.isnan(weeks_cover):
        cover_gap = weeks_cover - target_cover
        if weeks_cover < max(1.0, target_cover * 0.5):
            problems.append(f"Cover is materially low vs target ({weeks_cover:.1f}w vs {target_cover:.1f}w) — service/stockout risk.")
            contrib["Cover"] = max(contrib["Cover"], 2.0)
        elif weeks_cover > max(6.0, target_cover * 1.5):
            problems.append(f"Cover is high vs target ({weeks_cover:.1f}w vs {target_cover:.1f}w) — excess/obsolescence risk.")
            contrib["Cover"] = max(contrib["Cover"], 1.0)

    decision_mode = "Balanced"
    tradeoff = "Take targeted actions while monitoring signals."
    confidence = "High"

    if len(weekly_exec_df) < 6:
        confidence = "Low"
    elif np.isnan(fc_wape) and np.isnan(d_cv):
        confidence = "Medium"

    service_trigger = (
        (not np.isnan(weeks_cover) and weeks_cover < 1.0) or
        (not np.isnan(fc_wape) and fc_wape >= 0.35) or
        (not np.isnan(d_cv) and d_cv >= 0.60)
    )
    cash_trigger = (
        (not np.isnan(weeks_cover) and weeks_cover > 6.0) and
        (not np.isnan(d_cv) and d_cv < 0.35)
    )

    if preference == "Service-first":
        service_trigger = service_trigger or (not np.isnan(weeks_cover) and weeks_cover < target_cover)
    if preference == "Cash-first":
        cash_trigger = cash_trigger or (not np.isnan(weeks_cover) and weeks_cover > target_cover)

    if service_trigger:
        decision_mode = "Service"
        tradeoff = "Accept higher short-term cost/effort to protect availability while stabilizing the plan."
        owners = [
            "Demand Planning: refresh near-term assumptions; tighten override governance on top SKUs",
            "Inventory/Material Planning: rebalance allocation; increase review cadence; set exception alerts",
            "Supplier/Procurement: validate lead-time; evaluate expedite options / MOQ constraints",
            "Operations: prioritize constrained SKUs if needed",
        ]
    elif cash_trigger:
        decision_mode = "Cost/Cash"
        tradeoff = "Reduce excess exposure; accept slower service improvement to avoid write-offs."
        owners = [
            "Demand Planning: reduce optimistic bias; tune overrides; align forecast to latest signals",
            "Inventory: slow/stop replenishment; manage excess; redeploy/markdown if applicable",
            "Leadership: align risk tolerance and target cover; approve liquidation strategy if needed",
        ]
    else:
        owners = [
            "Demand Planning: refine assumptions; monitor top drivers weekly",
            "Cross-functional: short weekly check on high-impact SKUs and constraints",
        ]

    total_score = float(sum(contrib.values()))
    return {
        "decision_mode": decision_mode,
        "tradeoff": tradeoff,
        "confidence": confidence,
        "score": total_score,
        "contrib": contrib,
        "problems": problems[:8],
        "owners": owners,
        "top_reason": top_reason,
        "fc_wape_used": fc_wape,
        "fc_bias_used": fc_bias,
        "cover_gap": cover_gap,
    }

# Latest values (for LT-aware adjustments & business impact)
onhand_latest = np.nan
unit_cost_latest = np.nan
lt_latest = np.nan

if onhand_col != "(none)":
    oh = view["__onhand"].dropna()
    if len(oh):
        onhand_latest = float(oh.iloc[-1])

if cost_col != "(none)":
    uc = view["__cost"].dropna()
    if len(uc):
        unit_cost_latest = float(uc.iloc[-1])

if lt_col != "(none)":
    lt = view["__lt"].dropna()
    if len(lt):
        lt_latest = float(lt.iloc[-1])

decision = decision_engine(k_exec, weekly_exec, sim_wape, sim_bias, target_cover, preference=preference)

# Lead-time realism: long LT reduces execution feasibility (service mode)
if not np.isnan(lt_latest) and lt_latest > 35 and decision["decision_mode"] == "Service":
    decision["problems"] = ["Long lead time — inventory changes may not land in time; consider allocation/expedite."] + decision["problems"]
    if decision["confidence"] == "High":
        decision["confidence"] = "Medium"

# =======================
# Quantified impact estimates
# =======================
def quantify_tradeoffs(decision_mode, weeks_cover, target_cover, onhand_latest, unit_cost_latest):
    impact = {
        "inv_delta_units": np.nan,
        "cash_delta": np.nan,
        "service_delta_pts": np.nan
    }

    if np.isnan(weeks_cover) or np.isnan(onhand_latest):
        return impact

    if decision_mode == "Cost/Cash":
        desired_cover = min(6.0, target_cover)
    elif decision_mode == "Service":
        desired_cover = max(2.0, target_cover)
    else:
        desired_cover = target_cover

    if weeks_cover <= 0:
        return impact

    mean_demand_est = onhand_latest / weeks_cover
    inv_target = desired_cover * mean_demand_est
    inv_delta = inv_target - onhand_latest  # + build, - reduce

    impact["inv_delta_units"] = float(inv_delta)

    # cap unrealistic swings
    cap = onhand_latest * 0.75
    impact["inv_delta_units"] = float(np.clip(impact["inv_delta_units"], -cap, cap))

    if not np.isnan(unit_cost_latest):
        impact["cash_delta"] = -impact["inv_delta_units"] * unit_cost_latest  # reducing inv => +cash

    if decision_mode == "Service":
        impact["service_delta_pts"] = clamp((desired_cover - weeks_cover) * 1.0, -0.5, 3.0)
    elif decision_mode == "Cost/Cash":
        impact["service_delta_pts"] = clamp((desired_cover - weeks_cover) * 0.5, -2.0, 0.5)
    else:
        impact["service_delta_pts"] = clamp((desired_cover - weeks_cover) * 0.3, -1.0, 1.0)

    return impact

impact = quantify_tradeoffs(
    decision_mode=decision["decision_mode"],
    weeks_cover=k_exec["weeks_cover"],
    target_cover=target_cover,
    onhand_latest=onhand_latest,
    unit_cost_latest=unit_cost_latest,
)

# =======================
# Recommendation Output
# =======================
st.subheader("7) Recommendation")

with st.container(border=True):
    x, y, z = st.columns(3)
    x.metric("Decision Mode", decision["decision_mode"])
    y.metric("Decision Score", f"{decision['score']:.1f}")
    z.metric("Confidence", decision["confidence"])
    st.write("**Accepted trade-off:**", decision["tradeoff"])

    tags = []
    tags.append("📈 Volatile" if (not np.isnan(k_exec["d_cv"]) and k_exec["d_cv"] >= 0.35) else "📉 Stable")
    tags.append("🎯 Forecast risk" if (not np.isnan(decision["fc_wape_used"]) and decision["fc_wape_used"] >= 0.20) else "✅ Forecast OK / N/A")

    if not np.isnan(k_exec["weeks_cover"]):
        if k_exec["weeks_cover"] < target_cover:
            tags.append(f"📦 Below target cover ({k_exec['weeks_cover']:.1f}w < {target_cover:.1f}w)")
        elif k_exec["weeks_cover"] > target_cover:
            tags.append(f"📦 Above target cover ({k_exec['weeks_cover']:.1f}w > {target_cover:.1f}w)")
        else:
            tags.append("📦 On target cover")

    if not np.isnan(lt_latest):
        tags.append(f"⏱️ LT {lt_latest:.0f}d")

    tags.append(f"🧭 Pref: {preference}")
    st.caption(" | ".join(tags))

with st.expander("Metric definitions (quick reference)"):
    st.write("**WAPE** = Σ|A−F| / Σ|A|. Scale-independent forecast error.")
    st.write("**Bias** = Σ(F−A) / Σ(A). Positive = over-forecast; Negative = under-forecast.")
    st.write("**CV** = stdev / mean. Higher CV = more volatility.")
    st.write("**Weeks of cover** = On-hand / Avg weekly demand (in selected horizon).")

with st.container(border=True):
    st.markdown("**Why this recommendation? (Explainable risk decomposition)**")
    st.write(
        f"Top driver: **{decision['top_reason']}**. "
        "Risk points below show the exact contribution of each factor."
    )
    show_contrib_chart(decision["contrib"], title=f"Risk decomposition — {sku_sel} (Execution horizon)")

with st.container(border=True):
    st.markdown("**Estimated business impact (heuristic)**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest on-hand", "N/A" if np.isnan(onhand_latest) else f"{onhand_latest:,.0f}")
    c2.metric("Weeks of cover", "N/A" if np.isnan(k_exec["weeks_cover"]) else f"{k_exec['weeks_cover']:.1f}w")
    c3.metric("Inventory move (units)", "N/A" if np.isnan(impact["inv_delta_units"]) else f"{impact['inv_delta_units']:+,.0f}")
    if np.isnan(impact["cash_delta"]):
        c4.metric("Cash impact (est.)", "N/A")
    else:
        c4.metric("Cash impact (est.)", f"{impact['cash_delta']:+,.0f}")

    if not np.isnan(impact["service_delta_pts"]):
        st.caption(f"Service impact (rough): {impact['service_delta_pts']:+.1f} pts (conservative estimate)")

with st.container(border=True):
    st.markdown("**Problem overview (auto-detected)**")
    if decision["problems"]:
        for p in decision["problems"]:
            st.write("•", p)
    else:
        st.write("No major issues detected in the selected window.")

with st.container(border=True):
    st.markdown("**Who should act next**")
    for o in decision["owners"]:
        st.write("•", o)

st.text_area(
    "Planner note (copy/paste into Demand Review)",
    value=(
        f"SKU {sku_sel} | Region {region_sel}\n"
        f"Mode: {decision['decision_mode']} | Score: {decision['score']:.1f} | Confidence: {decision['confidence']}\n"
        f"Drivers: {decision['top_reason']} | Pattern: {k_exec['pattern']}\n"
        f"Key issues: {'; '.join(decision['problems'][:3]) if decision['problems'] else 'None'}\n"
        f"Next actions: {decision['owners'][0] if decision['owners'] else 'N/A'}"
    ),
    height=140
)

# =======================
# Export
# =======================
export = pd.DataFrame([{
    "region": region_sel,
    "sku": sku_sel,
    "exec_horizon_weeks": exec_weeks,
    "plan_horizon_weeks": plan_weeks,
    "pattern_exec": k_exec["pattern"],
    "zero_rate_exec": k_exec["zero_rate"],
    "demand_mean_exec": k_exec["dmean"],
    "demand_cv_exec": k_exec["d_cv"],
    "forecast_wape_used": decision["fc_wape_used"],
    "forecast_bias_used": decision["fc_bias_used"],
    "weeks_cover_exec": k_exec["weeks_cover"],
    "target_cover": target_cover,
    "lead_time_days_latest": lt_latest,
    "decision_preference": preference,
    "decision_mode": decision["decision_mode"],
    "decision_score": decision["score"],
    "confidence": decision["confidence"],
    "tradeoff": decision["tradeoff"],
    "risk_pts_volatility": decision["contrib"]["Volatility"],
    "risk_pts_forecast_error": decision["contrib"]["ForecastError"],
    "risk_pts_bias": decision["contrib"]["Bias"],
    "risk_pts_cover": decision["contrib"]["Cover"],
    "latest_onhand": onhand_latest,
    "latest_unit_cost": unit_cost_latest,
    "inv_delta_units_est": impact["inv_delta_units"],
    "cash_delta_est": impact["cash_delta"],
    "service_delta_pts_est": impact["service_delta_pts"],
    "top_problems": " | ".join(decision["problems"]),
}])

st.download_button(
    "Download summary (CSV)",
    data=export.to_csv(index=False).encode("utf-8"),
    file_name="planpulse_summary.csv",
    mime="text/csv",
)
