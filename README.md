# 📦 PlanPulse — Demand Planning Auto-Analyst

**PlanPulse** is an explainable, execution-focused **demand planning decision support tool** built to help planners quickly identify risk, prioritize SKUs, and make informed service vs cost trade-offs using industry-standard metrics.

> This project was built as a **learning-by-doing exercise** to deeply understand how real demand planning decisions are made — not just how forecasts are calculated.

---

## 🚀 What Problem Does PlanPulse Solve?

In real-world demand planning:

* Data is noisy and incomplete
* Forecast accuracy alone doesn’t tell you **what to do next**
* Planners must balance **service, inventory, and cash** under time pressure

PlanPulse focuses on **decision clarity**, not black-box prediction.

It answers:

* *Which SKUs need attention right now?*
* *Why are they risky?*
* *Should we prioritize service, cost, or stay balanced?*
* *Who should act next — demand planning, inventory, procurement, or leadership?*

---

## 🧠 Key Features

### 🔎 Automated Risk Detection

* Auto-ranks **Top-10 problem SKUs** by execution risk
* Uses explainable scoring based on:

  * Demand volatility (CV)
  * Forecast error (WAPE)
  * Forecast bias
  * Weeks of cover

### 🧭 Execution vs Planning Horizons

* Separates **near-term execution** (8–13 weeks) from **mid-term planning** (26–52 weeks)
* Prevents overreaction to short-term noise

### 📌 Demand Pattern Classification

* Identifies **Smooth / Erratic / Intermittent / Lumpy** demand
* Adjusts interpretation of forecast error accordingly

### 🧮 Explainable Decision Logic

* Every recommendation includes:

  * Risk decomposition by driver
  * Top reason for the decision
  * Accepted trade-off (Service vs Cost vs Balanced)

### 🔁 Planner Override Simulation

* Test “what-if” scenarios:

  * Forecast overrides (full horizon or next 4 weeks)
  * Target weeks of cover
  * Service-first vs Cash-first preferences

### 💰 Quantified Business Impact (Heuristic)

* Estimates:

  * Inventory change (units)
  * Cash impact
  * Service impact (directional)
* Caps unrealistic inventory moves
* Adjusts confidence for long lead times

### 📤 Exception-Based Outputs

* Downloadable:

  * **SKU-level decision summary**
  * **Exception log** for weekly planning cadence
* Copy-paste planner note for Demand Review meetings

---

## 📊 Metrics Used (Industry-Standard)

| Metric                        | Purpose                                   |
| ----------------------------- | ----------------------------------------- |
| WAPE                          | Scale-independent forecast accuracy       |
| Forecast Bias                 | Detects systematic over/under-forecasting |
| CV (Coefficient of Variation) | Measures demand volatility                |
| Weeks of Cover                | Inventory health indicator                |

> No proprietary or black-box metrics are used.

---

## ⚠️ What PlanPulse Is (and Is Not)

### ✅ What It *Is*

* A **decision-support tool** for demand planning execution
* Fully explainable and auditable
* Designed to support human planners, not replace them

### ❌ What It Is *Not*

* A forecasting engine
* A machine-learning black box
* A network or multi-echelon optimizer
* A replacement for IBP / S&OP processes

> PlanPulse is intentionally scoped to be **directionally correct, transparent, and practical**.

---

## 🧪 Trust & Validation

PlanPulse can be trusted **within its intended scope** because:

* It uses accepted supply chain metrics
* All decisions are explainable
* Uncertainty is explicitly surfaced (confidence levels, lead-time flags)
* Outputs align with how experienced planners reason through trade-offs

This tool is best used to:

> **Prioritize attention, structure discussions, and accelerate decisions — not automate them blindly.**

---

## 🛠 Tech Stack

* **Python**
* **Pandas / NumPy**
* **Matplotlib**
* **Streamlit**

---

## ▶️ How to Run Locally

```bash
# Clone repository
git clone https://github.com/your-username/planpulse.git
cd planpulse

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

---

## 📂 Input Data Requirements

PlanPulse works with most demand planning exports (CSV / XLSX).

**Required:**

* Date / Week / Period
* Demand / Actuals

**Optional (recommended):**

* SKU / Item
* Region / Channel
* Forecast
* On-hand inventory
* Lead time
* Unit cost

---

## 🎯 Learning Objective

This project was built to:

* Bridge theory and real planning decisions
* Learn demand planning by **building the logic planners use**
* Practice translating analytics into business actions

---

## 📣 Feedback Welcome

This is an active learning project.
Feedback from demand planners, supply chain professionals, and hiring managers is highly welcome.

---

### 👩‍💻 Author

**Rutwik Satish**
Graduate Student — Master of Science in Engineering Management
Interested in Demand Planning, Supply Chain Analytics, and Decision Support Systems

---
