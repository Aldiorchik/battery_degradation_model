# 🔋 Physics-Informed & Uncertainty-Aware Battery RUL Prediction

A modular lithium-ion battery degradation modeling framework combining physics-based methods and machine learning for robust Remaining Useful Life (RUL) estimation.

---

## 🚀 Project Overview

Lithium-ion batteries degrade over repeated charge–discharge cycles.  
A critical industrial metric is **Remaining Useful Life (RUL)** — the number of cycles remaining before capacity drops below 80% of its initial value.

This project implements:

- Physics-based degradation modeling
- Hybrid residual learning (ML-enhanced)
- Bootstrap-based uncertainty estimation
- Modular analysis pipeline
- Interactive Streamlit dashboard

Applications:
- Electric Vehicles (EV)
- Battery Management Systems (BMS)
- Grid storage
- Predictive maintenance

---

## 📊 Example Output (NASA B0005)

Linear EOL: 278 cycles  
95% CI: [244, 312] cycles  
Hybrid MAE improvement: ~18%

Observations:
- Degradation is near-linear in early cycles.
- Long-term extrapolation increases uncertainty.
- Hybrid modeling improves residual correction while maintaining interpretability.

---

## 🏗 Project Structure

battery-degradation-ml/
│
├── dashboard/
│   └── app.py
│
├── data/
│   └── raw/
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/
│   └── pipeline.py
│
├── requirements.txt
└── README.md

Core design principle:

One clean pipeline function handles full analysis:

from src.pipeline import run_full_analysis

results = run_full_analysis("data/raw/B0005")

---

## 🔬 Methodology

1️⃣ Capacity Extraction  
Discharge cycles are automatically detected.  
Capacity is computed via numerical integration:

Q = -∫ I dt

Only discharge (negative current) segments are used.

2️⃣ Physics-Based Degradation Model  
A linear degradation model is fitted:

capacity(k) = slope × k + intercept

End-of-Life (EOL) is derived analytically when:

capacity = 0.8 × initial_capacity

This ensures interpretability and stability.

3️⃣ Feature Engineering  

- Normalized capacity  
- Capacity delta  
- Rolling degradation slope  
- Rolling variance  

4️⃣ Hybrid Residual Learning  

capacity = linear_model + ML_residual

RandomForestRegressor is used to model nonlinear residual behavior.

Why hybrid?

- Pure ML overfits small windows.
- Pure physics ignores nonlinear effects.
- Hybrid balances robustness and flexibility.

5️⃣ Uncertainty Estimation  

Bootstrap resampling provides:

- EOL distribution
- 95% confidence interval

Uncertainty increases with prediction horizon — consistent with real systems.

---

## 📈 What Makes This Different?

This is NOT:

- A Kaggle regression demo
- A pure black-box model
- A curve-fitting notebook

This is:

✔ Physics-informed  
✔ Interpretable  
✔ Modular  
✔ Reproducible  
✔ Uncertainty-aware  

---

## 🖥 Dashboard

Launch:

python -m streamlit run dashboard/app.py

Dashboard features:

- Degradation curve visualization
- Linear vs Hybrid comparison
- 80% threshold indicator
- 95% EOL confidence band
- Summary metrics

---

## ⚙ Installation

Create environment:

python -m venv venv  
venv\Scripts\activate  

Install dependencies:

python -m pip install -r requirements.txt  

---

## ▶ Run Programmatically

from src.pipeline import run_full_analysis

results = run_full_analysis("data/raw/B0005")

print("Linear EOL:", results["linear_eol"])
print("95% CI:", results["eol_ci_low"], results["eol_ci_high"])

---

## 📌 Key Insights

- Early degradation trends are approximately linear.
- Exponential fits are unstable on short segments.
- Hybrid modeling reduces MAE.
- Long-horizon RUL prediction is inherently probabilistic.
- Bootstrap-based CI reflects uncertainty growth realistically.

---

## 🔮 Future Improvements

- Impedance-based features
- dQ/dV electrochemical indicators
- Temperature-aware modeling
- Bayesian RUL estimation
- Multi-battery validation
- Real-time BMS integration

---

## 📚 Dataset

NASA Battery Dataset 





