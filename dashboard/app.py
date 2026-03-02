import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
import os


sys.path.append(os.path.abspath(".."))

from src.pipeline import run_full_analysis


st.set_page_config(page_title="Battery RUL System", layout="wide")

st.title("🔋 Physics-Informed Battery RUL Prediction")

st.markdown(
"""
This dashboard estimates Remaining Useful Life (RUL) 
using a physics-informed hybrid modeling framework.
"""
)



raw_path = st.text_input(
    "Path to raw battery folder",
    value="data/raw/B0005"
)

run_button = st.button("Run Analysis")



if run_button:

    with st.spinner("Running full analysis..."):
        results = run_full_analysis(raw_path)

    capacities = results["capacities"]
    slope = results["linear_slope"]
    threshold = results["threshold"]
    hybrid_pred = results["hybrid_prediction"]
    eol_linear = results["linear_eol"]
    ci_low = results["eol_ci_low"]
    ci_high = results["eol_ci_high"]

    x = np.arange(len(capacities))

    

    intercept = capacities[0] - slope * 0
    future_x = np.arange(0, int(eol_linear * 1.2))
    future_linear = slope * future_x + intercept

    

    col1, col2 = st.columns([2, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(x, capacities, label="Observed Capacity", linewidth=2)

        ax.plot(
            future_x,
            future_linear,
            label="Linear Forecast",
            linestyle="--"
        )

        ax.plot(
            x,
            hybrid_pred,
            label="Hybrid Model",
            linewidth=2
        )

        ax.axhline(
            threshold,
            linestyle=":",
            label="80% Threshold"
        )

        ax.fill_between(
            [ci_low, ci_high],
            threshold * 0.95,
            threshold * 1.05,
            alpha=0.1,
            label="95% CI (EOL Range)"
        )

        ax.set_xlabel("Cycle")
        ax.set_ylabel("Capacity (Ah)")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

    with col2:
        st.subheader("📊 Results")

        st.metric("Linear EOL (cycles)", f"{eol_linear:.1f}")
        st.metric("95% CI - Lower", f"{ci_low:.1f}")
        st.metric("95% CI - Upper", f"{ci_high:.1f}")

        st.markdown("---")

        st.markdown("### Model Interpretation")

        st.write(
            """
            • Linear model captures global degradation trend  
            • Hybrid model corrects nonlinear residual behavior  
            • Bootstrap provides uncertainty band  
            """
        )

        st.markdown("---")

        st.write("### Technical Summary")

        st.write(f"Slope (Ah/cycle): {slope:.6f}")
        st.write(f"Initial Capacity: {capacities[0]:.4f}")
        st.write(f"Threshold (80%): {threshold:.4f}")
