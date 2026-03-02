import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from scipy.optimize import curve_fit


# ==========================================================
# DATA LOADING
# ==========================================================

def load_discharge_files(path):
    files = sorted(os.listdir(path))
    valid_files = []

    for f in files:
        try:
            df = pd.read_csv(os.path.join(path, f), encoding="utf-8-sig")
            df.columns = df.columns.str.strip().str.lower()

            required = {"current_measured", "voltage_measured", "time"}
            if required.issubset(df.columns):
                valid_files.append(f)
        except:
            continue

    return valid_files


def compute_capacity(df):
    discharge = df[df["current_measured"] < 0]

    if len(discharge) < 10:
        return np.nan

    time = discharge["time"].values
    current = discharge["current_measured"].values

    dt = np.diff(time)
    current_trim = current[:-1]

    capacity = -np.sum(current_trim * dt) / 3600
    return capacity


def extract_capacities(path):
    files = load_discharge_files(path)
    capacities = []

    for f in files:
        try:
            df = pd.read_csv(os.path.join(path, f), encoding="utf-8-sig")
            df.columns = df.columns.str.strip().str.lower()
            cap = compute_capacity(df)
            capacities.append(cap)
        except:
            capacities.append(np.nan)

    capacities = np.array(capacities)
    capacities = capacities[~np.isnan(capacities)]

    return capacities


# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

def build_features(capacities, window=10):

    df = pd.DataFrame({
        "cycle": np.arange(len(capacities)),
        "capacity": capacities
    })

    df["cap_norm"] = df["capacity"] / df["capacity"].iloc[0]
    df["delta_cap"] = df["capacity"].diff()
    df["rolling_std"] = df["capacity"].rolling(window).std()

    slopes = []
    for i in range(len(df)):
        if i < window:
            slopes.append(np.nan)
        else:
            x_local = np.arange(window)
            y_local = df["capacity"].iloc[i-window:i].values
            s, _ = np.polyfit(x_local, y_local, 1)
            slopes.append(s)

    df["rolling_slope"] = slopes

    df = df.dropna().reset_index(drop=True)

    return df


# ==========================================================
# MODELS
# ==========================================================

def linear_model(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    return slope, intercept, pred


def exponential_func(x, a, b, c):
    return a * np.exp(-b * x) + c


def exponential_model(x, y):
    try:
        popt, _ = curve_fit(
            exponential_func,
            x,
            y,
            maxfev=10000
        )
        pred = exponential_func(x, *popt)
        return popt, pred
    except:
        return None, np.full_like(y, np.nan)


def hybrid_model(x, y, X_features):

    slope, intercept, linear_pred = linear_model(x, y)

    residual = y - linear_pred

    rf = RandomForestRegressor(
        n_estimators=500,
        random_state=42
    )

    rf.fit(X_features, residual)

    hybrid_pred = linear_pred + rf.predict(X_features)

    return hybrid_pred


# ==========================================================
# UNCERTAINTY
# ==========================================================

def bootstrap_eol(x, y, threshold, n_boot=300):

    eols = []

    for _ in range(n_boot):
        idx = np.random.choice(len(x), len(x), replace=True)
        s, i = np.polyfit(x[idx], y[idx], 1)
        k = (threshold - i) / s
        eols.append(abs(k))

    return np.array(eols)


# ==========================================================
# FULL PIPELINE
# ==========================================================

def run_full_analysis(raw_path):

    capacities = extract_capacities(raw_path)

    if len(capacities) < 30:
        raise ValueError("Not enough valid discharge cycles detected.")

    df_feat = build_features(capacities)

    x = df_feat["cycle"].values
    y = df_feat["capacity"].values

    X_features = df_feat[
        ["cap_norm", "delta_cap", "rolling_slope", "rolling_std"]
    ]

    # ==========================
    # Linear Model
    # ==========================

    slope, intercept, linear_pred = linear_model(x, y)
    linear_mae = mean_absolute_error(y, linear_pred)

    threshold = 0.8 * y[0]
    eol_linear = abs((threshold - intercept) / slope)

    # ==========================
    # Exponential Model
    # ==========================

    exp_params, exp_pred = exponential_model(x, y)

    if exp_params is not None:
        exp_mae = mean_absolute_error(y, exp_pred)
    else:
        exp_mae = np.nan

    # ==========================
    # Hybrid Model
    # ==========================

    hybrid_pred = hybrid_model(x, y, X_features)
    hybrid_mae = mean_absolute_error(y, hybrid_pred)

    # ==========================
    # Uncertainty
    # ==========================

    eol_samples = bootstrap_eol(x, y, threshold)

    ci_low = np.percentile(eol_samples, 2.5)
    ci_high = np.percentile(eol_samples, 97.5)

    # ==========================
    # Return Results
    # ==========================

    return {
        "capacities": capacities,

        "linear_slope": slope,
        "linear_intercept": intercept,
        "linear_eol": eol_linear,
        "linear_mae": linear_mae,

        "exponential_params": exp_params,
        "exponential_mae": exp_mae,

        "hybrid_prediction": hybrid_pred,
        "hybrid_mae": hybrid_mae,

        "threshold": threshold,
        "eol_ci_low": ci_low,
        "eol_ci_high": ci_high,
    }