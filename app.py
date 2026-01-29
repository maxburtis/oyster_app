import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.interpolate import UnivariateSpline, PchipInterpolator
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🦪 Oyster Calculator")
st.markdown("**Sampling Protocol & App Guide:**")

with st.expander("Open README (Sampling protocol + how to run the app)", expanded=False):
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            st.markdown(f.read())
    except FileNotFoundError:
        st.warning("README.md not found in the app directory.")

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("Inputs")
CV = st.sidebar.number_input("Weight CV", value=0.35, min_value=0.05, max_value=1.0, step=0.05)
MARKET_WEIGHT = st.sidebar.number_input("Market size (g)", value=66.0, min_value=10.0)
MONTHS_AHEAD = st.sidebar.slider("Months to project", 1, 24, 12)

st.sidebar.header("Model smoothing")
SPLINE_SMOOTH = st.sidebar.slider(
    "Monotone spline smoothness (higher = smoother)",
    min_value=0.0,
    max_value=5.0,
    value=2.0,
    step=0.5,
    help="0 = follow data closely (more bumps). Higher values apply more smoothing."
)

st.sidebar.header("Bagging / Splits")
DENSITY_G_PER_L = st.sidebar.number_input("Density (g/L)", value=1940.85, min_value=100.0)
SPLIT_FACTOR = st.sidebar.number_input("Split factor (x initial volume)", value=2.0, min_value=1.1, step=0.1)
INITIAL_VOLUME_L = st.sidebar.number_input("Initial bag volume (L)", value=3.0, min_value=0.1, step=0.5)


uploaded = st.sidebar.file_uploader("Upload bio CSV", type=["csv"])

# Offer README as a downloadable file
try:
    with open("README.md", "rb") as f:
        st.sidebar.download_button(
            "Download README",
            f,
            file_name="README.md",
            mime="text/markdown",
        )
except FileNotFoundError:
    pass

# -----------------------------
# Sample CSV (matches 2025_bio.csv structure)
# -----------------------------
# Prefer serving a static file if it exists in the repo; otherwise fall back to embedded text.
try:
    with open("sample_oyster_sampling.csv", "rb") as f:
        st.sidebar.download_button(
            "Download sample CSV",
            f,
            file_name="sample_oyster_sampling.csv",
            mime="text/csv",
        )
except FileNotFoundError:
    sample_csv_text = """Bag,Date,Count,weight_g
    1,2025-06-04,200,1995.8048
    35,2025-06-04,200,1773.54472
    27,2025-06-04,200,2213.52896
    31,2025-06-04,200,1945.90968
    36,2025-06-04,200,2118.27464
    37,2025-06-04,200,1940.72744
    1,2025-06-17,197,2598.1132
    35,2025-06-17,200,2165.2192
    27,2025-06-17,199,2544.7408
    31,2025-06-17,200,2392.8668
    36,2025-06-17,198,2735.0712
    37,2025-06-17,199,2591.4032
    1,2025-07-03,198,3368.0376
    35,2025-07-03,197,2687.8152
    27,2025-07-03,197,3224.8744
    31,2025-07-03,198,2898.3888
    36,2025-07-03,198,2975.56352
    37,2025-07-03,199,3201.2928
    1,2025-07-23,195,4960.9952
    35,2025-07-23,194,3904.0064
    27,2025-07-23,196,4572.3168
    31,2025-07-23,196,4123.9328
    36,2025-07-23,197,4271.592
    37,2025-07-23,197,4484.2752
    1,2025-08-05,195,6216.7064
    35,2025-08-05,193,4754.5016
    27,2025-08-05,195,6038.61
    31,2025-08-05,195,5292.3384
    36,2025-08-05,195,5674.8136
    37,2025-08-05,195,6038.5736
    1,2025-09-01,192,8481.0368
    35,2025-09-01,193,6951.5624
    27,2025-09-01,193,8403.5904
    31,2025-09-01,194,7732.9736
    36,2025-09-01,194,8173.1464
    37,2025-09-01,194,8513.9976
    1,2025-09-09,190,10195.632
    35,2025-09-09,193,8685.8824
    27,2025-09-09,191,10362.6288
    31,2025-09-09,192,9571.6416
    36,2025-09-09,192,10027.5168
    37,2025-09-09,192,10402.4784
    1,2025-09-17,190,11867.952
    35,2025-09-17,192,9917.3536
    27,2025-09-17,190,11885.07
    31,2025-09-17,191,10910.7352
    36,2025-09-17,191,11568.903
    37,2025-09-17,191,11950.597
    1,2025-09-15,189,11546.272
    35,2025-09-15,191,9624.168
    27,2025-09-15,189,11478.69024
    31,2025-09-15,191,7724.67176
    36,2025-09-15,190,11207.40744
    37,2025-09-15,190,11632.0812
    1,2025-10-16,185,16948.187
    35,2025-10-16,189,13858.4178
    27,2025-10-16,185,16766.678
    31,2025-10-16,188,15620.7464
    36,2025-10-16,188,16531.407
    37,2025-10-16,188,17081.4104
    1,2025-10-23,183,19190.404
    35,2025-10-23,188,15722.6792
    27,2025-10-23,183,18905.556
    31,2025-10-23,186,17533.6312
    36,2025-10-23,186,18608.3832
    37,2025-10-23,186,19316.6912
    1,2025-10-30,181,21704.904
    35,2025-10-30,186,17777.2712
    27,2025-10-30,181,21444.762
    31,2025-10-30,184,19770.8072
    36,2025-10-30,184,21197.6824
    37,2025-10-30,184,21940.0656
    """
    sample_csv_text = "\n".join([line.strip() for line in sample_csv_text.strip().splitlines()]) + "\n"
    st.sidebar.download_button(
        "Download sample CSV",
        sample_csv_text,
        file_name="sample_oyster_sampling.csv",
        mime="text/csv",
    )

# -----------------------------
# Helpers
# -----------------------------
def frac_ready(mean_g, cv, threshold):
    if mean_g <= 0:
        return np.nan
    sigma = np.sqrt(np.log(1 + cv**2))
    mu = np.log(mean_g) - 0.5 * sigma**2
    z = (np.log(threshold) - mu) / sigma
    return 1 - norm.cdf(z)

def in_window(d, start_md, end_md):
    md = (d.month, d.day)
    if start_md <= end_md:
        return start_md <= md <= end_md
    else:
        return md >= start_md or md <= end_md

# -----------------------------
# Monotone regression helpers (pure numpy isotonic regression)
# -----------------------------
def _pava(y, w=None):
    """
    Pool Adjacent Violators Algorithm for isotonic regression (non-decreasing).
    Returns fitted values with the same length as y.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if w is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(w, dtype=float)

    # Each point starts as its own block
    v = y.copy()
    ww = w.copy()
    start = np.arange(n)

    m = n  # number of active blocks
    i = 0
    while i < m - 1:
        if v[i] <= v[i + 1] + 1e-12:
            i += 1
            continue

        # merge blocks i and i+1
        new_w = ww[i] + ww[i + 1]
        new_v = (v[i] * ww[i] + v[i + 1] * ww[i + 1]) / new_w
        v[i] = new_v
        ww[i] = new_w

        # remove block i+1 by shifting left
        v[i + 1 : m - 1] = v[i + 2 : m]
        ww[i + 1 : m - 1] = ww[i + 2 : m]
        start[i + 1 : m - 1] = start[i + 2 : m]
        m -= 1

        # step back to check for new violations
        i = max(i - 1, 0)

    # expand block values back to original length
    fitted = np.empty(n, dtype=float)
    block_starts = list(start[:m]) + [n]
    for bi in range(m):
        a = block_starts[bi]
        b = block_starts[bi + 1]
        fitted[a:b] = v[bi]
    return fitted

def fit_monotone_log_weight(age_days, log_wt):
    """
    Fits a monotone non-decreasing curve of log(weight) vs age_days and returns a predictor.
    Uses isotonic regression + linear interpolation, with gentle linear extrapolation.
    """
    age = np.asarray(age_days, dtype=float)
    y = np.asarray(log_wt, dtype=float)

    mask = np.isfinite(age) & np.isfinite(y)
    age = age[mask]
    y = y[mask]

    # sort by age
    order = np.argsort(age)
    x = age[order]
    y_sorted = y[order]

    # If duplicate x values exist, average them first (stabilizes isotonic fit)
    ux, inv = np.unique(x, return_inverse=True)
    y_mean = np.zeros_like(ux, dtype=float)
    counts = np.zeros_like(ux, dtype=float)
    for i, k in enumerate(inv):
        y_mean[k] += y_sorted[i]
        counts[k] += 1.0
    y_mean /= np.maximum(counts, 1.0)

    y_fit = _pava(y_mean)

    # slopes for extrapolation
    def _slope_left():
        if ux.size < 2:
            return 0.0
        dx = ux[1] - ux[0]
        return 0.0 if dx == 0 else (y_fit[1] - y_fit[0]) / dx

    def _slope_right():
        if ux.size < 2:
            return 0.0
        dx = ux[-1] - ux[-2]
        return 0.0 if dx == 0 else (y_fit[-1] - y_fit[-2]) / dx

    m_left = _slope_left()
    m_right = _slope_right()

    def predict(x_new):
        x_new = np.asarray(x_new, dtype=float).reshape(-1)
        y_pred = np.interp(x_new, ux, y_fit)
        # linear extrapolation outside bounds
        left_mask = x_new < ux[0]
        right_mask = x_new > ux[-1]
        if np.any(left_mask):
            y_pred[left_mask] = y_fit[0] + m_left * (x_new[left_mask] - ux[0])
        if np.any(right_mask):
            y_pred[right_mask] = y_fit[-1] + m_right * (x_new[right_mask] - ux[-1])
        return y_pred

    return predict

# -----------------------------
# Regularized monotone spline helper
# -----------------------------
def fit_regularized_monotone_spline_log_weight(age_days, log_wt, smoothness=2.0):
    """
    Regularized *monotone* spline for log(weight) vs age.
    Approach:
      1) Average duplicates at same age
      2) Enforce monotonicity with isotonic regression (PAVA)
      3) Fit a smoothing spline (UnivariateSpline) to the monotone targets
      4) Enforce monotonicity again by sampling the spline on a grid + PAVA
      5) Use PCHIP over the monotone spline samples for a smooth monotone predictor
    This yields a smooth, monotone curve without pyGAM.
    """
    age = np.asarray(age_days, dtype=float)
    y = np.asarray(log_wt, dtype=float)

    mask = np.isfinite(age) & np.isfinite(y)
    age = age[mask]
    y = y[mask]
    if age.size < 2:
        # degenerate case
        const = float(np.nanmean(y)) if age.size else 0.0
        return lambda x_new: np.full_like(np.asarray(x_new, dtype=float), const, dtype=float)

    # sort by age
    order = np.argsort(age)
    x = age[order]
    y_sorted = y[order]

    # average duplicates at same age
    ux, inv = np.unique(x, return_inverse=True)
    y_mean = np.zeros_like(ux, dtype=float)
    counts = np.zeros_like(ux, dtype=float)
    for i, k in enumerate(inv):
        y_mean[k] += y_sorted[i]
        counts[k] += 1.0
    y_mean /= np.maximum(counts, 1.0)

    # initial monotone target
    y_iso = _pava(y_mean)

    # smoothing spline: choose s based on smoothness slider and scale of data
    # s is roughly the allowed sum of squared residuals; scale it with n and smoothness^2
    n = ux.size
    base_s = max(n * 0.001, 1e-6)
    s = base_s * (smoothness ** 2) * n

    # Fit spline to monotone target; k=3 cubic
    spline = UnivariateSpline(ux, y_iso, s=s, k=min(3, max(1, n - 1)))

    # sample spline on a fine grid and re-enforce monotonicity
    grid_n = int(max(200, 20 * n))
    xg = np.linspace(float(ux.min()), float(ux.max()), grid_n)
    yg = spline(xg)
    yg_mono = _pava(yg)

    # Smooth monotone interpolator (shape-preserving)
    pchip = PchipInterpolator(xg, yg_mono, extrapolate=False)

    def predict(x_new):
        x_new = np.asarray(x_new, dtype=float).reshape(-1)

        # In-range interpolation (PCHIP preserves monotonicity for monotone data)
        y_pred = pchip(np.clip(x_new, xg[0], xg[-1]))

        # Monotone linear extrapolation outside bounds (force non-negative slope)
        if xg.size >= 2:
            m_left = max(0.0, (yg_mono[1] - yg_mono[0]) / (xg[1] - xg[0] + 1e-12))
            m_right = max(0.0, (yg_mono[-1] - yg_mono[-2]) / (xg[-1] - xg[-2] + 1e-12))
        else:
            m_left = 0.0
            m_right = 0.0

        left_mask = x_new < xg[0]
        right_mask = x_new > xg[-1]

        if np.any(left_mask):
            y_pred[left_mask] = yg_mono[0] + m_left * (x_new[left_mask] - xg[0])

        if np.any(right_mask):
            y_pred[right_mask] = yg_mono[-1] + m_right * (x_new[right_mask] - xg[-1])

        return y_pred

    return predict

# -----------------------------
# Main logic
# -----------------------------
if uploaded:
    # -----------------------------
    # Robust CSV load + normalization
    # -----------------------------
    dd = pd.read_csv(uploaded)
    dd.columns = dd.columns.map(lambda c: str(c).replace("\ufeff", ""))

    # Normalize column names: strip whitespace, lower-case, replace spaces with underscores
    dd.columns = (
        dd.columns.astype(str)
          .str.strip()
          .str.replace(r"\s+", "_", regex=True)
          .str.lower()
    )

    # Drop common blank columns from Excel exports (e.g., "Unnamed: 4")
    dd = dd.loc[:, ~dd.columns.astype(str).str.startswith("unnamed")].copy()

    # Map common variants to expected names
    rename_map = {
        "date": "Date",
        "sample_date": "Date",
        "sampling_date": "Date",
        "bag": "Bag",
        "bag_id": "Bag",
        "bagid": "Bag",
        "count": "Count",
        "n": "Count",
        "num": "Count",
        "number": "Count",
        "weight_g": "weight_g",
        "weight": "weight_g",
        "total_weight_g": "weight_g",
        "biomass_g": "weight_g",
        "avg_weight_g": "Avg_Weight_g",
        "average_weight_g": "Avg_Weight_g",
        "avg_wt_g": "Avg_Weight_g",
        "mean_weight_g": "Avg_Weight_g",
    }
    dd = dd.rename(columns={c: rename_map[c] for c in dd.columns if c in rename_map})

    # Validate required columns before sorting/processing
    required_base = {"Date", "Bag"}
    missing_base = sorted(required_base - set(dd.columns))
    if missing_base:
        st.error(
            "Your CSV is missing required column(s): "
            + ", ".join(missing_base)
            + "\n\nFound columns: "
            + ", ".join(list(dd.columns))
        )
        st.stop()

    # Parse Date safely
    dd["Date"] = pd.to_datetime(dd["Date"], errors="coerce")
    if dd["Date"].isna().all():
        st.error("Could not parse any dates in the 'Date' column. Please use a recognizable date format (e.g., 2025-06-04).")
        st.stop()

    # Drop rows without Date/Bag
    dd = dd.dropna(subset=["Date", "Bag"]).copy()

    # Ensure Bag is a clean string key
    dd["Bag"] = dd["Bag"].astype(str).str.strip()

    # Clean numeric columns (tolerate a few missing values)
    if "Count" in dd.columns:
        dd["Count"] = pd.to_numeric(dd["Count"], errors="coerce")
        # If a couple rows are missing Count, forward/back fill within each bag
        dd["Count"] = dd.groupby("Bag")["Count"].ffill().bfill()

    if "weight_g" in dd.columns:
        dd["weight_g"] = pd.to_numeric(dd["weight_g"], errors="coerce")

    dd = dd.sort_values(["Bag", "Date"]).reset_index(drop=True)

    st.caption("Expected columns: Date, Bag, and either Avg_Weight_g or (weight_g + Count). Column names are case/space-insensitive.")

    # Ensure Avg_Weight_g
    if "Avg_Weight_g" not in dd.columns:
        if {"weight_g","Count"}.issubset(dd.columns):
            dd["Avg_Weight_g"] = dd["weight_g"] / dd["Count"]
        else:
            st.error("CSV must contain Avg_Weight_g OR (weight_g + Count)")
            st.stop()

    # Age
    if "Age_days" not in dd.columns:
        dd["Age_days"] = (
            dd["Date"] - dd.groupby("Bag")["Date"].transform("min")
        ).dt.days.astype(float)

    dd = dd.dropna(subset=["Age_days","Avg_Weight_g"]).copy()
    dd["log_wt"] = np.log(dd["Avg_Weight_g"])

    # -----------------------------
    # Fit regularized monotone spline (smooth + monotone)
    # -----------------------------
    predict_log_wt = fit_regularized_monotone_spline_log_weight(
        dd["Age_days"].values,
        dd["log_wt"].values,
        smoothness=float(SPLINE_SMOOTH),
    )

    # -----------------------------
    # Build projection timeline (per-bag) + split schedule
    # -----------------------------
    dd["Date"] = pd.to_datetime(dd["Date"]).dt.normalize()

    # Observed bounds (global)
    obs_start = dd["Date"].min()
    obs_end   = dd["Date"].max()

    end_date = (obs_end + pd.Timedelta(days=30 * MONTHS_AHEAD)).normalize()

    # Use last observed Count per bag for biomass/volume projections
    if "Count" in dd.columns:
        last_counts = dd.groupby("Bag")["Count"].last()
    else:
        last_counts = None

    proj_bags = []
    for bag, g in dd.groupby("Bag"):
        g = g.sort_values("Date")
        b_start = g["Date"].min().normalize()
        b_end   = g["Date"].max().normalize()

        # per-bag observed month/day window
        start_md = (b_start.month, b_start.day)
        end_md   = (b_end.month, b_end.day)

        dates = pd.date_range(b_start, end_date, freq="D")
        growth = np.array([in_window(d, start_md, end_md) for d in dates], dtype=int)
        age_growing = np.cumsum(growth).astype(float)

        tmp = pd.DataFrame({
            "Bag": bag,
            "Date": dates,
            "GrowthDay": growth.astype(bool),
            "Age_growing_days": age_growing.astype(float),
        })

        tmp["pred_wt_g"] = np.exp(predict_log_wt(tmp["Age_growing_days"].values))
        tmp["pct_ready"] = tmp["pred_wt_g"].apply(lambda x: frac_ready(x, CV, MARKET_WEIGHT)) * 100

        if last_counts is not None:
            tmp["Count"] = float(last_counts.loc[bag])
            tmp["biomass_g"] = tmp["Count"] * tmp["pred_wt_g"]
            tmp["volume_L"] = tmp["biomass_g"] / float(DENSITY_G_PER_L)
        else:
            tmp["Count"] = np.nan
            tmp["biomass_g"] = np.nan
            tmp["volume_L"] = np.nan

        proj_bags.append(tmp)

    proj_all = pd.concat(proj_bags, ignore_index=True)

    # A global (mean-across-bags) series for plotting
    proj = (
        proj_all.groupby("Date", as_index=False)
                .agg(pred_wt_g=("pred_wt_g", "mean"),
                     pct_ready=("pct_ready", "mean"),
                     volume_L=("volume_L", "mean"))
    )

    # ---- Split schedule simulation (per bag) ----
    splits = []
    if last_counts is not None:
        for bag, g in proj_all.groupby("Bag"):
            g = g.sort_values("Date")
            split_n = 0

            for _, row in g.iterrows():
                crop_vol = row["volume_L"]
                if not np.isfinite(crop_vol):
                    continue

                # If each split doubles bag count, volume per bag halves
                vol_per_bag = crop_vol / (2 ** split_n)

                if vol_per_bag >= float(SPLIT_FACTOR) * float(INITIAL_VOLUME_L):
                    split_n += 1
                    splits.append({
                        "Bag": bag,
                        "Split #": split_n,
                        "Date": row["Date"],
                        "Vol_per_bag_L": vol_per_bag,
                    })

    splits_df = pd.DataFrame(splits)

    # -----------------------------
    # Plots
    # -----------------------------
    # --- stacked plots (top/middle/bottom) ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Plot 1: mean weight over time
    ax1.scatter(dd["Date"], dd["Avg_Weight_g"], alpha=0.7, label="Observed (all bags)")
    ax1.plot(proj["Date"], proj["pred_wt_g"], linewidth=2, label="Predicted (mean across bags)")
    ax1.set_ylabel("Mean weight (g)")
    ax1.set_title("Mean oyster weight over time")
    ax1.grid(alpha=0.3)
    ax1.legend()

    # Plot 2: % at/above market
    ax2.plot(proj["Date"], proj["pct_ready"], linewidth=2, label="% at/above market (mean across bags)")
    ax2.axhline(50, linestyle="--", alpha=0.5)
    ax2.axhline(90, linestyle="--", alpha=0.5)
    ax2.set_ylabel("% at market")
    ax2.set_ylim(0, 100)
    ax2.set_title("% of crop ready for market")
    ax2.grid(alpha=0.3)
    ax2.legend()

    # Plot 3: projected volume + split threshold + split events
    if "volume_L" in proj.columns and proj["volume_L"].notna().any():
        ax3.plot(proj["Date"], proj["volume_L"], linewidth=2, label="Projected crop volume (L, mean across bags)")
        ax3.axhline(float(SPLIT_FACTOR) * float(INITIAL_VOLUME_L), linestyle="--", alpha=0.7,
                    label="Split threshold (x initial volume)")
        if not splits_df.empty:
            ax3.scatter(splits_df["Date"], splits_df["Vol_per_bag_L"], alpha=0.9, label="Split events (per-bag volume)")
        ax3.set_ylabel("Volume (L)")
        ax3.set_title("Bag volume & split schedule")
        ax3.grid(alpha=0.3)
        ax3.legend()
    else:
        ax3.text(0.01, 0.5, "No Count column provided; volume/split schedule disabled.", transform=ax3.transAxes)
        ax3.set_title("Bag volume & split schedule")

    ax3.set_xlabel("Date")

    st.caption("Tip: If the predicted curve looks bumpy, increase 'Monotone spline smoothness' in the sidebar.")
    st.pyplot(fig)


    # -----------------------------
    # Download
    # -----------------------------
    st.download_button(
        "Download forecast CSV",
        proj_all.to_csv(index=False),
        "oyster_market_forecast.csv",
        "text/csv"
    )


else:
    st.info("Upload a bio CSV to begin.")

# -----------------------------
# Footer credits
# -----------------------------
st.markdown(
    """
---
**Credits:**  
This work was funded by a **SARE (Sustainable Agriculture Research & Education)** grant.
"""
)
