import matplotlib.pyplot as plt
import numpy as np
from replication.config import C0, C1
import seaborn as sns


def plot_coverage_width_bars(
    avg_covs,
    avg_wids,
    models,
    region,
    title_prefix: str = "",
):
    levels = ["90", "80", "70"]
    x = np.arange(len(levels))
    w = 0.35

    fig = plt.figure(figsize=(10, 4))

    prefix = f"{title_prefix} " if title_prefix else ""

    # ---------------- Coverage ----------------
    ax1 = plt.subplot(1, 2, 1)
    for i, covs in enumerate(avg_covs):
        color = C0 if models[i].startswith("TimesFM") else C1
        ax1.bar(x + w * i - w / 2, covs, width=w, label=models[i], color=color)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{L}%" for L in levels])
    ax1.axhline(0.9, color="gray", linestyle="--", linewidth=1)
    ax1.axhline(0.8, color="gray", linestyle="--", linewidth=1)
    ax1.axhline(0.7, color="gray", linestyle="--", linewidth=1)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Empirical Coverage")
    ax1.set_title(f"{prefix}{region.upper()} Coverage")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.4)

    # ---------------- Width ----------------
    ax2 = plt.subplot(1, 2, 2)
    for i, wids in enumerate(avg_wids):
        color = C0 if models[i].startswith("TimesFM") else C1
        ax2.bar(x + w * i - w / 2, wids, width=w, label=models[i], color=color)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{L}%" for L in levels])
    ax2.set_ylabel("Mean Interval Width")
    ax2.set_title(f"{prefix}{region.upper()} Width")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    return fig


def plot_cv_overlay_with_intervals(
    df,
    tfm_results,
    ar_results,
    region,
    title_prefix: str = "",
    horizon: int = 4,
):
    fig = plt.figure(figsize=(14, 6))

    prefix = f"{title_prefix} " if title_prefix else ""

    plt.plot(
        df.index,
        df["wili"],
        color="C0",
        linewidth=2.0,
        label="Actual",
    )
    plt.plot(
        tfm_results["date"],
        tfm_results["pred"],
        color=C0,
        linestyle="--",
        linewidth=2,
        label="TimesFM (CV)",
    )
    plt.plot(
        ar_results["date"],
        ar_results["pred"],
        color=C1,
        linestyle="-.",
        linewidth=2,
        label="ARIMA (CV)",
    )

    plt.fill_between(
        ar_results["date"],
        ar_results["lo90"],
        ar_results["hi90"],
        color=C1,
        alpha=0.12,
        label="ARIMA 90%",
    )
    plt.fill_between(
        tfm_results["date"],
        tfm_results["lo90"],
        tfm_results["hi90"],
        color=C0,
        alpha=0.17,
        label="TimesFM 90%",
    )

    plt.title(
        f"{prefix}{region.upper()} Expanding-Window CV (h={horizon}): "
        "TimesFM vs ARIMA (best order)"
    )
    plt.xlabel("Week")
    plt.ylabel("Weighted ILI (%)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    return fig

def boxplot_metrics(plot_df, subset_metrics, title, hue_order=None, palette=None):
    data = plot_df[plot_df["metric"].isin(subset_metrics)]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.boxplot(
        data=data,
        x="metric",
        y="value",
        hue="model",
        hue_order=hue_order,
        palette=palette,
        showfliers=False,
        ax=ax
    )
    
    ax.set_title(title)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.legend(title="Model")
    ax.grid(True, linestyle="--", alpha=0.3)
    
    plt.tight_layout()
    
    return fig