import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from replication.config import OUTPUT_DIR, STATES, INIT_WINDOW, STEP, HORIZON
from replication.data_loader import load_ili_data
from replication.models.timesfm_model import horizon_cv_timesfm
from replication.models.arima_model import select_best_arima
from replication.visualization.plots import (
    plot_cv_overlay_with_intervals,
    plot_coverage_width_bars,
    boxplot_metrics,
)
from replication.evaluation.metrics import cov_width_cols
from replication.versioning import generate_versioning_data


DATA_MODES = {
    "VERSIONED": True,
    "FINALIZED": False
}

# Define color palette and hue order for consistency
HUE_ORDER = ["TimesFM", "ARIMA"]
PALETTE = {"TimesFM": "#1f77b4", "ARIMA": "#ff7f0e"}


def main():
    """
    Compare TimesFM vs ARIMA under finalized vs versioned data.
    Outputs:
      - model_comparison_results.csv
      - model_comparison_plots.pdf
    """
    summary_rows = []

    pdf_path = OUTPUT_DIR / "model_comparison_plots.pdf"
    pdf = PdfPages(pdf_path)

    for state in STATES:
        print(f"\n{'='*22}\nProcessing {state.upper()}\n{'='*22}")

        # Generate versioning data for this state (overwrites previous state's data)
        # This only needs to run once per state, before any versioned CV
        print(f"Generating versioning data for {state.upper()}...")
        generate_versioning_data(
            region=state,
            start_epiweek=201650,
            end_epiweek=202352,
            init_train=INIT_WINDOW,
            api_sleep=1.0
        )
        print(f"Versioning data generation complete for {state.upper()}")

        try:
            df = load_ili_data(state)
            series = df["wili"].to_numpy()
            dates = df.index

            for mode_label, use_versioned in DATA_MODES.items():
                print(f"\n--- {state.upper()} | {mode_label} ---")

                # ---------------- TimesFM ----------------
                (
                    tfm_results,
                    tfm_cov,
                    tfm_width,
                    tfm_wis,
                    tfm_mae,
                    tfm_rmse,
                    tfm_mse,
                ) = horizon_cv_timesfm(
                    series,
                    dates,
                    use_versioned=use_versioned,
                    init_window=INIT_WINDOW,
                    step=STEP,
                    horizon=HORIZON,
                )

                # ---------------- ARIMA ----------------
                best_order, best_stats = select_best_arima(
                    series,
                    dates,
                    use_versioned=use_versioned,
                    init_window=INIT_WINDOW,
                    step=STEP,
                    horizon=HORIZON,
                )
                ar_results, ar_cov, ar_width, ar_wis, ar_mae, ar_rmse, ar_mse = best_stats

                print(
                    f"Best ARIMA{best_order} | "
                    f"MAE: {ar_mae:.4f} | RMSE: {ar_rmse:.4f} | "
                    f"MSE: {ar_mse:.4f} | WIS: {ar_wis:.4f}"
                )
                print(
                    f"TimesFM              | "
                    f"MAE: {tfm_mae:.4f} | RMSE: {tfm_rmse:.4f} | "
                    f"MSE: {tfm_mse:.4f} | WIS: {tfm_wis:.4f}"
                )

                # ---------------- Plots ----------------
                fig1 = plot_cv_overlay_with_intervals(
                    df,
                    tfm_results,
                    ar_results,
                    state,
                    title_prefix=mode_label,
                    horizon=HORIZON,
                )
                pdf.savefig(fig1)
                plt.close(fig1)

                fig2 = plot_coverage_width_bars(
                    [tfm_cov, ar_cov],
                    [tfm_width, ar_width],
                    ["TimesFM", f"ARIMA{best_order}"],
                    state,
                    title_prefix=mode_label,
                )
                pdf.savefig(fig2)
                plt.close(fig2)

                # ---------------- Metrics rows ----------------
                tfm_row = {
                    "region": state,
                    "model": "TimesFM",
                    "data_mode": mode_label.lower(),
                    "order": None,
                    "mae": float(tfm_mae),
                    "rmse": float(tfm_rmse),
                    "mse": float(tfm_mse),
                    "wis": float(tfm_wis),
                }
                tfm_row.update(cov_width_cols("int", tfm_cov, tfm_width))

                ar_row = {
                    "region": state,
                    "model": "ARIMA",
                    "data_mode": mode_label.lower(),
                    "order": str(best_order),
                    "mae": float(ar_mae),
                    "rmse": float(ar_rmse),
                    "mse": float(ar_mse),
                    "wis": float(ar_wis),
                }
                ar_row.update(cov_width_cols("int", ar_cov, ar_width))

                summary_rows.append(tfm_row)
                summary_rows.append(ar_row)

        except Exception as e:
            print(f"Error running {state}: {e}")
            import traceback
            traceback.print_exc()

    # Save summary metrics to CSV
    summary_df = pd.DataFrame(summary_rows)
    output_path = OUTPUT_DIR / "model_comparison_results.csv"
    summary_df.to_csv(output_path, index=False)
    
    print(f"\nSaved metrics to: {output_path}")

    # ---------------- Boxplot Summary Visualizations ----------------
    print("\n=== Generating Summary Boxplots ===")
    
    # Prepare data for boxplots
    metrics_df = summary_df.sort_values(["region", "model"]).reset_index(drop=True)
    print("\n=== Metrics Summary (first 5 rows) ===")
    print(metrics_df.head())
    
    metric_cols = [
        "wis", "mae", "rmse", "mse",
        "int_cov90", "int_cov80", "int_cov70",
        "int_wid90", "int_wid80", "int_wid70"
    ]
    
    plot_df = (
        metrics_df
        .melt(
            id_vars=["region", "model", "order", "data_mode"],
            value_vars=metric_cols,
            var_name="metric",
            value_name="value"
        )
    )
    
    # Generate boxplots and compute summary statistics for each data mode
    summary_stats_rows = []
    
    for mode in ["versioned", "finalized"]:
        mode_df = plot_df[plot_df["data_mode"] == mode]
        mode_title = mode.upper()
        
        # Compute summary statistics (mean, median, std) for each metric and model
        for model in HUE_ORDER:
            model_data = metrics_df[(metrics_df["model"] == model) & (metrics_df["data_mode"] == mode)]
            
            for metric in metric_cols:
                values = model_data[metric].dropna()
                summary_stats_rows.append({
                    "data_mode": mode,
                    "model": model,
                    "metric": metric,
                    "mean": values.mean(),
                    "median": values.median(),
                    "std": values.std(),
                    "min": values.min(),
                    "max": values.max(),
                    "count": len(values)
                })
        
        # 1) Point and probabilistic scores
        fig3 = boxplot_metrics(
            mode_df,
            ["wis", "mae", "rmse", "mse"],
            f"Point and Probabilistic Scores - {mode_title}",
            hue_order=HUE_ORDER,
            palette=PALETTE
        )
        pdf.savefig(fig3)
        plt.close(fig3)
        
        # 2) Coverage metrics
        fig4 = boxplot_metrics(
            mode_df,
            ["int_cov90", "int_cov80", "int_cov70"],
            f"Empirical Coverage Across Regions - {mode_title}",
            hue_order=HUE_ORDER,
            palette=PALETTE
        )
        pdf.savefig(fig4)
        plt.close(fig4)
        
        # 3) Interval widths
        fig5 = boxplot_metrics(
            mode_df,
            ["int_wid90", "int_wid80", "int_wid70"],
            f"Prediction Interval Widths Across Regions - {mode_title}",
            hue_order=HUE_ORDER,
            palette=PALETTE
        )
        pdf.savefig(fig5)
        plt.close(fig5)

    pdf.close()
    
    # Save summary statistics to CSV
    summary_stats_df = pd.DataFrame(summary_stats_rows)
    summary_stats_path = OUTPUT_DIR / "model_comparison_summary_stats.csv"
    summary_stats_df.to_csv(summary_stats_path, index=False)
    
    print(f"Saved plots to:          {pdf_path}")
    print(f"Saved summary stats to:  {summary_stats_path}")

    return summary_df


if __name__ == "__main__":
    summary_df = main()