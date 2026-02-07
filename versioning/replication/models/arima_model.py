import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from replication.evaluation.metrics import weighted_interval_score
from replication.data_loader import build_asof_series, date_to_epiweek

def horizon_cv_arima_order(series: pd.DataFrame, 
                           dates, 
                           order, 
                           use_versioned: bool = False, 
                           init_window=128, step=4, horizon=4):
    """Cross-validation for ARIMA with specific order."""
    preds, trues, medians, pdts = [], [], [], []
    lo90_all, hi90_all, lo80_all, hi80_all, lo70_all, hi70_all = [], [], [], [], [], []
    cov90_list, wid90_list, cov80_list, wid80_list, cov70_list, wid70_list = [], [], [], [], [], []
    wis_multi_list, mae_list, rmse_list, mse_list = [], [], [], []

    for start in range(init_window, len(series) - horizon, step):

        if use_versioned: 
            as_of_date=date_to_epiweek(dates[start])
            y_train = build_asof_series(as_of_date)
        else: 
            y_train = series[:start].astype(np.float64)


        y_test = series[start:start + horizon].astype(np.float64)
        test_dates = dates[start:start + horizon]
        if len(y_test) < horizon:
            break
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                res = ARIMA(y_train, order=order).fit()
                fc = res.get_forecast(steps=horizon)
                mean = np.asarray(fc.predicted_mean)
                ci90 = fc.conf_int(alpha=0.10)
                ci80 = fc.conf_int(alpha=0.20)
                ci70 = fc.conf_int(alpha=0.30)

            def to_lo_hi(ci):
                if hasattr(ci, "iloc"):
                    return ci.iloc[:, 0].to_numpy(), ci.iloc[:, 1].to_numpy()
                return ci[:, 0], ci[:, 1]

            lo90, hi90 = to_lo_hi(ci90)
            lo80, hi80 = to_lo_hi(ci80)
            lo70, hi70 = to_lo_hi(ci70)

            h = min(horizon, len(mean))
            for j in range(h):
                t = y_test[j]
                med = mean[j]
                l90, h90_ = lo90[j], hi90[j]
                l80, h80_ = lo80[j], hi80[j]
                l70, h70_ = lo70[j], hi70[j]

                cov90_list.append((t >= l90) & (t <= h90_))
                wid90_list.append(h90_ - l90)
                cov80_list.append((t >= l80) & (t <= h80_))
                wid80_list.append(h80_ - l80)
                cov70_list.append((t >= l70) & (t <= h70_))
                wid70_list.append(h70_ - l70)

                wis_multi_list.append(
                    weighted_interval_score(
                        t, [l90, l80, l70], [h90_, h80_, h70_], [0.10, 0.20, 0.30], med
                    )
                )

                ae = np.abs(t - med)
                se = (t - med) ** 2
                mae_list.append(ae)
                rmse_list.append(np.sqrt(se))
                mse_list.append(se)

                preds.append(med)
                trues.append(t)
                medians.append(med)
                pdts.append(test_dates[j])
                lo90_all.append(l90)
                hi90_all.append(h90_)
                lo80_all.append(l80)
                hi80_all.append(h80_)
                lo70_all.append(l70)
                hi70_all.append(h70_)
        except Exception as e:
            print(f"ARIMA {order} fold failed: {e}")
            continue

    results = pd.DataFrame({
        "date": pdts, "true": trues, "pred": preds,
        "lo90": lo90_all, "hi90": hi90_all,
        "lo80": lo80_all, "hi80": hi80_all,
        "lo70": lo70_all, "hi70": hi70_all
    })
    avg_cov = [float(np.mean(cov90_list)), float(np.mean(cov80_list)), float(np.mean(cov70_list))]
    avg_wid = [float(np.mean(wid90_list)), float(np.mean(wid80_list)), float(np.mean(wid70_list))]
    avg_wis_multi = float(np.mean(wis_multi_list)) if len(wis_multi_list) else np.inf
    avg_mae = float(np.mean(mae_list)) if len(mae_list) else np.inf
    avg_rmse = float(np.mean(rmse_list)) if len(rmse_list) else np.inf
    avg_mse = float(np.mean(mse_list)) if len(mse_list) else np.inf
    return results, avg_cov, avg_wid, avg_wis_multi, avg_mae, avg_rmse, avg_mse

def select_best_arima(series, dates, use_versioned, init_window=128, step=4, horizon=4):
    """Select best ARIMA order from candidates."""
    orders = [(1,1,0), (2,1,0)]
    outcomes = {}
    for order in orders:
        outcomes[order] = horizon_cv_arima_order(series, dates, order, use_versioned, init_window, step, horizon)

    def score_tuple(stats):
        _, _, _, wis, mae, rmse, _ = stats
        return (wis, rmse, mae)

    best = min(orders, key=lambda o: score_tuple(outcomes[o]))
    return best, outcomes[best]