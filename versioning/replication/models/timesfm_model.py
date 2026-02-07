import numpy as np
import pandas as pd
import timesfm
from replication.evaluation.metrics import weighted_interval_score, compute_point_metrics
from replication.data_loader import build_asof_series, date_to_epiweek

def init_timesfm_model(horizon=4, context_len=128):
    """Initialize TimesFM model."""
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model.compile(
        timesfm.ForecastConfig(
            max_context=context_len,
            max_horizon=horizon,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            fix_quantile_crossing=True,
            infer_is_positive=True,
        )
    )
    return model

def extract_tfm_intervals(quant_fc):
    """
    Build 90/80/70% intervals using off-grid quantiles 0.05/0.95, 0.10/0.90, 0.15/0.85.
    """
    # If first column is mean, drop it and keep q10..q90 anchors
    if quant_fc.shape[1] >= 10:
        qcols = quant_fc[:, 1:]   # q10..q90 (9 columns)
    else:
        qcols = quant_fc          # rare case: only quantiles

    qs_anchor = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90], dtype=np.float32)
    qs_dense = np.array([0.01, 0.025, 0.05, 0.075,
                         0.10, 0.125, 0.15,
                         0.85, 0.875, 0.90, 0.925, 0.95, 0.975, 0.99], dtype=np.float32)

    H = qcols.shape[0]
    q05 = np.empty(H, dtype=np.float32)
    q10 = np.empty(H, dtype=np.float32)
    q15 = np.empty(H, dtype=np.float32)
    q85 = np.empty(H, dtype=np.float32)
    q90 = np.empty(H, dtype=np.float32)
    q95 = np.empty(H, dtype=np.float32)
    q50 = np.empty(H, dtype=np.float32)

    for t in range(H):
        qt = qcols[t, :].astype(np.float32)
        qt = np.maximum.accumulate(qt)
        qt = np.minimum.accumulate(qt[::-1])[::-1]

        q_dense = np.interp(qs_dense, qs_anchor, qt)
        q05[t] = q_dense[2]   # 0.05
        q10[t] = q_dense[4]   # 0.10
        q15[t] = q_dense[6]   # 0.15
        q85[t] = q_dense[7]   # 0.85
        q90[t] = q_dense[8]   # 0.90
        q95[t] = q_dense[11]  # 0.95
        q50[t] = qt[4]        # 0.50 anchor

    # Noncrossing projection
    Q = np.stack([q05, q10, q15, q50, q85, q90, q95], axis=1)
    for t in range(Q.shape[0]):
        Q[t] = np.maximum.accumulate(Q[t])

    lo90, hi90 = Q[:, 0], Q[:, 6]
    lo80, hi80 = Q[:, 1], Q[:, 5]
    lo70, hi70 = Q[:, 2], Q[:, 4]
    median = Q[:, 3]

    intervals = {
        "90": (lo90, hi90, 0.10),
        "80": (lo80, hi80, 0.20),
        "70": (lo70, hi70, 0.30),
    }
    return intervals, median

def horizon_cv_timesfm(series, 
                       dates, 
                       use_versioned: bool = False, 
                       init_window: int =128, 
                       step: int =4, 
                       horizon:int =4):
    """Cross-validation for TimesFM."""
    preds, trues, medians, pdts = [], [], [], []
    lo90_all, hi90_all, lo80_all, hi80_all, lo70_all, hi70_all = [], [], [], [], [], []
    cov90_list, cov80_list, cov70_list = [], [], []
    wid90_list, wid80_list, wid70_list = [], [], []
    wis_multi_list, mae_list, rmse_list, mse_list = [], [], [], []

    model = init_timesfm_model(horizon=horizon, context_len=128)
    
    for start in range(init_window, len(series) - horizon, step):

        if use_versioned: 
            as_of_date = date_to_epiweek(dates[start])
            train = build_asof_series(as_of_date).values.flatten().astype(np.float32)
        else: 
            train = series[:start].astype(np.float32)

        test = series[start:start+horizon].astype(np.float32)
        test_dates = dates[start:start+horizon]

        if len(test) < horizon:
            break

        model.compile(timesfm.ForecastConfig(
            max_context=min(len(train), 512), #type:ignore
            max_horizon=horizon,
            normalize_inputs=True,
            infer_is_positive=True,
            use_continuous_quantile_head=True,
            fix_quantile_crossing=True,
        ))
        
        try:
            point_fc, quant_fc = model.forecast(horizon=horizon, inputs=[train]) #type:ignore
            point_fc = np.nan_to_num(np.array(point_fc[0]).flatten())
            quant_fc = np.nan_to_num(np.array(quant_fc[0]))

            intervals, med_arr = extract_tfm_intervals(quant_fc)

            for j in range(horizon):
                t = test[j]
                lo90, hi90 = intervals["90"][0][j], intervals["90"][1][j]
                lo80, hi80 = intervals["80"][0][j], intervals["80"][1][j]
                lo70, hi70 = intervals["70"][0][j], intervals["70"][1][j]
                med = med_arr[j]
                pred = point_fc[j]

                cov90_list.append((t >= lo90) & (t <= hi90))
                wid90_list.append(hi90 - lo90)
                cov80_list.append((t >= lo80) & (t <= hi80))
                wid80_list.append(hi80 - lo80)
                cov70_list.append((t >= lo70) & (t <= hi70))
                wid70_list.append(hi70 - lo70)

                wis_multi_list.append(weighted_interval_score(
                    t, [lo90, lo80, lo70], [hi90, hi80, hi70], [0.10, 0.20, 0.30], med
                ))

                mae, rmse, mse = compute_point_metrics(np.array([t]), np.array([pred]))
                mae_list.append(mae)
                rmse_list.append(rmse)
                mse_list.append(mse)

                preds.append(pred)
                trues.append(t)
                medians.append(med)
                pdts.append(test_dates[j])
                lo90_all.append(lo90)
                hi90_all.append(hi90)
                lo80_all.append(lo80)
                hi80_all.append(hi80)
                lo70_all.append(lo70)
                hi70_all.append(hi70)
        except Exception as e:
            print(f"TimesFM fold failed: {e}")
            continue

    results = pd.DataFrame({
        "date": pdts, "true": trues, "pred": preds,
        "lo90": lo90_all, "hi90": hi90_all,
        "lo80": lo80_all, "hi80": hi80_all,
        "lo70": lo70_all, "hi70": hi70_all
    })

    avg_cov = [np.mean(cov90_list), np.mean(cov80_list), np.mean(cov70_list)]
    avg_wid = [np.mean(wid90_list), np.mean(wid80_list), np.mean(wid70_list)]
    avg_wis_multi = np.mean(wis_multi_list)
    avg_mae = np.mean(mae_list)
    avg_rmse = np.mean(rmse_list)
    avg_mse = np.mean(mse_list)

    return results, avg_cov, avg_wid, avg_wis_multi, avg_mae, avg_rmse, avg_mse

