import numpy as np

def weighted_interval_score(y_true, lowers, uppers, alphas, median):
    """Calculate weighted interval score."""
    K = len(alphas)
    wis = 0.5 * np.abs(y_true - median)
    for l, u, a in zip(lowers, uppers, alphas):
        score = (u - l)
        if y_true < l:
            score += (2 / a) * (l - y_true)
        elif y_true > u:
            score += (2 / a) * (y_true - u)
        wis += (a / 2) * score
    return wis / (K + 0.5)

def coverage(y_true, lower, upper):
    """Calculate empirical coverage."""
    return np.mean((y_true >= lower) & (y_true <= upper))

def mean_width(lower, upper):
    """Calculate mean interval width."""
    return np.mean(upper - lower)

def compute_point_metrics(y_true, y_pred):
    """Calculate MAE, RMSE, MSE."""
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred)**2)
    rmse = float(np.sqrt(mse))
    return mae, rmse, mse

def cov_width_cols(prefix, cov_vec, wid_vec):
    """Format coverage and width into dictionary columns."""
    return {
        f"{prefix}_cov90": float(cov_vec[0]),
        f"{prefix}_cov80": float(cov_vec[1]),
        f"{prefix}_cov70": float(cov_vec[2]),
        f"{prefix}_wid90": float(wid_vec[0]),
        f"{prefix}_wid80": float(wid_vec[1]),
        f"{prefix}_wid70": float(wid_vec[2]),
    }