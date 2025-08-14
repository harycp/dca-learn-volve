import pandas as pd
import numpy as np
from scipy.optimize import curve_fit


def hyperbolic(t, qi, di, b):
    return qi / (np.abs((1 + b * di * t)) ** (1 / b))


def exponential(t, qi, di):
    return qi * np.exp(-di * t)


def harmonic(t, qi, di):
    return qi / (1 + di * t)


def exponential_fitting(T, Q):
    T_normalized = T / max(T)
    Q_normalized = Q / max(Q)

    params, _ = curve_fit(exponential, T_normalized, Q_normalized)
    qi, di = params

    qi = qi * max(Q)
    di = di / max(T)

    return {"qi": qi, "di": di, "b": 0}


def harmonic_fitting(T, Q):
    T_normalized = T / max(T)
    Q_normalized = Q / max(Q)

    params, _ = curve_fit(harmonic, T_normalized, Q_normalized)
    qi, di = params

    qi = qi * max(Q)
    di = di / max(T)

    return {"qi": qi, "di": di, "b": 1}


def hyperbolic_fitting(T, Q):
    T_normalized = T / max(T)
    Q_normalized = Q / max(Q)

    params, _ = curve_fit(hyperbolic, T_normalized, Q_normalized)
    qi, di, b = params

    qi = qi * max(Q)
    di = di / max(T)

    return {"qi": qi, "di": di, "b": b}


def error_function(original_data, model_date):
    n = len(original_data)
    RMSE = np.sqrt((1/n) * np.square(original_data - model_date).sum())
    return RMSE


def get_days(df, col_name="date"):

    df["days"] = (df[col_name] - df[col_name].min()).dt.days

    return df

# building the main function


def arps(df, date_col, production_smooth_col):
    """arps model
    fitting all arps models (exponential, hyperbolic, harmonic) and return the parameters and data for visualization

    parameters:
    df: pd.DataFrame
        data frame with two columns: date and production_smoothed

    return
    ------
    models_params: dictionary
    visualization_data : pd.DataFrame

    """
    df = get_days(df, col_name=date_col)
    T = df["days"]
    Q = df[production_smooth_col]

    # fitting the exponential
    exp_params = exponential_fitting(T, Q)
    Q_exp = exponential(T, exp_params["qi"], exp_params["di"])
    exp_errors = error_function(Q, Q_exp)

    # fitting the harmonic model
    h_params = harmonic_fitting(T, Q)
    Q_h = harmonic(T, h_params["qi"], h_params["di"])
    h_errors = error_function(Q, Q_h)

    # fitting the hyperbolic model
    hy_params = hyperbolic_fitting(T, Q)
    Q_hy = hyperbolic(T, hy_params["qi"], hy_params["di"], hy_params["b"])
    hy_errors = error_function(Q, Q_hy)

    # visualization data
    vis_data = pd.DataFrame({
        "Time": T,
        "original": Q,
        "exponential": Q_exp,
        "harmonic": Q_h,
        "hyperbolic": Q_hy
    }).set_index("Time")

    # params for all models
    params_dict = pd.DataFrame({
        "model": ["exponential", "harmonic", "hyperbolic"],
        "qi": [exp_params["qi"], h_params["qi"], hy_params["qi"]],
        "di": [exp_params["di"], h_params["di"], hy_params["di"]],
        "b": [exp_params["b"], h_params["b"], hy_params["b"]],
        "RMSE": [exp_errors, h_errors, hy_errors]
    })

    # return the params and the data for the visualization
    return params_dict, vis_data
