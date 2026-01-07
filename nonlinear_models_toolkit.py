import numpy as np
import pandas as pd
import statsmodels.api as sm


# -----------------------------
# 1. Базовая модель (Ферхюльст)
# -----------------------------
def simulate_verhulst(n_steps=100, x0=0.005, A=None, K=None, a_norm=None,
                      noise_std=0.0, random_state=None):
    rng = np.random.default_rng(random_state)
    if a_norm is not None:
        if K is None:
            K = 1.0
        x = [x0 / K if K != 0 else x0]
        for _ in range(n_steps - 1):
            xn = x[-1]
            xn1 = xn + a_norm * xn * (1 - xn)
            if noise_std > 0:
                xn1 += rng.normal(0.0, noise_std)
            x.append(xn1)
        X = np.array(x) * K
        dX = np.diff(X, prepend=X[0])
        return pd.DataFrame({'t': np.arange(n_steps), 'X': X, 'dX': dX})
    else:
        assert A is not None and K is not None, \
            "Provide either (A,K) or a_norm (+optional K)"
        X = [x0]
        for _ in range(n_steps - 1):
            xn = X[-1]
            xn1 = xn + A * xn * (K - xn)
            if noise_std > 0:
                xn1 += rng.normal(0.0, noise_std)
            X.append(xn1)
        X = np.array(X)
        dX = np.diff(X, prepend=X[0])
        return pd.DataFrame({'t': np.arange(n_steps), 'X': X, 'dX': dX})


# -----------------------------
# 2. Модель с запаздыванием
# -----------------------------
def simulate_delayed(n_steps=100, x0=0.005, x1=None,
                     G=None, M=None, gm_norm=None,
                     noise_std=0.0, random_state=None):
    """
    x_{t+1} = x_t + g * x_t * (1 - x_{t-1})
    Нормированная форма (gm_norm = g*M):
        x_{t+1} = x_t + gm_norm * x_t * (1 - x_{t-1})
    """
    rng = np.random.default_rng(random_state)
    if gm_norm is not None:
        if M is None:
            M = 1.0
        if x1 is None:
            x1 = x0
        x = [x0 / M, x1 / M]
        for _ in range(n_steps - 2):
            xn = x[-1]
            xnm1 = x[-2]
            xn1 = xn + gm_norm * xn * (1 - xnm1)
            if noise_std > 0:
                xn1 += rng.normal(0.0, noise_std)
            x.append(xn1)
        X = np.array(x) * M
        dX = np.diff(X, prepend=X[0])
        return pd.DataFrame({'t': np.arange(len(X)), 'X': X, 'dX': dX})
    else:
        assert G is not None and M is not None, \
            "Provide either (G,M) or gm_norm (+optional M)"
        if x1 is None:
            x1 = x0
        X = [x0, x1]
        for _ in range(n_steps - 2):
            xn = X[-1]
            xnm1 = X[-2]
            xn1 = xn + G * xn * (M - xnm1)
            if noise_std > 0:
                xn1 += rng.normal(0.0, noise_std)
            X.append(xn1)
        X = np.array(X)
        dX = np.diff(X, prepend=X[0])
        return pd.DataFrame({'t': np.arange(len(X)), 'X': X, 'dX': dX})


# -----------------------------
# 3. Комбинированная модель (по формуле 1.6)
# -----------------------------
def simulate_combined(n_steps=150, x0=0.01, x1=None,
                      gamma=0.0, RQ=1.0,
                      noise_std=0.0, random_state=None):
    """
    Комбинированное отображение (формула 1.6 Полунина, нормированное):
        x_{t+1} = x_t + x_t * RQ * (1 - x_t - gamma * x_{t-1})
    где:
      RQ = Q*K — нормированная интенсивность,
      gamma — коэффициент влияния предыстории.
    """
    rng = np.random.default_rng(random_state)
    if x1 is None:
        x1 = x0
    x = [float(x0), float(x1)]
    for _ in range(n_steps - 2):
        xn = x[-1]
        xnm1 = x[-2]
        incr = RQ * xn * (1 - xn - gamma * xnm1)
        xn1 = xn + incr
        if noise_std > 0:
            xn1 += rng.normal(0.0, noise_std)
        x.append(xn1)
    arr = np.array(x)
    dX = np.diff(arr, prepend=arr[0])
    return pd.DataFrame({'t': np.arange(len(arr)), 'X': arr, 'dX': dX})


# -----------------------------
# 4. Вспомогательные функции
# -----------------------------
def lagged_dataframe(series, lags):
    df = pd.DataFrame({'y': series})
    for L in lags:
        df[f'y_lag{L}'] = df['y'].shift(L)
    return df.dropna()


def ols_with_betas(y, X, add_const=True):
    X_std = X.copy()
    mu = X_std.mean()
    sigma = X_std.std(ddof=0).replace(0, 1.0)
    X_std = (X_std - mu) / sigma

    if add_const:
        X_sm = sm.add_constant(X_std)
    else:
        X_sm = X_std

    model = sm.OLS(y, X_sm).fit()

    y_sigma = y.std(ddof=0)
    betas = {}
    for col in X_std.columns:
        beta = model.params.get(col, 0) * (X_std[col].std(ddof=0) / y_sigma)
        betas[col] = beta
    betas = pd.Series(betas)

    return model, betas, model.params


def fit_lag_regression(series, lags=(1,), difference=False):
    """
    Безопасная регрессия по лагам: чистка NaN/inf, проверка длины.
    """
    s = pd.Series(series, dtype=float)
    if difference:
        s = s.diff()

    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) <= max(lags):
        raise ValueError(
            f"Недостаточно конечных наблюдений после очистки: len={len(s)}, "
            f"max_lag={max(lags)}; возможно, траектория деградирует в NaN/inf."
        )

    df = lagged_dataframe(s, lags)
    if df.empty:
        raise ValueError("Пустая матрица лагов после dropna().")

    y = df['y']
    X = df[[f'y_lag{L}' for L in lags]]
    model, betas, params = ols_with_betas(y, X, add_const=True)
    return {
        'model': model,
        'betas': betas,
        'params': params,
        'adj_r2': model.rsquared_adj,
        'nobs': int(model.nobs)
    }


# -----------------------------
# 5. Скан по (gamma, RQ)
# -----------------------------
def scan_gamma_RQ(n_steps=160, warmup=40,
                  gammas=None, RQs=None,
                  lags=(1, 6, 7, 8, 9),
                  noise_std=0.0, random_state=0):
    """
    Пробегает сетку (gamma, RQ), симулирует комбинированную модель (1.6),
    режет warmup и оценивает регрессии.
    """
    if gammas is None:
        gammas = np.linspace(0.0, 0.9, 19)
    if RQs is None:
        RQs = np.linspace(0.4, 4.5, 41)

    rows = []
    for g in gammas:
        for rq in RQs:
            df = simulate_combined(
                n_steps=n_steps, x0=0.01, x1=0.011,
                gamma=g, RQ=rq,
                noise_std=noise_std, random_state=random_state
            )
            series = df['X'].iloc[warmup:]

            beta_lag1 = np.nan
            adjr2_lag1 = np.nan
            adjr2_all = np.nan
            sig_far = np.nan

            try:
                res1 = fit_lag_regression(series, lags=(1,), difference=False)
                beta_lag1 = res1['betas'].get('y_lag1', np.nan)
                adjr2_lag1 = res1['adj_r2']
            except Exception:
                pass

            try:
                resN = fit_lag_regression(series, lags=lags, difference=False)
                adjr2_all = resN['adj_r2']
                pvals = resN['model'].pvalues
                sig_far = sum(
                    int(pvals.get(f'y_lag{L}', 1.0) < 0.05)
                    for L in [6, 7, 8, 9]
                )
            except Exception:
                pass

            rows.append({
                'gamma': g,
                'RQ': rq,
                'beta_lag1': beta_lag1,
                'adjr2_lag1': adjr2_lag1,
                'adjr2_all': adjr2_all,
                'sig_far_lags': sig_far
            })

    return pd.DataFrame(rows)
