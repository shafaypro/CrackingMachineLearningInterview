# Time Series Analysis & Forecasting

Time series data is sequential data collected over time. It appears in finance (stock prices), operations (server metrics), retail (sales), and IoT. Understanding how to model, forecast, and evaluate time series is essential for ML and data science interviews.

---

## Table of Contents
1. [Core Concepts](#core-concepts)
2. [Stationarity](#stationarity)
3. [Decomposition](#decomposition)
4. [ARIMA & SARIMA](#arima--sarima)
5. [Prophet](#prophet)
6. [ML/DL Approaches](#mldl-approaches)
7. [Feature Engineering for Time Series](#feature-engineering-for-time-series)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Cross-Validation for Time Series](#cross-validation-for-time-series)
10. [Interview Q&A](#interview-qa)
11. [Common Pitfalls](#common-pitfalls)
12. [Related Topics](#related-topics)

---

## Core Concepts

| Concept | Definition |
|---------|-----------|
| **Trend** | Long-term increase or decrease in the data |
| **Seasonality** | Repeating pattern at fixed intervals (weekly, monthly, yearly) |
| **Cyclicality** | Irregular fluctuations not at fixed periods (business cycles) |
| **Noise / Residual** | Random variation after removing trend and seasonality |
| **Lag** | A previous time step value: `y(t-1)`, `y(t-7)` |
| **Autocorrelation** | Correlation of a series with its own past values |
| **Stationarity** | Statistical properties (mean, variance) do not change over time |

---

## Stationarity

Most time series models (ARIMA) require stationary data. A stationary series has:
- Constant mean
- Constant variance
- Constant autocovariance structure

### Augmented Dickey-Fuller (ADF) Test

```python
from statsmodels.tsa.stattools import adfuller
import pandas as pd

def test_stationarity(series, significance=0.05):
    result = adfuller(series.dropna())
    p_value = result[1]
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Critical Values: {result[4]}")
    if p_value < significance:
        print("Result: STATIONARY (reject H0)")
    else:
        print("Result: NON-STATIONARY (fail to reject H0)")

# Example
import numpy as np
np.random.seed(42)
ts = pd.Series(np.cumsum(np.random.randn(200)))  # Random walk (non-stationary)
test_stationarity(ts)
```

### Making a Series Stationary

```python
# Method 1: Differencing (most common)
ts_diff = ts.diff().dropna()

# Method 2: Log transform (stabilizes variance)
ts_log = np.log(ts)

# Method 3: Log + differencing (handles both trend and variance)
ts_log_diff = np.log(ts).diff().dropna()

# Check again
test_stationarity(ts_diff)
```

---

## Decomposition

```python
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Additive: Y(t) = Trend + Seasonal + Residual  (when seasonal variation is constant)
# Multiplicative: Y(t) = Trend * Seasonal * Residual  (when seasonal variation grows with level)

result = seasonal_decompose(ts, model='additive', period=12)

# Components
trend = result.trend
seasonal = result.seasonal
residual = result.resid
```

---

## ARIMA & SARIMA

**ARIMA(p, d, q)**:
- **p** — AutoRegressive order: how many lagged values of y
- **d** — Integration order: how many times to difference to achieve stationarity
- **q** — Moving Average order: how many lagged forecast errors

**SARIMA(p, d, q)(P, D, Q, m)**: Seasonal extension where m = seasonal period.

### Selecting Parameters with ACF/PACF

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ACF: helps identify q (MA order)  — cuts off after lag q
# PACF: helps identify p (AR order) — cuts off after lag p
plot_acf(ts_diff, lags=40)
plot_pacf(ts_diff, lags=40)
```

### Fitting ARIMA

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Fit ARIMA(2,1,2)
model = ARIMA(ts, order=(2, 1, 2))
result = model.fit()
print(result.summary())

# Forecast
forecast = result.forecast(steps=12)

# Auto-select parameters
from pmdarima import auto_arima
auto_model = auto_arima(ts, seasonal=True, m=12, stepwise=True, suppress_warnings=True)
print(auto_model.summary())

# SARIMA with seasonality (monthly data, period=12)
sarima = SARIMAX(ts, order=(1,1,1), seasonal_order=(1,1,1,12))
sarima_result = sarima.fit()
forecast = sarima_result.forecast(steps=24)
```

---

## Prophet

Facebook Prophet is designed for business forecasting with:
- Automatic trend changepoint detection
- Multiple seasonalities (daily, weekly, yearly)
- Holiday effects
- Handles missing data and outliers well

```python
from prophet import Prophet
import pandas as pd

# Prophet requires columns: 'ds' (datetime) and 'y' (value)
df = pd.DataFrame({'ds': pd.date_range('2020-01-01', periods=365), 'y': np.random.randn(365).cumsum()})

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.05,  # Flexibility of trend (higher = more flexible)
    seasonality_prior_scale=10.0,   # Strength of seasonality
)

# Add custom seasonality
model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

# Add holidays
from prophet.make_holidays import make_holidays_df
holidays = make_holidays_df(year_list=[2020, 2021, 2022], country='US')
model = Prophet(holidays=holidays)

# Add external regressors
# model.add_regressor('temperature')

model.fit(df)

# Forecast
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# Plot components
fig = model.plot_components(forecast)
```

---

## ML/DL Approaches

### Lag Features with Gradient Boosting

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

def create_lag_features(series, lags=[1, 7, 14, 28]):
    df = pd.DataFrame({'y': series})
    for lag in lags:
        df[f'lag_{lag}'] = df['y'].shift(lag)
    # Rolling statistics
    df['rolling_mean_7'] = df['y'].shift(1).rolling(7).mean()
    df['rolling_std_7'] = df['y'].shift(1).rolling(7).std()
    return df.dropna()

df_features = create_lag_features(ts)
X = df_features.drop('y', axis=1)
y = df_features['y']

model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05)
model.fit(X[:-30], y[:-30])
preds = model.predict(X[-30:])
```

### LSTM for Sequence Forecasting

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Prepare sequences
def create_sequences(data, seq_length=30):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(ts.values.reshape(-1, 1))

X, y = create_sequences(scaled, seq_length=30)
X = X.reshape(X.shape[0], X.shape[1], 1)  # (samples, timesteps, features)

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(30, 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X[:-30], y[:-30], epochs=20, batch_size=32, verbose=0)
```

---

## Feature Engineering for Time Series

```python
import pandas as pd

def time_series_features(df, date_col='date'):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Calendar features
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['day_of_year'] = df[date_col].dt.dayofyear
    df['week_of_year'] = df[date_col].dt.isocalendar().week
    df['quarter'] = df[date_col].dt.quarter
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
    df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)

    # Cyclical encoding (avoids discontinuity Dec→Jan)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    return df
```

---

## Evaluation Metrics

| Metric | Formula | Use Case |
|--------|---------|---------|
| **MAE** | mean(\|y - ŷ\|) | Robust to outliers, easy to interpret |
| **MSE** | mean((y - ŷ)²) | Penalizes large errors |
| **RMSE** | √MSE | Same units as target |
| **MAPE** | mean(\|y-ŷ\|/\|y\|) × 100 | Percentage error, but unstable when y≈0 |
| **sMAPE** | mean(2\|y-ŷ\|/(|y|+\|ŷ\|)) | Symmetric MAPE, bounded 0-200% |
| **MASE** | MAE / MAE_naive | Scale-free; compares to naive forecast |

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def evaluate_forecast(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}
```

---

## Cross-Validation for Time Series

**Never use random cross-validation for time series** — it causes data leakage. Always use forward-chaining (walk-forward validation).

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5, gap=0)

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    # Each fold: training set grows, validation window moves forward
    print(f"Fold {fold}: train={len(X_train)}, val={len(X_val)}")
```

---

## Interview Q&A

**Q1: What is stationarity and why does it matter?**
A stationary series has constant mean, variance, and autocorrelation over time. ARIMA and other statistical models assume stationarity because their parameters (mean, variance structure) would otherwise keep shifting. Non-stationary series lead to spurious regressions. Fix with differencing, log transforms, or seasonal adjustment.

**Q2: What's the difference between AR, MA, and ARIMA models?**
- **AR(p)**: regresses y on its own past p values. Good when past values predict future ones (autocorrelation).
- **MA(q)**: regresses y on past q forecast errors. Good when shocks have short-lived effects.
- **ARIMA(p,d,q)**: combines AR + differencing (d times) + MA. d makes the series stationary; p and q capture remaining patterns.

**Q3: How do you choose p and d and q in ARIMA?**
- d: difference until ADF test shows stationarity (typically d=0 or d=1)
- p: look at PACF plot — significant lags before cutoff
- q: look at ACF plot — significant lags before cutoff
- Alternatively: use `auto_arima` with AIC/BIC minimization

**Q4: When would you use Prophet over ARIMA?**
Use Prophet when: data has strong multiple seasonalities (weekly + yearly), there are holidays/events to model, stakeholders need interpretable components, or you have missing data. Prophet handles these automatically. Use ARIMA when: data is low-frequency (monthly), you need strict statistical rigor, or you're doing multivariate forecasting.

**Q5: What is data leakage in time series?**
Using future information to predict the past. Common mistakes:
- Using random train/test split (test data includes older timestamps than training data)
- Computing rolling features without properly shifting (using the target time step's value in the feature)
- Scaling the entire dataset before splitting (test statistics leak into training scaler)
Always split in time order, and shift lag features by at least 1 step.

**Q6: How do you handle seasonality in a gradient boosting model?**
Extract calendar features (month, day of week, etc.) with cyclical encoding (sin/cos). Add lag features at seasonal periods (e.g., lag_7 for weekly data, lag_365 for yearly). Add Fourier terms to model smooth seasonal patterns. GBM can learn these patterns if given the right features.

**Q7: What's the difference between forecasting and anomaly detection in time series?**
Forecasting predicts future values (point forecast or interval). Anomaly detection identifies values that deviate significantly from expected behavior — often using forecast residuals (if residual > threshold, it's an anomaly). LSTM autoencoders, Isolation Forest on lag features, and statistical process control (Z-score on rolling stats) are common approaches.

**Q8: How would you evaluate a forecasting model on multiple SKUs/stores?**
Use MASE (Mean Absolute Scaled Error) — it's scale-free so it's comparable across series with different magnitudes. Also use Weighted MAPE (weighted by volume) so high-volume SKUs drive the score. Report both mean and distribution of errors to catch underperforming segments.

---

## Common Pitfalls

| Pitfall | Problem | Fix |
|---------|---------|-----|
| Random CV on time series | Data leakage from future | Use `TimeSeriesSplit` |
| Not checking stationarity | ARIMA fails or gives spurious results | ADF test; difference if needed |
| Forgetting to shift lag features | Target leakage | Always `.shift(1)` before computing lags |
| Scaling before split | Test statistics contaminate training | Fit scaler on training set only |
| MAPE on near-zero values | Division by zero / infinity | Use sMAPE or MASE instead |
| Ignoring holidays | Poor accuracy around holidays | Add holiday indicators or use Prophet |
| Overfitting with too many lags | Noise, not signal | Use feature importance to prune; regularize |

---

## Related Topics

| Topic | Why It's Related |
|-------|-----------------|
| [Feature Engineering](./intro_feature_engineering.md) | Time features, lag features, rolling statistics |
| [Classic Question Bank](../README.md#classic-question-bank) | ARIMA is covered in the main Q&A |
| [Study Pattern](../docs/study-pattern.md) | Time Series is an Advanced (🔴) topic |
