import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "data_source/yogyakarta_air_quality_2024.csv"
MODEL_PATH = "model.joblib"
METRICS_PATH = "metrics.json"

TARGET = "pm25_next"
BASE_FEATURES = ["pm10", "pm25", "so2", "co", "o3", "no2", "hc"]
LAGS = [1, 2, 3]
ROLLING_WINDOWS = [3, 7]

FINAL_FEATURES = ["pm25", "pm25_lag1", "pm25_roll3", "month"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Forecasting target: tomorrow's PM2.5
    df[TARGET] = df["pm25"].shift(-1)

    # Calendar features
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    for col in BASE_FEATURES:
        for lag in LAGS:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    for col in BASE_FEATURES:
        for window in ROLLING_WINDOWS:
            df[f"{col}_roll{window}"] = df[col].shift(1).rolling(window, min_periods=1).mean()

    return df


def main():
    df = pd.read_csv(DATA_PATH)
    df = build_features(df)

    feature_cols = FINAL_FEATURES

    model_df = df.dropna(subset=[TARGET]).copy()

    model_df[feature_cols] = model_df[feature_cols].fillna(model_df[feature_cols].median())

    print(f"Usable rows after target/feature prep: {len(model_df)} / {len(df)}")

    split_idx = int(len(model_df) * 0.8)
    train_df = model_df.iloc[:split_idx]
    test_df = model_df.iloc[split_idx:]

    x_train, y_train = train_df[feature_cols], train_df[TARGET]
    x_test, y_test = test_df[feature_cols], test_df[TARGET]

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    preds = model.predict(x_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    # Naive baseline: "tomorrow = today" persistence forecast
    baseline_preds = test_df["pm25"].values
    baseline_mae = mean_absolute_error(y_test, baseline_preds)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_preds))

    print(f"Test MAE:  {mae:.2f}  (baseline persistence MAE:  {baseline_mae:.2f})")
    print(f"Test RMSE: {rmse:.2f}  (baseline persistence RMSE: {baseline_rmse:.2f})")
    print(f"Test R^2:  {r2:.3f}")

    importances = sorted(
        zip(feature_cols, model.feature_importances_), key=lambda x: -x[1]
    )
    print("\nTop 10 feature importances:")
    for name, imp in importances[:10]:
        print(f"  {name:15s} {imp:.4f}")

    joblib.dump({"model": model, "feature_cols": feature_cols, "base_features": BASE_FEATURES,
                 "lags": LAGS, "rolling_windows": ROLLING_WINDOWS,
                 "medians": model_df[feature_cols].median().to_dict()}, MODEL_PATH)

    metrics = {
        "mae": mae, "rmse": rmse, "r2": r2,
        "baseline_mae": baseline_mae, "baseline_rmse": baseline_rmse,
        "n_train": len(train_df), "n_test": len(test_df),
        "top_features": [{"feature": n, "importance": float(i)} for n, i in importances[:10]],
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model to {MODEL_PATH} and metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
