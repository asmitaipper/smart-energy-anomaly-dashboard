import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path


DATA_PATH = Path("data/energy_usage_sample.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "energy_iforest.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df_num = df.copy()
    df_num["device_id"] = df_num["device_id"].astype("category").cat.codes
    df_num["room"] = df_num["room"].astype("category").cat.codes

    agg = (
        df_num.groupby(["timestamp", "device_id", "room"])
        .agg(
            power_kw=("power_kw", "mean"),
            voltage=("voltage", "mean"),
        )
        .reset_index()
    )

    return agg


def train_isolation_forest(X_scaled):
    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)
    return model


def main():
    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Building features...")
    feats = build_features(df)
    feature_cols = ["power_kw", "voltage"]
    X = feats[feature_cols].values

    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Training IsolationForest...")
    model = train_isolation_forest(X_scaled)

    print(f"Saving model to {MODEL_PATH} and scaler to {SCALER_PATH}")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
