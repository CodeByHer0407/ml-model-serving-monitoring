from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMRegressor


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train_FD001.txt"

V1_MODEL_PATH = (
    PROJECT_ROOT / "models" / "rul_model_v1.joblib"
)

V2_MODEL_PATH = (
    PROJECT_ROOT / "models" / "rul_model_v2.joblib"
)


COLUMNS = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_training_data() -> pd.DataFrame:
    """Load NASA C-MAPSS FD001 training data."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: {DATA_PATH}"
        )

    return pd.read_csv(
        DATA_PATH,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
    )


def calculate_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate RUL for run-to-failure training data."""

    df = df.copy()

    max_cycles = (
        df.groupby("engine_id")["cycle"]
        .transform("max")
    )

    df["rul"] = max_cycles - df["cycle"]

    return df


def train_v2() -> None:
    """Train candidate V2 while preserving V1 feature schema."""

    print("Loading V1 artifact...")

    v1_artifact = joblib.load(V1_MODEL_PATH)

    feature_names = v1_artifact["feature_names"]

    print("Feature count:", len(feature_names))
    print("Feature schema:", feature_names)

    print("\nLoading training data...")

    df = load_training_data()
    df = calculate_rul(df)

    print("Training data shape:", df.shape)

    X_train = df[feature_names]
    y_train = df["rul"]

    print("Training features shape:", X_train.shape)

    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )

    print("\nTraining candidate model V2...")

    model.fit(
        X_train,
        y_train,
    )

    print("V2 training completed.")

    artifact = {
        "model": model,
        "feature_names": feature_names,
        "model_version": "v2",
        "dataset": "NASA C-MAPSS FD001",
        "postprocessing": "clip predictions to zero",
        "training_config": {
            "n_estimators": 500,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "random_state": 42,
        },
    }

    joblib.dump(
        artifact,
        V2_MODEL_PATH,
    )

    print("\nV2 artifact saved:")
    print(V2_MODEL_PATH)


if __name__ == "__main__":
    train_v2()