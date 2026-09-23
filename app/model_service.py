from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class ModelService:
    def __init__(self, model_path: Path):
        self.model_path = model_path

        self.model = None
        self.feature_names = None
        self.model_version = None
        self.dataset = None
        self.postprocessing = None

        self.load_error = None

    def load_model(
        self,
        model_path: Path | None = None,
    ) -> None:
        if model_path is not None:
            self.model_path = model_path
        try:
            artifact = joblib.load(self.model_path)

            if not isinstance(artifact, dict):
                raise ValueError(
                    "Model artifact must be a dictionary."
                )

            required_keys = {
                "model",
                "feature_names",
                "model_version",
                "postprocessing",
            }

            missing_keys = required_keys - set(artifact.keys())

            if missing_keys:
                raise ValueError(
                    f"Model artifact missing required keys: "
                    f"{sorted(missing_keys)}"
                )

            model = artifact["model"]
            feature_names = artifact["feature_names"]

            if not hasattr(model, "predict"):
                raise ValueError(
                    "Loaded model does not provide predict()."
                )

            if not isinstance(feature_names, list):
                raise ValueError(
                    "feature_names must be stored as a list."
                )

            if hasattr(model, "n_features_in_"):
                if model.n_features_in_ != len(feature_names):
                    raise ValueError(
                        "Model feature count does not match "
                        "stored feature schema."
                    )

            self.model = model
            self.feature_names = feature_names
            self.model_version = artifact["model_version"]
            self.dataset = artifact.get("dataset")
            self.postprocessing = artifact["postprocessing"]

            self.load_error = None

        except Exception as exc:
            self.model = None
            self.feature_names = None
            self.model_version = None
            self.dataset = None
            self.postprocessing = None

            self.load_error = str(exc)

    @property
    def is_ready(self) -> bool:
        return (
            self.model is not None
            and self.feature_names is not None
            and self.model_version is not None
        )

    def predict(self, sensor_data: dict) -> float:
        if not self.is_ready:
            raise RuntimeError(
                "Model service is not ready."
            )

        missing = set(self.feature_names) - set(sensor_data)
        extra = set(sensor_data) - set(self.feature_names)

        if missing or extra:
            raise ValueError(
                f"Missing features: {sorted(missing)}. "
                f"Unexpected features: {sorted(extra)}."
            )

        X = pd.DataFrame(
            [[sensor_data[name] for name in self.feature_names]],
            columns=self.feature_names,
        )

        X = X.apply(
            pd.to_numeric,
            errors="raise",
        )

        if not np.isfinite(
            X.to_numpy(dtype=float)
        ).all():
            raise ValueError(
                "All features must be finite numbers."
            )

        cycle = X["cycle"].iloc[0]

        if cycle < 1 or not float(cycle).is_integer():
            raise ValueError(
                "Cycle must be a positive integer."
            )

        prediction = self.model.predict(X)[0]

        # Artifact specifies:
        # "clip predictions to zero"
        prediction = max(
            float(prediction),
            0.0,
        )

        return prediction