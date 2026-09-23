import json
from pathlib import Path

import joblib


class ModelRegistry:
    def __init__(
        self,
        models_dir: Path,
        state_path: Path,
    ):
        self.models_dir = models_dir
        self.state_path = state_path

        self.models = self._discover_models()

        if not self.models:
            raise RuntimeError(
                "No model artifacts found."
            )

        self._initialize_state()

    def _discover_models(self) -> dict:
        """Discover valid versioned model artifacts."""

        models = {}

        for model_path in sorted(
            self.models_dir.glob("rul_model_v*.joblib")
        ):
            artifact = joblib.load(model_path)

            if not isinstance(artifact, dict):
                continue

            version = artifact.get("model_version")

            if not version:
                continue

            models[version] = {
                "path": model_path,
                "dataset": artifact.get("dataset"),
                "model_type": type(
                    artifact.get("model")
                ).__name__,
                "feature_count": len(
                    artifact.get("feature_names", [])
                ),
                "training_config": artifact.get(
                    "training_config"
                ),
            }

        return models

    def _initialize_state(self) -> None:
        """Create registry state if it does not exist."""

        self.state_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if self.state_path.exists():
            state = self._read_state()

            active_version = state.get(
                "active_version"
            )

            if active_version not in self.models:
                raise ValueError(
                    "Registry active model does not exist."
                )

            return

        default_version = (
            "v1"
            if "v1" in self.models
            else sorted(self.models)[0]
        )

        self._write_state(
            {
                "active_version": default_version,
                "previous_version": None,
            }
        )

    def _read_state(self) -> dict:
        with self.state_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    def _write_state(self, state: dict) -> None:
        """Write state atomically."""

        temp_path = self.state_path.with_suffix(
            ".tmp"
        )

        with temp_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                state,
                file,
                indent=2,
            )

        temp_path.replace(self.state_path)

    def list_models(self) -> list[dict]:
        state = self._read_state()

        active_version = state[
            "active_version"
        ]

        result = []

        for version, metadata in self.models.items():
            result.append(
                {
                    "version": version,
                    "active": (
                        version == active_version
                    ),
                    "dataset": metadata["dataset"],
                    "model_type": metadata[
                        "model_type"
                    ],
                    "feature_count": metadata[
                        "feature_count"
                    ],
                    "training_config": metadata[
                        "training_config"
                    ],
                }
            )

        return result

    def get_model_path(
        self,
        version: str,
    ) -> Path:
        if version not in self.models:
            raise KeyError(
                f"Model version '{version}' not found."
            )

        return self.models[version]["path"]

    def get_active_version(self) -> str:
        return self._read_state()[
            "active_version"
        ]

    def get_previous_version(
        self,
    ) -> str | None:
        return self._read_state().get(
            "previous_version"
        )

    def get_active_model_path(self) -> Path:
        return self.get_model_path(
            self.get_active_version()
        )

    def activate(
        self,
        version: str,
    ) -> dict:
        if version not in self.models:
            raise KeyError(
                f"Model version '{version}' not found."
            )

        state = self._read_state()

        current = state["active_version"]

        if current == version:
            return state

        new_state = {
            "active_version": version,
            "previous_version": current,
        }

        self._write_state(new_state)

        return new_state

    def rollback(self) -> dict:
        state = self._read_state()

        previous = state.get(
            "previous_version"
        )

        if previous is None:
            raise ValueError(
                "No previous model version "
                "available for rollback."
            )

        current = state["active_version"]

        new_state = {
            "active_version": previous,
            "previous_version": current,
        }

        self._write_state(new_state)

        return new_state