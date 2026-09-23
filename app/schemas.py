from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )

    cycle: int = Field(..., ge=1)

    setting_1: float
    setting_2: float

    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_17: float
    sensor_20: float
    sensor_21: float


class PredictionResponse(BaseModel):
    predicted_rul: float
    model_version: str
    unit: str = "cycles"