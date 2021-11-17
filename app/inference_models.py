from typing import List

from pydantic import BaseModel, create_model


FEATURE_NAMES = ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"]
TARGET_NAMES = ["setosa", "versicolor", "virginica"]


InferenceRequest = create_model(
    "InferenceRequest",
    **{feature: (float, None) for feature in FEATURE_NAMES},
)


class InferenceBatchRequest(BaseModel):
    data: List[InferenceRequest]


class Prediction(BaseModel):
    classes: List[str]
