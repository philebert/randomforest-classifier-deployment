from typing import List

import joblib
import uvicorn as uvicorn
from fastapi import FastAPI

from app.inference_models import InferenceBatchRequest, Prediction, TARGET_NAMES


def predict_sklearn(data: List) -> List[str]:
    classes_int = model.predict(data).tolist()
    return [TARGET_NAMES[i] for i in classes_int]


model = joblib.load("app/random-forest-iris.joblib")
app = FastAPI()


@app.post("/", response_model=Prediction)
def predict(batch: InferenceBatchRequest) -> Prediction:
    predicted_classes = predict_sklearn(
        [[s.sepal_length_cm, s.sepal_width_cm, s.petal_length_cm, s.petal_width_cm] for s in batch.data]
    )
    return Prediction(classes=predicted_classes)


if __name__ == "__main__":
    uvicorn.run(app, host="localhost")
