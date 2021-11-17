from typing import List

import uvicorn as uvicorn
from fastapi import FastAPI
import onnxruntime as rt

from app.inference_models import InferenceBatchRequest, Prediction, TARGET_NAMES


def predict_onnx(data: List) -> List[str]:
    classes_int = inference_session.run(["output_label"], {"input": data})[0].tolist()
    return [TARGET_NAMES[i] for i in classes_int]


inference_session = rt.InferenceSession("app/random-forest-iris.onnx")

app = FastAPI()


@app.post("/", response_model=Prediction)
def predict(batch: InferenceBatchRequest) -> Prediction:
    predicted_classes = predict_onnx(
        [[s.sepal_length_cm, s.sepal_width_cm, s.petal_length_cm, s.petal_width_cm] for s in batch.data]
    )
    return Prediction(classes=predicted_classes)


if __name__ == "__main__":
    uvicorn.run(app, host="localhost")
