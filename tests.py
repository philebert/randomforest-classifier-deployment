from app.inference_models import TARGET_NAMES
from app.inference_onnx import predict_onnx
from app.inference_scikitlearn import predict_sklearn


def test_sklearn_inference():
    prediction = predict_sklearn(data=[[0.0, 0.0, 0.0, 0.0]])
    assert len(prediction) == 1
    assert prediction[0] in TARGET_NAMES


def test_onnx_inference():
    prediction = predict_onnx(data=[[0.0, 0.0, 0.0, 0.0]])
    assert len(prediction) == 1
    assert prediction[0] in TARGET_NAMES
