from pathlib import Path

import numpy
import numpy as np
from joblib import dump
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

numpy.random.seed(42)


def onnx_export(model: RandomForestClassifier, output: Path):
    """
    Export sklearn random forest classifier to onnx
    :param model:
    :param output:
    :return:
    """
    initial_type = [("input", FloatTensorType([None, 4]))]
    onx = convert_sklearn(model, initial_types=initial_type)
    with open(output, "wb") as f:
        f.write(onx.SerializeToString())


def train(x: np.array, y: np.array) -> RandomForestClassifier:
    """
    Train a random forest classifier on the whole iris data set, no cross-validation
    :param x:
    :param y:
    :return:
    """
    model = RandomForestClassifier()
    model.fit(x, y)
    return model


def train_and_serialize(x: np.array, y: np.array, output_folder: Path = Path("./app")) -> RandomForestClassifier:
    """
    Train model, serialize as joblib and onnx
    :param x:
    :param y:
    :param output_folder:
    :return:
    """
    model = train(x, y)
    dump(model, output_folder / "random-forest-iris.joblib")
    onnx_export(model, output_folder / "random-forest-iris.onnx")
    return model


if __name__ == "__main__":
    data, target = datasets.load_iris(return_X_y=True)
    train_and_serialize(data, target)
