#!/usr/bin/env python3

import time
import timeit
from functools import partial
from typing import List

import docker as docker
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from urllib3 import Retry

from app.inference_models import TARGET_NAMES, FEATURE_NAMES
from app.inference_onnx import predict_onnx
from app.inference_scikitlearn import predict_sklearn
from training import train_and_serialize


def time_inference(repeat: int = 10, number: int = 1_0):
    """
    Measure inference time
    :param repeat:
    :param number:
    :return:
    """
    for f, runtime in ((predict_sklearn, "sklearn"), (predict_onnx, "onnx")):
        for batch_size in (1, 10, 100):
            data = [[0.0, 0.0, 0.0, 0.0] for _ in range(batch_size)]
            min_time = min(timeit.Timer(partial(f, data)).repeat(repeat=repeat, number=number))
            print(f"Timing for {runtime} batch size {batch_size}: {min_time}")


def build_docker_images() -> List[str]:
    """
    Create a set of docker images
    :return:
    """

    images = []

    dockerfiles = {
        "debian": ["scikitlearn", "onnx"],
        "alpine": ["onnx"],
    }

    for image, runtimes in dockerfiles.items():
        for runtime in runtimes:
            tag = f"app-{image}-{runtime}"
            print(f"Building image {tag}")
            docker_image = docker.from_env().images.build(
                path=".",
                dockerfile=f"Dockerfile.{image}",
                buildargs={
                    "RUNTIME": runtime,
                },
                tag=tag,
            )
            images.append(tag)
            print(f"Size: {round(docker_image[0].attrs.get('Size') / 10**6)} MB")
    return images


def test_images(model, x_test, y_test, images: List[str], host_port: int = 8000, n_memory_samples: int = 5):
    """
    Run container and use client to test web service
    :param images:
    :return:
    """

    print("Testing images...")

    y_pred = model.predict(x_test)
    expected_output = [TARGET_NAMES[class_id] for class_id in y_pred]
    x_test_json = {"data": [dict(zip(FEATURE_NAMES, sample)) for sample in x_test.tolist()]}
    print(confusion_matrix(y_test, y_pred))

    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))

    for image in images:
        print(f"Image: {image}")
        c = docker.from_env().containers.run(image, ports={80: host_port}, detach=True)
        try:
            r = session.get(f"http://localhost:{host_port}/docs/")
            assert r.status_code == 200
            predictions = session.post(f"http://localhost:{host_port}/", json=x_test_json)
            assert predictions.status_code == 200, f"Code was {predictions.status_code}, {predictions.json()}"
            predicted_classes = predictions.json().get("classes")
            assert len(expected_output) == len(predicted_classes)
            assert all([expected == actual for expected, actual in zip(expected_output, predicted_classes)])
            stats = c.stats(stream=False)
            memory_usages = []
            for _ in range(n_memory_samples):
                time.sleep(1)
                memory_usages.append(stats.get("memory_stats").get("usage") / 10**6)
            print(f'Memory usage mean: {np.mean(memory_usages)}, std: {np.std(memory_usages)} MB')
        except Exception as e:
            print(f"failed with: {e}")
        finally:
            c.kill()

    print("Tests done.")


def main():
    # train the model and serialize
    data, target = datasets.load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=42)
    model = train_and_serialize(x_train, y_train)

    # measure inference time
    time_inference()

    # build docker images
    images = build_docker_images()

    # test images
    test_images(model, x_test, y_test, images)


if __name__ == "__main__":
    main()
