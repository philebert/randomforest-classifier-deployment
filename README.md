# Deployment of a random forest classifier

A scikit-learn based random forest classifier with default configuration is trained on the iris dataset.  

Two different approaches are used for inference:

* saving the model via joblib and using scikit-learn for inference
* using onnxruntime after converting the model to onnx file format

The following table shows inference time for different batch sizes:

<table>
    <thead>
        <tr>
            <th rowspan="2">Runtime</th>
            <th colspan="3">Batch Size</th>
        </tr>
        <tr style="text-align: center">
            <td>1</td>
            <td>10</td>
            <td>100</td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Sklearn</td>
            <td style="text-align: right">6.96 ms</td>
            <td style="text-align: right">7.13 ms</td>
            <td style="text-align: right">7.56 ms</td>
        </tr>
        <tr>
            <td>ONNX</td>
            <td style="text-align: right">0.0167 ms</td>
            <td style="text-align: right">0.0291 ms</td>
            <td style="text-align: right">0.137 ms</td>
        </tr>
    </tbody>
</table>

Docker images are created based on both, Debian Buster slim and alpine. 
Fastapi web framework along with gunicorn server and uvicorn worker class is used to offer predictions as a web service.

<table>
    <thead>
        <tr>
            <th>Docker Base Image</th>
            <th>Runtime</th>
            <th>Docker Image Size (MB)</th>
            <th>Container Memory Usage IDLE (MB)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=2>slim-buster</td>
            <td>scikit-learn</td>
            <td style="text-align: right">587</td>
            <td style="text-align: right">164</td>
        </tr>
        <tr>
            <td>onnx</td>
            <td style="text-align: right">317</td>
            <td style="text-align: right">67.7</td>
        </tr>
        <tr>
            <td>alpine</td>
            <td>onnx</td>
            <td style="text-align: right">163</td>
            <td style="text-align: right">53.1</td>
        </tr>
    </tbody>
</table>

## Remarks

* Benchmark done on system with Intel® Core™ i7-7700HQ

## References

* https://cloudblogs.microsoft.com/opensource/2020/12/17/accelerate-simplify-scikit-learn-model-inference-onnx-runtime/
* https://scikit-learn.org/stable/auto_examples/applications/plot_prediction_latency.html