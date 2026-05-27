import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("../onnx_format.onnx")

x_data = np.array([[0.0], [2.0], [4.0]], dtype=np.float32)

outputs = session.run(None, {"X": x_data})

y_data = outputs[0]
print("Predictions (Y):")
print(y_data)
