import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("../iris_model.onnx")

x_data = np.array([[6.9,3.1,5.4,2.1], [5.0,3.3,1.4,0.2], [5.1,3.4,1.5,0.2], [6.1,2.9,4.7,1.4]], dtype=np.float32)

outputs = session.run(None, {"X": x_data})

y_data = outputs[0]
print("Predictions (Y):")
print(y_data)

