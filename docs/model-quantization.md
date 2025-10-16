# Model Quantization: Deep Dive

This document provides a comprehensive guide to model quantization, including theory, practical steps, and best practices for PyTorch and ONNX models.

---

## What is Model Quantization?

Model quantization is the process of reducing the precision of the numbers used to represent a model's parameters and activations. The goal is to make models smaller, faster, and more efficient for deployment, especially on edge devices (mobile, IoT, embedded systems).

### Why Quantize?

- **Reduced Model Size:** Lower precision (e.g., INT8 vs FP32) means less memory usage.
- **Faster Inference:** Integer operations are faster than floating-point on many hardware platforms.
- **Lower Power Consumption:** Useful for battery-powered devices.
- **Deployment on Edge Devices:** Many edge accelerators require quantized models.

---

## Types of Quantization

1. **Post-Training Quantization (PTQ):**
   - Quantize a trained model without retraining.
   - Fast, but may reduce accuracy.
2. **Quantization-Aware Training (QAT):**
   - Simulate quantization during training.
   - Higher accuracy, but requires retraining.
3. **Dynamic Quantization:**
   - Weights quantized at runtime (common for NLP models).
4. **Static Quantization:**
   - Both weights and activations quantized using calibration data.

---

## Quantization in PyTorch

PyTorch provides built-in support for quantization:

### 1. Post-Training Static Quantization

```python
import torch
from torchvision.models import resnet18
from torch.quantization import quantize_dynamic, quantize, get_default_qconfig

model = resnet18(pretrained=True)
model.eval()

# Prepare model for static quantization
model.qconfig = get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)

# Calibrate with representative data
for data, _ in calibration_loader:
    model(data)

# Convert to quantized model
quantized_model = torch.quantization.convert(model, inplace=True)
```

### 2. Dynamic Quantization

```python
quantized_model = quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

### 3. Quantization-Aware Training (QAT)

```python
model.train()
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
torch.quantization.prepare_qat(model, inplace=True)
# Train as usual
# ...
# Convert after training
quantized_model = torch.quantization.convert(model.eval(), inplace=True)
```

#### PyTorch Quantization Tips

- Use `fbgemm` for x86 CPUs, `qnnpack` for ARM.
- Always calibrate with representative data.
- QAT yields better accuracy for complex models.
- Some layers (e.g., BatchNorm) may need fusion before quantization.

---

## Quantization in ONNX

ONNX supports quantization via ONNX Runtime and external tools.

### 1. Export PyTorch Model to ONNX

```python
import torch
model = ... # your trained model

torch.onnx.export(model, dummy_input, 'model.onnx', opset_version=13)
```

### 2. ONNX Quantization Tools

- **ONNX Runtime Quantization API**
- **Neural Network Compression Framework (NNCF)**
- **Hugging Face Optimum**

#### Example: ONNX Runtime PTQ

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantized_model_path = quantize_dynamic(
    'model.onnx', 'model-quantized.onnx', weight_type=QuantType.QInt8
)
```

---

## ONNX Runtime Preprocess Command Explained

**REF:** <https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/ReadMe.md>

The following command:

```fish
python -m onnxruntime.quantization.preprocess --input mobilenetv2-7.onnx --output mobilenetv2-7-infer.onnx
```

**What it does:**

- Runs the ONNX Runtime quantization preprocessing tool as a Python module.
- Takes an input ONNX model (`mobilenetv2-7.onnx`).
- Produces a new ONNX model (`mobilenetv2-7-infer.onnx`) that is optimized for inference and quantization.

**Details:**

- The preprocess step may convert model ops to quantization-friendly formats, fuse layers, and ensure the model is ready for static quantization/calibration.
- It does not quantize the model, but prepares it for subsequent quantization steps.
- The output model is typically used as input for static quantization (e.g., with `quantize_static`).

**When to use:**

- Before running static quantization or calibration on models exported from frameworks that may not be quantization-ready.

**Reference:**

- [ONNX Runtime Quantization Preprocess](https://onnxruntime.ai/docs/performance/quantization.html)

#### Example: ONNX Runtime QAT

- Requires training with quantization simulation (see ONNX docs)

#### ONNX Quantization Tips

- ONNX quantization is hardware-agnostic and widely supported.
- Use ONNX for deployment on non-PyTorch platforms (TensorRT, OpenVINO, etc.).
- Quantize after exporting from PyTorch for maximum compatibility.

#### Calibration in ONNX Quantization

**Calibration:**
For static quantization (PTQ with activations), ONNX Runtime requires calibration data to determine optimal quantization parameters (scales and zero points). Calibration data should be representative of real-world inputs.

**How to Calibrate:**

- Use ONNX Runtime's `quantize_static` API and provide a calibration dataset.
- The API will run inference on the calibration data and collect activation statistics.

**Example: Static Quantization with Calibration**

```python
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

class MyDataReader(CalibrationDataReader):
   def __init__(self, input_data):
      self.data = input_data
      self.enum_data = iter(self.data)
   def get_next(self):
      return next(self.enum_data, None)

calibration_data = [ { 'input': preprocessed_input1 }, { 'input': preprocessed_input2 }, ... ]
data_reader = MyDataReader(calibration_data)

quantize_static(
   'model.onnx',
   'model-quantized.onnx',
   data_reader,
   weight_type=QuantType.QInt8,
   activation_type=QuantType.QInt8
)
```

**Summary:**

- Preprocessing and calibration are essential for accurate static quantization in ONNX.
- Dynamic quantization does not require calibration data, but static quantization does.
- Always use representative data for calibration to minimize accuracy loss.

---

## Should You Quantize in PyTorch or ONNX?

| Criteria             | PyTorch Quantization | ONNX Quantization |
| -------------------- | -------------------- | ----------------- |
| **Ease of Use**      | Good                 | Good              |
| **Accuracy**         | QAT best in PyTorch  | Similar with QAT  |
| **Deployment**       | PyTorch-only         | Cross-platform    |
| **Hardware Support** | CPU, some GPU        | CPU, GPU, Edge    |
| **Tooling**          | Native               | ONNX Runtime      |
| **Speed**            | Fast                 | Fast              |

**Recommendation:**

- If deploying on PyTorch-only environments, quantize in PyTorch.
- For cross-platform deployment (TensorRT, OpenVINO, mobile, edge), quantize in ONNX.
- For best accuracy, use QAT in PyTorch, then export and quantize in ONNX if needed.

---

## Best Practices

- Always validate accuracy after quantization.
- Use representative calibration data.
- Profile inference speed and memory usage.
- Test on target hardware.
- Consider mixed precision (FP16 + INT8) for some platforms.
- Use model fusion (Conv+BN+ReLU) before quantization.

---

## Troubleshooting

- **Accuracy drop:** Use QAT, check calibration data, try different quantization schemes.
- **Unsupported layers:** Some custom layers may not be quantizable; replace or rewrite.
- **Deployment errors:** Check ONNX opset version, hardware compatibility, and runtime logs.

---

## References

- [PyTorch Quantization Docs](https://pytorch.org/docs/stable/quantization.html)
- [ONNX Runtime Quantization](https://onnxruntime.ai/docs/performance/quantization.html)
- [Ultralytics Model Export](https://docs.ultralytics.com/modes/export/)
- [TensorRT Quantization](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#quantization)
- [OpenVINO Quantization](https://docs.openvino.ai/latest/openvino_docs_MO_DG_prepare_model_convert_model_Quantization.html)

---

For further help, consult the official documentation or community forums for PyTorch and ONNX.

---

## EWQ and FastEWQ Quantization

### What is EWQ?

**Equal Weight Quantization (EWQ)** is a quantization method that divides the weight distribution into equal-sized bins and assigns quantized values based on bin centers. This approach aims to minimize quantization error by ensuring each bin contains an equal number of weights, rather than equal value ranges.

**Key Points:**

- EWQ is a non-uniform quantization technique.
- It is especially useful for weight distributions with heavy tails or outliers.
- EWQ can improve accuracy over uniform quantization in some cases.

**Workflow:**

1. Sort all weights in a layer.
2. Divide them into N bins, each with the same number of weights.
3. Assign each weight the value of its bin center.
4. Store bin centers and assignments for inference.

### What is FastEWQ?

**FastEWQ** is an optimized version of EWQ designed for speed and scalability. It uses efficient algorithms (e.g., histogram-based or approximate sorting) to quickly assign weights to bins, making it suitable for large models and real-time applications.

**Key Points:**

- FastEWQ reduces computation time compared to standard EWQ.
- It may use heuristics or parallelization for bin assignment.
- Accuracy is similar to EWQ, but with much faster quantization.

**Workflow:**

1. Estimate bin edges using histograms or sampling.
2. Assign weights to bins in parallel or with vectorized operations.
3. Quantize weights as in EWQ.

### Practical Notes

- EWQ and FastEWQ are mostly used in research and custom deployment pipelines.
- They are not natively supported in PyTorch or ONNX, but can be implemented as custom quantization functions.
- For most users, standard PTQ/QAT is sufficient, but EWQ/FastEWQ may help in cases where uniform quantization causes large accuracy drops.

### References

- [Equal Weight Quantization (arXiv)](https://arxiv.org/abs/2106.03561)
- [FastEWQ: Efficient Non-Uniform Quantization (arXiv)](https://arxiv.org/abs/2302.06675)

---

## Guide: ONNX Model Quantization with EWQ and FastEWQ

EWQ and FastEWQ are not natively supported in ONNX Runtime, but you can implement them as custom quantization steps before exporting or after loading an ONNX model. Below is a practical guide and example for applying EWQ/FastEWQ to ONNX models.

### Step 1: Load ONNX Model and Extract Weights

Use `onnx` and `numpy` to manipulate model weights.

```python
import onnx
import numpy as np

def get_weight_tensors(onnx_model):
   weights = {}
   for tensor in onnx_model.graph.initializer:
      arr = np.frombuffer(tensor.raw_data, dtype=np.float32).reshape(tuple(tensor.dims))
      weights[tensor.name] = arr
   return weights

model = onnx.load('model.onnx')
weights = get_weight_tensors(model)
```

### Step 2: Apply EWQ Quantization

Sort weights, bin into equal-count bins, and assign bin centers.

```python
def ewq_quantize(arr, num_bins=256):
   flat = arr.flatten()
   sorted_idx = np.argsort(flat)
   bins = np.array_split(sorted_idx, num_bins)
   centers = [np.mean(flat[bin]) for bin in bins]
   quantized = np.zeros_like(flat)
   for i, bin in enumerate(bins):
      quantized[bin] = centers[i]
   return quantized.reshape(arr.shape)

# Quantize all weights
for name, arr in weights.items():
   weights[name] = ewq_quantize(arr, num_bins=256)
```

### Step 3: Apply FastEWQ Quantization (Efficient Binning)

Use histogram-based binning for speed.

```python
def fastewq_quantize(arr, num_bins=256):
   flat = arr.flatten()
   hist, bin_edges = np.histogram(flat, bins=num_bins)
   bin_indices = np.digitize(flat, bin_edges[:-1])
   centers = [(flat[bin_indices == i]).mean() if np.any(bin_indices == i) else 0 for i in range(1, num_bins+1)]
   quantized = np.array([centers[i-1] for i in bin_indices])
   return quantized.reshape(arr.shape)

# Quantize all weights
for name, arr in weights.items():
   weights[name] = fastewq_quantize(arr, num_bins=256)
```

### Step 4: Update ONNX Model with Quantized Weights

Write quantized weights back to the ONNX model.

```python
for tensor in model.graph.initializer:
   arr = weights[tensor.name].astype(np.float32)
   tensor.raw_data = arr.tobytes()
onnx.save(model, 'model-ewq.onnx')
```

### Step 5: Validate Quantized Model

Test the quantized ONNX model for accuracy and performance using ONNX Runtime.

```python
import onnxruntime as ort
session = ort.InferenceSession('model-ewq.onnx')
outputs = session.run(None, {'input': input_data})
# Compare outputs to original model
```

---

**Notes:**

- Adjust `num_bins` for desired quantization granularity.
- EWQ/FastEWQ can be adapted for activations as well as weights.
- Always validate accuracy after quantization.

For more advanced usage, see the referenced papers and adapt binning strategies for your model and hardware.

---

## Public Library Support for EWQ and FastEWQ

Currently, there are no widely adopted public libraries or official ONNX/PyTorch toolkits that directly implement EWQ or FastEWQ quantization. These methods are mainly found in academic research and may require custom implementation based on published algorithms.

**What to do:**

- Use the example code above as a starting point for your own implementation.
- Check the referenced papers for algorithm details and possible open-source code links.
- Monitor GitHub and PyPI for new projects, as support may emerge in the future.

For standard quantization (uniform, dynamic, QAT), use PyTorch and ONNX Runtime built-in APIs.
