# analyze_onnx_model.py
import time

import numpy as np
import onnx
import onnxruntime as ort


def analyze_onnx_model(model_path):
    """Comprehensive analysis of your ONNX model"""

    # Load and validate the model
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    print("✅ ONNX model is valid")

    # Model metadata
    print(f"\n📊 Model Information:")
    print(f"IR Version: {model.ir_version}")
    print(f"Producer: {model.producer_name} {model.producer_version}")
    print(f"Domain: {model.domain}")
    print(f"Model Version: {model.model_version}")

    # Graph information
    graph = model.graph
    print(f"\n🔗 Graph Structure:")
    print(f"Nodes: {len(graph.node)}")
    print(f"Inputs: {len(graph.input)}")
    print(f"Outputs: {len(graph.output)}")
    print(f"Initializers: {len(graph.initializer)}")

    # Input/Output details
    print(f"\n📥 Input Information:")
    for i, input_tensor in enumerate(graph.input):
        if input_tensor.name not in [init.name for init in graph.initializer]:
            shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
            print(f"  Input {i}: {input_tensor.name}")
            print(f"    Shape: {shape}")
            print(f"    Type: {input_tensor.type.tensor_type.elem_type}")

    print(f"\n📤 Output Information:")
    for i, output_tensor in enumerate(graph.output):
        shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
        print(f"  Output {i}: {output_tensor.name}")
        print(f"    Shape: {shape}")
        print(f"    Type: {output_tensor.type.tensor_type.elem_type}")

    # Operators used
    ops = {}
    for node in graph.node:
        if node.op_type in ops:
            ops[node.op_type] += 1
        else:
            ops[node.op_type] = 1

    print(f"\n⚙️ Operations Used:")
    for op, count in sorted(ops.items()):
        print(f"  {op}: {count}")

    # Test with ONNX Runtime
    print(f"\n🚀 ONNX Runtime Test:")
    session = ort.InferenceSession(model_path)

    # Available providers
    print(f"Available Providers: {ort.get_available_providers()}")
    print(f"Current Providers: {session.get_providers()}")

    # Performance test
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_name = session.get_outputs()[0].name

    # Handle dynamic shapes
    if any(dim is None or (isinstance(dim, str)) for dim in input_shape):
        input_shape = [1, 3, 640, 640]  # Default for YOLO

    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Warm-up runs
    for _ in range(10):
        _ = session.run(None, {input_name: dummy_input})

    # Timing test

    start_time = time.time()
    num_runs = 1000
    for _ in range(num_runs):
        outputs = session.run(None, {input_name: dummy_input})
        outputs_np = np.array(outputs)
    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs * 1000  # ms
    print(f"Average inference time: {avg_time:.3f} ms")
    print(f"Throughput: {1000/avg_time:.1f} inferences/second")

    return {
        'input_name': input_name,
        'input_shape': input_shape,
        'output_name': output_name,
        'output_shape': outputs_np[0].shape,
        'avg_inference_time_ms': avg_time
    }


model_info = analyze_onnx_model('./models/yolo_chicken_egg_infer.quant.onnx')
print(f"\n📋 Summary: {model_info}")
