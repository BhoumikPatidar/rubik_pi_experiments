"""
MobileNetV2: CPU (PyTorch) vs HTP NPU (onnxruntime-qnn QNN EP)
Rubik Pi 3 - QCS6490 Hexagon V68 HTP
"""
import time
import os
import numpy as np
import torch
import torchvision.models as models
import onnxruntime as ort

ONNX_PATH = "mobilenetv2.onnx"
INPUT_SHAPE = (1, 3, 224, 224)
N_WARMUP = 5
N_BENCH = 30

# ── 1. Export to ONNX ────────────────────────────────────────────────────────
print("=" * 60)
print("Step 1: Export MobileNetV2 → ONNX")
model_pt = models.mobilenet_v2(weights=None)
model_pt.eval()
dummy = torch.randn(*INPUT_SHAPE)

torch.onnx.export(
    model_pt,
    dummy,
    ONNX_PATH,
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
)
print(f"  Saved → {ONNX_PATH} ({os.path.getsize(ONNX_PATH) / 1e6:.1f} MB)")

# ── 2. CPU baseline with onnxruntime ─────────────────────────────────────────
print("\nStep 2: CPU baseline via onnxruntime CPUExecutionProvider")
sess_cpu = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
x_np = np.random.randn(*INPUT_SHAPE).astype(np.float32)
inp = {"input": x_np}

for _ in range(N_WARMUP):
    sess_cpu.run(None, inp)

cpu_times = []
for _ in range(N_BENCH):
    t0 = time.perf_counter()
    sess_cpu.run(None, inp)
    cpu_times.append((time.perf_counter() - t0) * 1000)

cpu_times.sort()
print(f"  Median: {cpu_times[N_BENCH//2]:.2f} ms | Min: {cpu_times[0]:.2f} ms | p90: {cpu_times[int(N_BENCH*0.9)]:.2f} ms")

# ── 3. HTP NPU via QNN Execution Provider ────────────────────────────────────
print("\nStep 3: HTP NPU via QNNExecutionProvider (FP16)")

# QNN EP options for HTP backend
qnn_options = {
    "backend_path": "/usr/lib/libQnnHtp.so",
    "htp_performance_mode": "burst",
    "htp_graph_finalization_optimization_level": "3",
    "enable_htp_fp16_precision": "1",  # FP16 on HTP (no INT8 quantization needed)
}

try:
    sess_npu = ort.InferenceSession(
        ONNX_PATH,
        providers=[("QNNExecutionProvider", qnn_options)],
    )

    active_providers = sess_npu.get_providers()
    print(f"  Active providers: {active_providers}")

    # Check if QNN EP actually took it
    if "QNNExecutionProvider" not in active_providers:
        print("  WARNING: QNN EP not active — fell back to CPU")

    for _ in range(N_WARMUP):
        sess_npu.run(None, inp)

    npu_times = []
    for _ in range(N_BENCH):
        t0 = time.perf_counter()
        sess_npu.run(None, inp)
        npu_times.append((time.perf_counter() - t0) * 1000)

    npu_times.sort()
    npu_median = npu_times[N_BENCH // 2]
    cpu_median = cpu_times[N_BENCH // 2]

    print(f"  Median: {npu_median:.2f} ms | Min: {npu_times[0]:.2f} ms | p90: {npu_times[int(N_BENCH*0.9)]:.2f} ms")

    print("\n" + "=" * 60)
    print("Results summary")
    print("=" * 60)
    print(f"  CPU (onnxruntime)   median: {cpu_median:.1f} ms")
    print(f"  HTP NPU (QNN EP)    median: {npu_median:.1f} ms")
    if npu_median < cpu_median:
        print(f"  Speedup: {cpu_median / npu_median:.2f}x  (NPU is faster)")
    else:
        print(f"  Speedup: {cpu_median / npu_median:.2f}x  (CPU is faster — model may be FP32-heavy for this HTP config)")

    # Verify outputs match (sanity check)
    out_cpu = sess_cpu.run(None, inp)[0]
    out_npu = sess_npu.run(None, inp)[0]
    max_diff = np.abs(out_cpu - out_npu).max()
    print(f"\n  Output max diff CPU vs NPU: {max_diff:.4f}")

except Exception as e:
    print(f"  QNN EP failed: {e}")
    print("  Hint: check that /usr/lib/libQnnHtp.so exists and ADSP is accessible")
