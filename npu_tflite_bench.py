"""
MobileNetV2 INT8: CPU (TFLite XNNPACK) vs Hexagon HTP (QNN TFLite Delegate)
Rubik Pi 3 - QCS6490 Hexagon V68
"""
import time
import os
import ctypes
import numpy as np
from ai_edge_litert.interpreter import Interpreter, load_delegate

MODEL = "mobilenet_v2_quant.tflite"
N_WARMUP = 3
N_BENCH = 20

def bench(interp, x_in, x_out, n_warmup, n_bench):
    inp_detail = interp.get_input_details()[0]
    out_detail = interp.get_output_details()[0]
    interp.set_tensor(inp_detail['index'], x_in)

    for _ in range(n_warmup):
        interp.invoke()

    times = []
    for _ in range(n_bench):
        interp.set_tensor(inp_detail['index'], x_in)
        t0 = time.perf_counter()
        interp.invoke()
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    return times

# Input: uint8, shape [1, 224, 224, 3]
x_in = np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8)

# ── 1. CPU baseline (XNNPACK delegate, the default) ──────────────────────────
print("=" * 60)
print("MobileNetV2 INT8 — Rubik Pi 3 / QCS6490")
print("=" * 60)
print(f"\n[1] CPU (ai-edge-litert XNNPACK delegate)")
cpu_interp = Interpreter(MODEL, num_threads=8)
cpu_interp.allocate_tensors()
cpu_times = bench(cpu_interp, x_in, None, N_WARMUP, N_BENCH)
print(f"    Median: {cpu_times[N_BENCH//2]:.2f} ms | Min: {cpu_times[0]:.2f} ms | p90: {cpu_times[int(N_BENCH*0.9)]:.2f} ms")

# ── 2. HTP via QNN TFLite Delegate ───────────────────────────────────────────
print(f"\n[2] Hexagon HTP (libQnnTFLiteDelegate.so)")
QNN_DELEGATE = "/usr/lib/libQnnTFLiteDelegate.so"

try:
    # Delegate options for HTP INT8 (values must be strings matching qtld-net-run CLI format)
    delegate_options = {
        "backend":              "htp",
        "htp_performance_mode": "3",   # 3 = HighPerformance
        "log_level":            "4",   # 4 = error only
    }
    qnn_delegate = load_delegate(QNN_DELEGATE, options=delegate_options)
    htp_interp = Interpreter(MODEL, experimental_delegates=[qnn_delegate])
    htp_interp.allocate_tensors()

    htp_times = bench(htp_interp, x_in, None, N_WARMUP, N_BENCH)
    htp_median = htp_times[N_BENCH // 2]
    cpu_median = cpu_times[N_BENCH // 2]

    print(f"    Median: {htp_median:.2f} ms | Min: {htp_times[0]:.2f} ms | p90: {htp_times[int(N_BENCH*0.9)]:.2f} ms")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  CPU  (XNNPACK, 8 threads) : {cpu_median:.1f} ms")
    print(f"  HTP  (Hexagon V68 INT8)   : {htp_median:.1f} ms")
    speedup = cpu_median / htp_median
    print(f"  NPU speedup               : {speedup:.2f}x")

    # Verify correctness
    cpu_interp.set_tensor(cpu_interp.get_input_details()[0]['index'], x_in)
    cpu_interp.invoke()
    out_cpu = cpu_interp.get_tensor(cpu_interp.get_output_details()[0]['index'])

    htp_interp.set_tensor(htp_interp.get_input_details()[0]['index'], x_in)
    htp_interp.invoke()
    out_htp = htp_interp.get_tensor(htp_interp.get_output_details()[0]['index'])

    max_diff = int(np.abs(out_cpu.astype(np.int32) - out_htp.astype(np.int32)).max())
    top1_cpu = int(np.argmax(out_cpu))
    top1_htp = int(np.argmax(out_htp))
    print(f"\n  Top-1 CPU={top1_cpu}, HTP={top1_htp}  |  Max INT8 diff: {max_diff}")

except Exception as e:
    print(f"    HTP delegate failed: {e}")
    import traceback; traceback.print_exc()
