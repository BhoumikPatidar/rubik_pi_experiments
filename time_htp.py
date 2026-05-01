"""Time HTP inference using delta method via subprocess."""
import subprocess, time, os, sys

env = os.environ.copy()
env["ADSP_LIBRARY_PATH"] = "/usr/lib;/usr/lib/rfsa/adsp;/dsp"
env["LD_LIBRARY_PATH"] = "/usr/lib:" + env.get("LD_LIBRARY_PATH", "")

base_cmd = [
    "qtld-net-run",
    "--model", "mobilenet_v2_quant.tflite",
    "--backend", "htp",
    "--htp_precision", "0",
    "--htp_performance_mode", "3",
    "--profiling", "0",
    "--cache_dir", "htp_cache",
    "--model_token", "mnv2_quant",
    "--verbose", "0",
]

def write_input_list(n, path):
    with open(path, "w") as f:
        for _ in range(n): f.write("npu_inputs/input_quant.bin\n")

OVERHEAD_N = 1
BENCH_N = 21

write_input_list(OVERHEAD_N, "/tmp/inp_over.txt")
write_input_list(BENCH_N, "/tmp/inp_bench.txt")

# Warmup run (prime CDSP)
print("Priming CDSP...", flush=True)
subprocess.run(base_cmd + ["--input", "/tmp/inp_over.txt"],
               capture_output=True, env=env)

# Measure overhead (1 inference)
print("Timing overhead run (1 inference)...", flush=True)
times_over = []
for i in range(3):
    t0 = time.perf_counter()
    subprocess.run(base_cmd + ["--input", "/tmp/inp_over.txt"],
                   capture_output=True, env=env)
    times_over.append(time.perf_counter() - t0)
T_over = sorted(times_over)[1] * 1000  # median of 3

# Measure 21 inferences
print("Timing 21-inference run...", flush=True)
times_bench = []
for i in range(3):
    t0 = time.perf_counter()
    subprocess.run(base_cmd + ["--input", "/tmp/inp_bench.txt"],
                   capture_output=True, env=env)
    times_bench.append(time.perf_counter() - t0)
T_bench = sorted(times_bench)[1] * 1000  # median of 3

per_inf = (T_bench - T_over) / (BENCH_N - OVERHEAD_N)

print(f"\n1-run  total: {T_over:.0f} ms")
print(f"21-run total: {T_bench:.0f} ms")
print(f"Per-inference HTP (delta): {per_inf:.1f} ms")
print(f"\nCPU (XNNPACK, 8T):  9.1 ms  (from ai-edge-litert benchmark)")
print(f"HTP (Hexagon V68):  {per_inf:.1f} ms")
if per_inf < 9.1:
    print(f"NPU speedup: {9.1/per_inf:.2f}x")
else:
    print(f"CPU faster: {per_inf/9.1:.2f}x slower (HTP process overhead dominates)")
