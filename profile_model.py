import time
import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity, record_function

# ── Model setup ──────────────────────────────────────────────────────────────
print(f"PyTorch {torch.__version__}")
print(f"Device: CPU | Threads: {torch.get_num_threads()}")

model = models.mobilenet_v2(weights=None)
model.eval()

x = torch.randn(1, 3, 224, 224)

# ── Warmup ────────────────────────────────────────────────────────────────────
print("\nWarming up (5 runs)...")
with torch.no_grad():
    for _ in range(5):
        _ = model(x)

# ── Simple latency benchmark ──────────────────────────────────────────────────
N = 20
print(f"Benchmarking latency ({N} runs)...")
times = []
with torch.no_grad():
    for _ in range(N):
        t0 = time.perf_counter()
        _ = model(x)
        times.append((time.perf_counter() - t0) * 1000)

times.sort()
print(f"  Median : {times[N//2]:.2f} ms")
print(f"  Min    : {times[0]:.2f} ms")
print(f"  Max    : {times[-1]:.2f} ms")
print(f"  p90    : {times[int(N*0.9)]:.2f} ms")

# ── torch.profiler ────────────────────────────────────────────────────────────
print("\nRunning torch.profiler (5 profiled steps)...")
with profile(
    activities=[ProfilerActivity.CPU],
    record_shapes=True,
    profile_memory=True,
    with_stack=False,
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./prof_logs"),
) as prof:
    with torch.no_grad():
        for step in range(5):
            with record_function("inference"):
                _ = model(x)
            prof.step()

# Top 20 ops by CPU time
print("\n── Top 20 ops by CPU time ──────────────────────────────────────────────")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

# Top 10 ops by self CPU memory
print("── Top 10 ops by self CPU memory ───────────────────────────────────────")
print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

# Trace already saved by tensorboard_trace_handler above
trace_path = "./prof_logs/"
print(f"\nChrome trace saved → {trace_path}*.pt.trace.json")
print("  Open with: chrome://tracing  (load the JSON file)")
print("\nDone.")
