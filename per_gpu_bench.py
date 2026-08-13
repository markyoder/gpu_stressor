'''
# Simple stress-test for GPUs. Note this is best run in conjunction with nvidia-smi to collect additional
# data, especially temperature.
# Those data can be collected directly using (import pynvml) but there might be some confusion over the GPU
# naming -- torch is likely to see only the devices it's told about or allocated by SLURM, and so will call them
# cuda:0, cuda:1, etc. Those names might be the same as the machine level names, so "soft" cuda:0 might point to
# cuda:3 (cuda:{not 1}), so it might be better to just stick to running nvidia-smi separately.
'''
#
import time
import sys
import torch
import argparse

def stress_gpu_mm(duration=25, n=8192, log_interval=1.0, log_filename='mm_stress_log.csv'):
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    #
    # Min logging interval, in seconds
    #log_interval = 2
    #
    if log_interval == 0:
        log_interval = None

    for _ in range(10):
        torch.mm(a, b)
    torch.cuda.synchronize()
    #
    flops_per_matmul = 2 * n**3
    batch_size = 20

    #log = open("stress_log.csv", "w")
    #
    print(f'** Stress test MM, for duration={duration}, n={n}, log_interval={log_interval}, log_filename={log_filename}')
    #
    with open(log_filename, "w") as log:
        log.write("elapsed_s,tflops,pct_of_first\n")
        #
        first = None
        t0 = time.perf_counter()
        prev_elapsed_total = 0.
        while time.perf_counter() - t0 < duration:
            t1 = time.perf_counter()
            for _ in range(batch_size):
                torch.mm(a, b)
            torch.cuda.synchronize()
            elapsed_batch = time.perf_counter() - t1
            elapsed_total = time.perf_counter() - t0
            #
            tflops = (flops_per_matmul * batch_size / elapsed_batch) / 1e12
            if first is None:
                first = tflops
            pct = 100 * tflops / first
            #
            # control output volume:
            #if int(elapsed_total) > int(prev_elapsed_total) or output_all:
            if log_interval is None or (( (elapsed_total - prev_elapsed_total) > log_interval) ):
                #
                prev_elapsed_total=elapsed_total
                #
                line = f"t={elapsed_total:6.1f}s  {tflops:7.1f} TFLOP/s  ({pct:5.1f}% of first)"
                sys.stdout.write("\n" + line.ljust(60))
                sys.stdout.flush()
                log.write(f"{elapsed_total:.2f},{tflops:.2f},{pct:.2f}\n")
                log.flush()

        print()  # final newline so the shell prompt doesn't land on top of the last line

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=25)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--log_interval", type=float, default=1.0)
    parser.add_argument("--log_filename", type=str, default="stress_log.csv")
    args = parser.parse_args()
    #
    duration = args.duration
    n = args.n
    log_interval = args.log_interval
    log_filename = args.log_filename
    #
    # just in case None gets passed
    #n = 8192
#    n = n or 8192
#    duration = duration or 25
    #
    z = stress_gpu_mm(duration=duration, n=n, log_interval=log_interval, log_filename=log_filename)

    