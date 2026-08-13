'''
# Code mostly by CLAUDE
'''

import argparse
import time
import torch

def stress_gpu_mm(n=8192, duration=25, n_steps_1=10, n_steps_2=20):
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)

    for _ in range(int(n_steps_1)):
        torch.mm(a, b)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    while time.perf_counter() - t0 < duration:
        for _ in range(int(n_steps_2)):
            torch.mm(a, b)
        torch.cuda.synchronize()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=25)
    parser.add_argument("--n", type=int, default=8192)
    args = parser.parse_args()
    #
    z = stress_gpu_mm(args.n, args.duration)
