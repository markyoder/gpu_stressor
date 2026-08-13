'''
# Code mostly by CLAUDE
'''

import argparse
import time
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--duration", type=int, default=25)
parser.add_argument("--n", type=int, default=8192)
args = parser.parse_args()

a = torch.randn(args.n, args.n, device="cuda", dtype=torch.bfloat16)
b = torch.randn(args.n, args.n, device="cuda", dtype=torch.bfloat16)

for _ in range(10):
    torch.mm(a, b)
torch.cuda.synchronize()

t0 = time.perf_counter()
while time.perf_counter() - t0 < args.duration:
    for _ in range(20):
        torch.mm(a, b)
    torch.cuda.synchronize()