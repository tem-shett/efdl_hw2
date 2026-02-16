import os
import random
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from syncbn import SyncBatchNorm

def benchmark_fn(rank, world_size):
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    hid_dims = [128, 256, 512, 1024]
    batch_sizes = [32, 64]
    
    NUM_ITERS = 100
    WARMUP = 10

    if rank == 0:
        print(f"\n{'Dim':<8} | {'Batch':<8} | {'Time (ms)':<15} | {'Mem (MB)':<10}") # line from ChatGPT
        print("-" * 50)

    for d in hid_dims:
        for b in batch_sizes:
            torch.cuda.empty_cache()
            model = SyncBatchNorm(d).to(device)
            model.train()
            x = torch.randn(b, d, device=device, requires_grad=True)
            for _ in range(WARMUP):
                out = model(x)
                out.sum().backward()
            torch.cuda.synchronize(device)
            dist.barrier()
            start_time = time.perf_counter()
            for _ in range(NUM_ITERS):
                out = model(x)
                loss = out.sum()
                loss.backward()
            torch.cuda.synchronize(device)
            dist.barrier()
            end_time = time.perf_counter()
            avg_time_ms = (end_time - start_time) / NUM_ITERS * 1000

            mem = torch.cuda.max_memory_allocated(device) / 1024**2

            if rank == 0:
                print(f"{d:<8} | {b:<8} | {avg_time_ms:<15.4f} | {mem:<10.2f}") # line from ChatGPT

    dist.destroy_process_group()

if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(random.randint(25000, 30000))
    
    world_size = 2
    mp.spawn(benchmark_fn, args=(world_size,), nprocs=world_size, join=True)