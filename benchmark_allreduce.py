import os
import torch
import time
import random
import tracemalloc
import pandas as pd
from allreduce import ring_allreduce, butterfly_allreduce
import torch.distributed as dist
import torch.multiprocessing as mp

def benchmark_worker(rank, world_size, port, vector_sizes, results_queue):
    os.environ['MASTER_ADDR'] = "127.0.0.1"
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    def fn(allreduce_fn, reduce_name):
        data = data_orig.clone()
        dist.barrier()
        tracemalloc.start()
        start_time = time.perf_counter()
        
        allreduce_fn(data)
        
        dist.barrier()
        end_time = time.perf_counter()
        _, mem_used = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        if rank == 0:
            results_queue.put({
                'type': reduce_name,
                'workers': world_size,
                'size': n_elements,
                'time(ms)': (end_time - start_time) * 1000,
                'mem(KB)': mem_used / 1024
            })

    for n_elements in vector_sizes:
        data_orig = torch.randn(n_elements)
        if n_elements % world_size == 0:
            fn(lambda data: butterfly_allreduce(data, rank, world_size), "butterfly")
        fn(lambda data: ring_allreduce(data, rank, world_size), "my ring")
        fn(lambda data: dist.all_reduce(data, op=dist.ReduceOp.SUM), "torch")

    dist.destroy_process_group()

def run_benchmarks():
    vector_sizes = [1000, 10000, 100000]
    worker_counts = [2, 4, 8, 16]

    all_data = []

    for ws in worker_counts:
        port = random.randint(25000, 30000)
        print(f"{ws} workers")
        ctx = mp.get_context('spawn')
        q = ctx.Queue()
        mp.spawn(benchmark_worker, args=(ws, port, vector_sizes, q), nprocs=ws, join=True)
        while not q.empty():
            all_data.append(q.get())

    return pd.DataFrame(all_data)

if __name__ == "__main__":
    run_benchmarks()
