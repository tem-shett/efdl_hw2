import os
import random
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

def benchmark_fn(rank, world_size):
    # 1. Умный выбор устройства
    # Если GPU всего один, оба процесса сядут на cuda:0 (это создаст нагрузку, но сработает)
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        device_id = rank % n_gpus 
        device = torch.device(f"cuda:{device_id}")
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    # Инициализация
    try:
        dist.init_process_group(backend, rank=rank, world_size=world_size)
    except Exception as e:
        print(f"Init failed: {e}")
        return

    # Параметры
    hid_dims = [128, 256, 512, 1024]
    batch_sizes = [32, 64]
    
    # Фиксированное количество итераций гарантирует, что процессы не разминутся
    NUM_ITERS = 100
    WARMUP = 10

    if rank == 0:
        print(f"\n{'Dim':<8} | {'Batch':<8} | {'Time (ms)':<15} | {'Mem (MB)':<10}")
        print("-" * 50)

    for d in hid_dims:
        for b in batch_sizes:
            try:
                # Очистка памяти перед каждым новым размером
                torch.cuda.empty_cache()
                
                model = nn.SyncBatchNorm(d, affine=True).to(device)
                model.train()
                # Вход [Batch, Channels, Length] - стандарт для 1D
                x = torch.randn(b, d, 128, device=device, requires_grad=True)
                
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

                # --- Warmup ---
                # Прогоняем "вхолостую", чтобы инициализировать буферы NCCL
                for _ in range(WARMUP):
                    optimizer.zero_grad()
                    out = model(x)
                    out.sum().backward()
                    optimizer.step()
                
                # Синхронизация перед стартом таймера
                torch.cuda.synchronize(device)
                dist.barrier()
                
                # --- Benchmark ---
                start_time = time.perf_counter()
                
                for _ in range(NUM_ITERS):
                    optimizer.zero_grad()
                    # Важно: SyncBN требует коммуникации и на forward, и на backward
                    out = model(x)
                    loss = out.sum()
                    loss.backward()
                    optimizer.step()
                
                # Синхронизация после окончания
                torch.cuda.synchronize(device)
                dist.barrier() # Ждем, пока все добегут
                
                end_time = time.perf_counter()
                avg_time_ms = (end_time - start_time) / NUM_ITERS * 1000

                # Замер памяти
                mem = torch.cuda.max_memory_allocated(device) / 1024 / 1024

                if rank == 0:
                    print(f"{d:<8} | {b:<8} | {avg_time_ms:<15.4f} | {mem:<10.2f}")
                    
            except RuntimeError as e:
                if rank == 0:
                    print(f"OOM or Error at {d}x{b}: {e}")
                break

    dist.destroy_process_group()

if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(random.randint(25000, 30000))
    
    # Если на Kaggle 1 GPU, лучше использовать world_size=2 для тестов, 
    # но они будут делить одну видеокарту.
    world_size = 2
    
    # Уменьшаем таймаут, чтобы не ждать вечно при ошибке
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

    mp.spawn(benchmark_fn, args=(world_size,), nprocs=world_size, join=True)