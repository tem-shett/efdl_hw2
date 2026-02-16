import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.benchmark as benchmark

def benchmark_fn(rank, world_size):
    # Настройка окружения
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")
    
    # Параметры бенчмарка
    hid_dims = [128, 256, 512, 1024]
    batch_sizes = [32, 64]
    results = []

    for d in hid_dims:
        for b in batch_sizes:
            # Инициализация слоя и данных
            # nn.SyncBatchNorm требует наличия группы процессов
            model = nn.SyncBatchNorm(d, affine=True).to(device)
            x = torch.randn(b, d, device=device, requires_grad=True)
            
            # Сбрасываем статистику памяти перед замером
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)

            # Описываем задачу для Timer
            # Мы замеряем forward + backward вместе
            timer = benchmark.Timer(
                stmt='model(x).sum().backward()',
                setup='model.zero_grad()',
                globals={'model': model, 'x': x},
                num_threads=1,
                label='SyncBatchNorm Performance',
                sub_label=f'Dim: {d}, Batch: {b}',
                description='Forward + Backward'
            )
            
            # Проводим замер
            measurement = timer.blocked_autorange(min_run_time=1.0)
            
            # Замер пиковой памяти
            max_mem = 0
            if device.type == 'cuda':
                max_mem = torch.cuda.max_memory_allocated(device) / 1024 / 1024  # в MB

            if rank == 0:
                results.append((d, b, measurement, max_mem))

    # Вывод результатов в красивом виде
    if rank == 0:
        print(f"\n{'Dimension':<10} | {'Batch':<8} | {'Time (ms)':<15} | {'Memory (MB)':<12}")
        print("-" * 55)
        for d, b, m, mem in results:
            # m.mean возвращает время в секундах
            time_ms = m.mean * 1000
            print(f"{d:<10} | {b:<8} | {time_ms:<15.4f} | {mem:<12.2f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    # Запускаем 2 процесса для эмуляции распределенной работы
    # Даже на одном GPU или CPU это заставит SyncBN выполнять сетевую синхронизацию
    world_size = 2
    mp.spawn(benchmark_fn, args=(world_size,), nprocs=world_size, join=True)