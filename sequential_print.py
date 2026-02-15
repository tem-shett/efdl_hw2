import os

import torch
import torch.distributed as dist


def run_sequential(rank, size, num_iter=10):
    """
    Prints the process rank sequentially in two orders over `num_iter` iterations,
    separating the output for each iteration by `---`.
    Example (3 processes, num_iter=2):
    ```
    Process 0
    Process 1
    Process 2
    Process 2
    Process 1
    Process 0
    ---
    Process 0
    Process 1
    Process 2
    Process 2
    Process 1
    Process 0
    ```
    """
    tensor = torch.tensor(0)
    for iteration in range(num_iter):
        if rank == 0:
            print(f"Process {rank}")
            dist.send(tensor=tensor, dst=1)
            dist.recv(tensor=tensor, src=1)
            print(f"Process {rank}")
            if iteration != num_iter - 1:
                print("---")
        else:
            dist.recv(tensor=tensor, src=rank-1)
            print(f"Process {rank}")
            if rank != size - 1:
                dist.send(tensor=tensor, dst=rank+1)
                dist.recv(tensor=tensor, src=rank+1)
            print(f"Process {rank}")
            dist.send(tensor=tensor, dst=rank-1)


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(rank=local_rank, backend="gloo")

    run_sequential(local_rank, dist.get_world_size())
