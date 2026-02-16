import os
import random

import torch
import torch.distributed as dist
from torch.multiprocessing import Process


def init_process(rank, size, fn, master_port, backend="gloo"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


def butterfly_allreduce(send, rank, size):
    """
    Performs Butterfly All-Reduce over the process group. Modifies the input tensor in place.
    Args:
        send: torch.Tensor to be averaged with other processes.
        rank: Current process rank (in a range from 0 to size)
        size: Number of workers
    """

    buffer_for_chunk = torch.empty((size,send.shape[0]//size), dtype=torch.float)
    send = list(torch.chunk(send, size, dim=0))

    send_futures = []

    for i, elem in enumerate(send):
        if i != rank:
            send_futures.append(dist.isend(elem, i))

    recv_futures = []

    for i, elem in enumerate(buffer_for_chunk):
        if i != rank:
            recv_futures.append(dist.irecv(elem, i))
        else:
            elem.copy_(send[i])

    for future in recv_futures:
        future.wait()

    # compute the average
    torch.mean(buffer_for_chunk, dim=0, out=send[rank])

    for i in range(size):
        if i != rank:
            send_futures.append(dist.isend(send[rank], i))

    recv_futures = []

    for i, elem in enumerate(send):
        if i != rank:
            recv_futures.append(dist.irecv(elem, i))

    for future in recv_futures:
        future.wait()
    for future in send_futures:
        future.wait()


def ring_allreduce(send, rank, size):
    """
    Performs Ring All-Reduce over the process group. Modifies the input tensor in place.
    Args:
        send: torch.Tensor to be averaged with other processes.
        rank: Current process rank (in a range from 0 to size)
        size: Number of workers
    """
    chunks = list(torch.chunk(send, size, dim=0))

    for i in range(size - 1):
        send_future = dist.isend(chunks[(rank - i - 1) % size], (rank+1)%size)
        res = torch.empty_like(chunks[(rank - i - 2) % size])
        recv_future = dist.irecv(res, (rank-1)%size)
        recv_future.wait()
        send_future.wait()
        chunks[(rank - i - 2) % size] += res
    
    chunks[rank] /= size

    for i in range(size - 1):
        send_future = dist.isend(chunks[(rank - i) % size], (rank+1)%size)
        recv_future = dist.irecv(chunks[(rank - i - 1) % size], (rank-1)%size)
        recv_future.wait()
        send_future.wait()
    
    # send[:] = torch.concat(chunks)


def run_butterfly_allreduce(rank, size):
    """Simple point-to-point communication."""
    torch.manual_seed(rank)
    tensor = torch.randn((size,), dtype=torch.float)
    print("Rank ", rank, " has data ", tensor)
    butterfly_allreduce(tensor, rank, size)
    print("Rank ", rank, " has data ", tensor)

def run_ring_allreduce(rank, size):
    torch.manual_seed(rank)
    tensor = torch.randn((size * 5 + 1,), dtype=torch.float)
    garbage = torch.tensor(1)
    if rank != 0:
        dist.recv(garbage, rank - 1)
    print("Rank ", rank, " has data ", tensor)
    if rank != size - 1:
        dist.send(garbage, rank + 1)
    ring_allreduce(tensor, rank, size)
    if rank != 0:
        dist.recv(garbage, rank - 1)
    else:
        print("After ring allreduce:")
    print("Rank ", rank, " has data ", tensor)
    if rank != size - 1:
        dist.send(garbage, rank + 1)


if __name__ == "__main__":
    size = 5
    processes = []
    port = random.randint(25000, 30000)
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run_ring_allreduce, port))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
