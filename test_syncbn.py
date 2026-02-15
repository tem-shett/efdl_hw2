import torch
import pytest
import os
import torch.distributed as dist
import torch.nn as nn
import numpy as np
import random
from syncbn import SyncBatchNorm

def worker_fn(rank, world_size, port, input: torch.Tensor, grad_output, result_queue):
    bn = SyncBatchNorm(num_features=input.shape[1], eps=1e-7, momentum=0.1)
    bn.train()
    input = input.clone().detach()
    input.requires_grad = True
    output = bn(input)
    output.backward(grad_output)

    result_queue.put({
        'rank': rank,
        'output': output.detach().numpy(),
        'grad_input': input.grad.detach().numpy()
    })


def init_process(rank, world_size, port, input, grad_output, result_queue):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    worker_fn(rank, world_size, port, input, grad_output, result_queue)
    dist.destroy_process_group()


@pytest.mark.parametrize("num_workers", [1, 4])
@pytest.mark.parametrize("hidden_size", [128, 256, 512, 1024])
@pytest.mark.parametrize("batch_size", [32, 64])
def test_batchnorm(num_workers, hidden_size, batch_size):
    # Verify that the implementation of SyncBatchNorm gives the same results (both for outputs
    # and gradients with respect to input) as torch.nn.BatchNorm1d on a variety of inputs.

    # This can help you set up the worker processes. Child processes launched with `spawn` can still run
    # torch.distributed primitives, but you can also communicate their outputs back to the main process to compare them
    # with the outputs of a non-synchronous BatchNorm.
    torch.manual_seed(179)
    
    input = torch.randn((batch_size, hidden_size), dtype=torch.float32)
    input.requires_grad = True
    bn = nn.BatchNorm1d(hidden_size, eps=1e-7, momentum=0.1, affine=False)
    bn.train()
    output = bn(input)
    output.retain_grad()
    loss = output[:batch_size // 2].sum()
    loss.backward()
    grad_output = output.grad.detach()
    grad_input = input.grad.detach()
    output = output.detach()
    input = input.detach()

    chunk_size = batch_size // num_workers
    inputs = list(input.detach().split(chunk_size, dim=0))
    grad_outputs = list(grad_output.split(chunk_size, dim=0))
    outputs = list(output.detach().split(chunk_size, dim=0))
    grad_inputs = list(grad_input.split(chunk_size, dim=0))

    ctx = torch.multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    port = random.randint(25000, 30000)
    proccesses = []
    
    for rank in range(num_workers):
        p = ctx.Process(target=init_process, args=(rank, num_workers, port, inputs[rank], grad_outputs[rank], result_queue))
        p.start()
        proccesses.append(p)
    
    
    for _ in range(num_workers):
        res = result_queue.get()
        rank = res['rank']
        torch.testing.assert_close(torch.from_numpy(res['output']), outputs[rank], rtol=0, atol=1e-3, msg="Forward mismatch")
        torch.testing.assert_close(torch.from_numpy(res['grad_input']), grad_inputs[rank], rtol=0, atol=1e-3, msg="Backward mismatch")
    
    for p in proccesses:
        p.join()
