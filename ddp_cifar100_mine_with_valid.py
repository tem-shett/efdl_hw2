import os
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import CIFAR100

from syncbn import SyncBatchNorm

torch.set_num_threads(1)


def init_process(local_rank, fn, backend="nccl"):
    """Initialize the distributed environment."""
    dist.init_process_group(backend, rank=local_rank)
    size = dist.get_world_size()
    fn(local_rank, size)


class Net(nn.Module):
    """
    A very simple model with minimal changes from the tutorial, used for the sake of simplicity.
    Feel free to replace it with EffNetV2-XL once you get comfortable injecting SyncBN into models programmatically.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(6272, 128)
        self.fc2 = nn.Linear(128, 100)
        self.bn1 = SyncBatchNorm(128)  # to be replaced with SyncBatchNorm

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return output


def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

def validation(rank, size, model, batch_size):
    model.eval()
    device = next(model.parameters()).device
    
    valid_dataset = CIFAR100(
        "./cifar",
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        ),
        download=True,
        train=False
    )

    shard_size = len(valid_dataset) // size

    scatter_inputs = None
    scatter_targets = None
    if rank == 0:
        inputs, targets = next(iter(DataLoader(valid_dataset, batch_size=len(valid_dataset))))
        scatter_inputs = list(inputs[:shard_size * size].to(device).chunk(size))
        scatter_targets = list(targets[:shard_size * size].to(device).chunk(size))

    input = torch.zeros((shard_size, *valid_dataset[0][0].shape), device=device)
    if isinstance(valid_dataset[0][1], int):
        targets = torch.zeros((shard_size,), dtype=torch.int64, device=device)
    else:
        targets = torch.zeros((shard_size, *valid_dataset[0][1].shape), dtype=torch.int64, device=device)
    dist.scatter(input, scatter_inputs, src=0)
    dist.scatter(targets, scatter_targets, src=0)

    correct = 0
    with torch.no_grad():
        for i in range(0, shard_size, batch_size):
            data = input[i:i+batch_size]
            target = targets[i:i+batch_size]
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    model.train()
    acc = torch.tensor(correct / shard_size, device=device)
    dist.reduce(acc, dst=0, op=dist.ReduceOp.SUM)
    if rank == 0:
        return acc.item() / size

def run_training(rank, size, gradient_accumulation=2):
    torch.manual_seed(0)

    if rank != 0:
        dist.barrier()
    dataset = CIFAR100(
        "./cifar",
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        ),
        download=True,
    )
    if rank == 0:
        dist.barrier()
    # where's the validation dataset?
    loader = DataLoader(dataset, sampler=DistributedSampler(dataset, size, rank), batch_size=64)

    model = Net()
    device = torch.device(f"cuda:{rank}")  # replace with "cuda" afterwards
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    num_batches = len(loader)

    start_time = time.time()
    torch.cuda.reset_peak_memory_stats(device)

    for _ in range(10):
        epoch_loss = torch.zeros((1,), device=device)

        sum_loss = 0
        sum_acc = 0
        sum_B = 0
        for i, datatarget in enumerate(loader):
            data, target = datatarget
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            epoch_loss += loss.detach()
            loss.backward()
            if (i + 1) % gradient_accumulation == 0 or i + 1 == num_batches:
                average_gradients(model)
                optimizer.step()
                optimizer.zero_grad()

            acc = (output.argmax(dim=1) == target).float().mean()

            B = data.shape[0]
            tensor = torch.concat([loss.clone().detach().unsqueeze(0) * B, acc.clone().detach().unsqueeze(0) * B, torch.tensor([B], device=device)])
            dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM)

            sum_loss += tensor[0].item()
            sum_acc += tensor[1].item()
            sum_B += tensor[2].item()
        # where's the validation loop

        if rank == 0:
            print(f"loss: {sum_loss / sum_B}, acc: {sum_acc / sum_B}")
        
        val_acc = validation(rank, size, model, 64)
        if rank == 0:
            print(f"Validation acc: {val_acc}")
    
    print(f"Rank={rank} peak memory: {torch.cuda.max_memory_allocated(device) / 1024**2}MB")

    if rank == 0:
        print(f"Time: {time.time() - start_time}")


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    init_process(local_rank, fn=run_training, backend="nccl")  # replace with "nccl" when testing on several GPUs
