import torch
import torch.distributed as dist
from torch.autograd import Function
from torch.autograd.function import FunctionCtx
from torch.nn.modules.batchnorm import _BatchNorm


class sync_batch_norm(Function):
    """
    A version of batch normalization that aggregates the activation statistics across all processes.

    This needs to be a custom autograd.Function, because you also need to communicate between processes
    on the backward pass (each activation affects all examples, so loss gradients from all examples affect
    the gradient for each activation).

    For a quick tutorial on torch.autograd.function, see
    https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    """

    @staticmethod
    def forward(ctx: FunctionCtx, input: torch.Tensor, running_mean, running_var, eps: float, momentum: float):
        # Compute statistics, sync statistics, apply them to the input
        # Also, store relevant quantities to be used on the backward pass with `ctx.save_for_backward`FunctionCtx
        sm = input.sum(dim=0)
        sm2 = (input ** 2).sum(dim=0)
        tensor = torch.concat((sm, sm2, torch.tensor([input.shape[0]], device=input.device)))
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        sm_reduce = tensor[:input.shape[1]]
        sm2_reduce = tensor[input.shape[1]:input.shape[1]*2]
        B = tensor[-1]

        mu = sm_reduce / B
        sigma2 = sm2_reduce / B - mu ** 2

        with torch.no_grad():
            running_mean *= (1 - momentum)
            running_mean += momentum * mu
            running_var *= (1 - momentum)
            running_var += momentum * sigma2

        invsigma = 1 / (sigma2 + eps).sqrt()
        output = (input - mu) * invsigma
        ctx.save_for_backward(output, invsigma, B)
        return output

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: torch.Tensor):
        # don't forget to return a tuple of gradients wrt all arguments of `forward`!
        output, invsigma, B = ctx.saved_tensors
        tensor = torch.concat([grad_output.sum(dim=0), (grad_output * output).sum(dim=0)])
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        meang = tensor[:output.shape[1]] / B
        meangoutput = tensor[output.shape[1]:output.shape[1]*2] / B
        return (grad_output - meang - output * meangoutput) * invsigma, None, None, None, None, None


class SyncBatchNorm(_BatchNorm):
    """
    Applies Batch Normalization to the input (over the 0 axis), aggregating the activation statistics
    across all processes. You can assume that there are no affine operations in this layer.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__(
            num_features,
            eps,
            momentum,
            affine=False,
            track_running_stats=True,
            device=None,
            dtype=None,
        )
        # your code here
        self.running_mean = torch.zeros((num_features,))
        self.running_var = torch.ones((num_features,))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # You will probably need to use `sync_batch_norm` from above
        if self.training:
            return sync_batch_norm.apply(input, self.running_mean, self.running_var, self.eps, self.momentum)

        return (input - self.running_mean) / (self.running_var + self.eps).sqrt()
