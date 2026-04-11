from collections.abc import Callable, Iterable
import torch
from torch import nn
from torch.nn import functional as F
from ..utils import nn_utils

class MLP(nn.Module):
    """Multi-layer perceptorn.

    for example, if we have 10 inputs, 3 outputs, and want two hidden layer of sizes 128 and 64:

    ```python
    model = vb.models.MLP([10, 128, 64, 3])
    ```

    Args:
        channels (int | Iterable[int] | None):
            list of widths of linear layers. First value is number of input channels and last is number of output channels.
        act_cls (Callable | None, optional): activation function class. Defaults to nn.ReLU.
        bn (bool, optional): if True enables batch norm. Defaults to False.
        dropout (float, optional): dropout probability. Defaults to 0.
        ortho_init (bool, optional): if true ises orthgonal init. defaults to False.
        cls (Callable, optional):
            you can change it from using nn.Linear to some other class with same API. Defaults to nn.Linear.
    """
    def __init__(
        self,
        channels: Iterable[int],
        act_cls: Callable | None = nn.ReLU,
        bn: bool = False,
        dropout: float = 0,
        ortho_init: bool = False,
        cls: Callable = nn.Linear,
    ):
        super().__init__()
        channels = list(channels)

        layers = []

        # if len(channels) = 2, this entire thing is skipped (is empty) so we get only head
        for i,o in zip(channels[:-2], channels[1:-1]):
            layers.append(cls(i, o, not bn))
            if act_cls is not None: layers.append(act_cls())
            if bn: layers.append(nn.BatchNorm1d(o))
            if dropout > 0: layers.append(nn.Dropout1d(dropout))

        self.layers = nn_utils.Sequential(*layers)
        self.head = cls(channels[-2], channels[-1])

        if ortho_init:
            generator=torch.Generator().manual_seed(0)
            for p in self.parameters():
                if p.ndim >= 2:
                    torch.nn.init.orthogonal_(p, generator=generator)

    def forward(self, x: torch.Tensor):
        if x.ndim > 2:
            x = x.flatten(1,-1)

        for l in self.layers: x = l(x)
        return self.head(x)


def rmsnorm(x: torch.Tensor, dims=(-2,-1), with_mean:bool=True):
    for dim in dims:
        if with_mean: x = x - x.mean(dim, keepdim=True)
        x = x / x.square().mean(dim, keepdim=True).sqrt().clip(min=1e-8)
    return x

class RMSNormWithMean(nn.Module):
    def __init__(self, dims=(-2,-1), with_mean:bool=True):
        super().__init__()
        self.dims = dims
        self.with_mean = with_mean

    def forward(self, x: torch.Tensor):
        return rmsnorm(x, self.dims, self.with_mean)

class MOE(nn.Module):
    """Basic MOE"""
    def __init__(self, in_channels: int, out_channels: int, n_experts: int,
                 hidden_channels: int, n_hidden:int, rms_norm:bool=False, with_mean:bool=True):
        super().__init__()
        self.n_experts = n_experts
        self.weighter = nn.Linear(in_channels, n_experts)
        self.W_ih = nn.Parameter(torch.randn(n_experts, in_channels, hidden_channels))
        self.W_hh = nn.ParameterList(
            nn.Parameter(torch.randn(n_experts, hidden_channels, hidden_channels)) for _ in range(n_hidden-1)
        )
        self.W_ho = nn.Parameter(torch.randn(n_experts, hidden_channels, out_channels))
        self.leak = nn.Parameter(torch.linspace(-2, 2, n_experts).unsqueeze(-1)) # (N, 1)

        self.rms_norm = rms_norm
        self.with_mean = with_mean

    def leaky_relu(self, x: torch.Tensor):
        x = torch.where(x < 0, x*self.leak, x)
        return x

    def forward(self, x: torch.Tensor):
        # x is (..., I) - (B, L, I) for LSTM
        weights = torch.softmax(self.weighter(x), -1) # (..., N)

        # W_ih is (N, I, H)
        x = torch.einsum("...i,nih->...nh", x, self.W_ih) # (..., N, H)
        x = self.leaky_relu(x)
        for W in self.W_hh:
            # W is (N, H, H), second H is z
            x = torch.einsum("...nh,nhz->...nz", x, W)
            x = self.leaky_relu(x)

        # W_ho s (N, H, O)
        x = torch.einsum("...nh,nho->...no", x, self.W_ho) * weights.unsqueeze(-1) # (..., N, O)

        if self.rms_norm: x = rmsnorm(x, (-2,-1), with_mean=self.with_mean)
        return x.mean(-2)



class TensorMOE(nn.Module):
    """Two-dimensional MOE also kronecker net"""
    def __init__(self, in_channels: int, out_channels: int, hidden_m: int, hidden_n: int,
                 n_hidden:int, rms_norm:bool=False, with_mean:bool=True):
        super().__init__()
        self.hidden_m = hidden_m
        self.hidden_n = hidden_n
        self.weighter_m = nn.Linear(in_channels, hidden_m)
        self.weighter_n = nn.Linear(in_channels, hidden_n)
        self.W_im = nn.Parameter(torch.randn(hidden_n, in_channels, hidden_m))
        self.W_in = nn.Parameter(torch.randn(hidden_m, in_channels, hidden_n))
        self.W_mm = nn.ParameterList(
            nn.Parameter(torch.randn(hidden_n, hidden_m, hidden_m)) for _ in range(n_hidden-1)
        )
        self.W_nn = nn.ParameterList(
            nn.Parameter(torch.randn(hidden_m, hidden_n, hidden_n)) for _ in range(n_hidden-1)
        )
        self.W_mo = nn.Parameter(torch.randn(hidden_n, hidden_m, out_channels))
        self.W_no = nn.Parameter(torch.randn(hidden_m, hidden_n, out_channels))

        self.leak_m = nn.Parameter(torch.linspace(-2, 2, hidden_m))
        self.leak_n = nn.Parameter(torch.linspace(-2, 2, hidden_n))

        self.rms_norm = rms_norm
        self.with_mean = with_mean

    def forward(self, x: torch.Tensor):
        # x is (..., I) - (B, L, I) for LSTM
        weights_n = torch.softmax(self.weighter_n(x), -1) # (..., N)
        weights_m = torch.softmax(self.weighter_m(x), -1) # (..., M)

        # W_im is (N, I, M)
        x_m = torch.einsum("...i,nim->...nm", x, self.W_im) # (..., N, M)
        x_m = torch.where(x_m < 0, x_m*self.leak_m, x_m)

        x_n = torch.einsum("...i,min->...mn", x, self.W_in) # (..., M, N)
        x_n = torch.where(x_n < 0, x_n*self.leak_n, x_n)

        x = x_m + x_n.mT # (..., N, M)

        for W_m, W_n in zip(self.W_mm, self.W_nn):
            # W_m is (N, M, M), second M is z
            x = torch.einsum("...nm,nmz->...nz", x, W_m) # (..., N, M)
            x = torch.where(x < 0, x*self.leak_m, x)

            # W_n is (M, N, N), second N is z
            x = torch.einsum("...nm,mnz->...zm", x, W_n) # (..., N, M)
            x = torch.where(x < 0, x*self.leak_n.unsqueeze(-1), x)

        # W_mo s (N, M, O)
        x_m = torch.einsum("...nm,nmo->...no", x, self.W_mo) * weights_n.unsqueeze(-1) # (..., N, O)
        x_n = torch.einsum("...nm,mno->...mo", x, self.W_no) * weights_m.unsqueeze(-1) # (..., M, O)

        if self.rms_norm:
            x_m = rmsnorm(x_m, (-2,-1), with_mean=self.with_mean)
            x_n = rmsnorm(x_n, (-2,-1), with_mean=self.with_mean)

        x = x_m.mean(-2) + x_n.mean(-2) # (..., O)
        return x