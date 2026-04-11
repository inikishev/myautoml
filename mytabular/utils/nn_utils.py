from collections.abc import Callable, Sequence, Iterable

import torch
from torch import nn

class FuncModule(nn.Module):
    """Wrap ``func`` into a torch.nn.Module.

    Args:
        func (Callable): _description_
    """
    def __init__(self, func:Callable):
        super().__init__()
        self.func = func

    def forward(self, x): return self.func(x)

def func_to_named_module(func:Callable, name:str | None = None) -> nn.Module:
    """Wrap ``func`` into a torch.nn.Module, except the module will have the same name as the function (or ``name`` if it isn't ``None``).

    Args:
        func (Callable): The function to convert.
        name (str): Optional name of the module, if not specified then the name will be the name of the function.

    Returns:
        nn.Module: The named module.
    """
    if name is None: name = func.__name__ if hasattr(func, '__name__') else func.__class__.__name__
    name = ''.join([i for i in name if i.isalnum() or i == '_'])
    cls = type(name, (FuncModule,), {})
    return cls(func)


def ensure_module(x, named=True) -> nn.Module:
    if isinstance(x, str): raise TypeError(f"ensure module got string {x}")
    if isinstance(x, nn.Module): return x
    if isinstance(x, Callable):
        if named: return func_to_named_module(x)
        return FuncModule(x)
    if isinstance(x, Sequence):
        return torch.nn.Sequential(*(ensure_module(i) for i in x))
    raise TypeError(f"Can't convert {x} to module")


class ModuleList(torch.nn.ModuleList):
    """Module list that also accepts callables."""
    def __init__(self, modules: Iterable[torch.nn.Module | Callable] | None = None):
        super().__init__([ensure_module(i) for i in modules] if modules is not None else None)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f'Forward not implemented for {self.__class__.__name__}')

class Sequential(torch.nn.Sequential):
    """Sequential module that also accepts callables."""
    def __init__(self, *args: torch.nn.Module | Callable):
        super().__init__(*[ensure_module(i) for i in args])