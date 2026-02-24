from itertools import batched
import math

from collections.abc import Generator, Sequence

from typing import Any, Generic, Protocol, TypeVar

import numpy as np
import torch

__all__ = ["TensorDataLoader", "LightDataLoader"]
# ----------------------------------- types ---------------------------------- #
_T_co = TypeVar("_T_co", covariant=True)

class _HasIterDunder(Protocol[_T_co]):
    def __iter__(self) -> _T_co: ...

class _SupportsLenAndGetitem(Protocol[_T_co]):
    def __len__(self) -> int: ...
    def __getitem__(self, __k: int, /) -> _T_co: ...

def _get_generator(seed: int|None|torch.Generator, device):
    if isinstance(seed, torch.Generator):
        generator = seed
        seed = generator.seed()
    elif seed is not None:
        generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = None
        seed = None

    generator_state = None
    if generator is not None:
        generator_state = generator.get_state().clone()

    return generator, generator_state, seed

# ----------------------------- tensor dataloader ---------------------------- #

_TensorOrTuple = TypeVar("_TensorOrTuple", bound=torch.Tensor | Sequence[torch.Tensor])

class TensorDataLoader(Generic[_TensorOrTuple]):
    def __init__(
        self,
        data: _TensorOrTuple,
        batch_size: int = 1,
        shuffle: bool = False,
        memory_efficient: bool = False,
        seed: int | torch.Generator | None = None,
    ):
        """A very fast DataLoader for datasets that are fully loaded into memory as tensors.

        Args:
            data (Tensor | Sequence[Tensor]):
                single tensor with all samples stacked along the first dimension, or tuple of tensors that have the same size of the first dimension.
                For example, if you have 500 samples of shape (32,32) and 500 labels of shape (10),
                you can pass a tuple of two tensors of the following shapes: `(samples[500, 32, 32], labels[500, 10])`.
            batch_size (int, optional): how many samples per batch to load (default: 1).
            shuffle (bool, optional): set to True to have the data reshuffled at every epoch (default: False).
            memory_efficient (bool, optional):
                enables memory efficient dataloader.
                During shuffling before each epoch, this uses two times the memory that `data` uses.
                But when `memory_efficient` is enabled, no additional memory will be used.
                It is slightly slower on my laptop, but much faster on Google Colab (default: False).
            seed (int | torch.Generator | None, optional):
                seed for shuffling, set to None to let pytorch use a random seed.
                Can also be a torch.Generator, but make sure it is on the same device as `data`. Defaults to None.
        """

        self.data: _TensorOrTuple = data
        self._shuffled_data = None
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.memory_efficient = memory_efficient

        self._istensor = isinstance(self.data, torch.Tensor)
        self.device = self.data.device if isinstance(self.data, torch.Tensor) else self.data[0].device

        self.generator, self.generator_state, self.seed = _get_generator(seed, self.device)

    def reset_rng(self):
        if self.generator is None: return
        assert self.generator_state is not None
        self.generator.set_state(self.generator_state.clone())

    def data_length(self):
        ref = self.data if self._istensor else self.data[0]
        return ref.size(0) # pyright:ignore[reportAttributeAccessIssue]

    def __len__(self):
        return math.ceil(self.data_length() / self.batch_size)

    def _fast_iter(self) -> Generator[_TensorOrTuple, None, None]:
        if self.shuffle:
            idxs = torch.randperm(self.data_length(), generator = self.generator, device = self.device)
            if self._istensor:
                self._shuffled_data = torch.index_select(self.data, 0, idxs) # pyright:ignore[reportCallIssue,reportArgumentType]
            else:
                self._shuffled_data = [torch.index_select(i, 0, idxs) for i in self.data]
        else:
            self._shuffled_data = self.data

        if self._istensor:
            yield from self._shuffled_data.split(self.batch_size) # pyright:ignore[reportAttributeAccessIssue]
        else:
            yield from zip(*(i.split(self.batch_size) for i in self._shuffled_data))

    def _memory_efficient_iter(self) -> Generator[_TensorOrTuple, None, None]:
        if self.shuffle:
            idxs = torch.randperm(self.data_length(), generator = self.generator, device = self.device)

            for batch_indices in idxs.split(self.batch_size):
                if self._istensor:
                    yield self.data[batch_indices] # pyright:ignore[reportCallIssue,reportArgumentType]
                else:
                    yield [i[batch_indices] for i in self.data]

        else:
            if self._istensor:
                yield from self.data.split(self.batch_size) # pyright:ignore[reportAttributeAccessIssue]
            else:
                yield from zip(*(i.split(self.batch_size) for i in self.data))

    def __iter__(self) -> Generator[_TensorOrTuple, None, None]:
        if self.memory_efficient: return self._memory_efficient_iter()
        return self._fast_iter()

# ----------------------------- light dataloader ----------------------------- #
class _SupportsLenAndGetitems(Protocol[_T_co]):
    def __len__(self) -> int: ...
    def __getitems__(self, __k: Any) -> Sequence[_T_co]: ...

_SampleOrTuple = TypeVar("_SampleOrTuple", bound=torch.Tensor | Any | Sequence[torch.Tensor] | Sequence[Any])

class LightDataLoader(Generic[_SampleOrTuple]):
    def __init__(
        self,
        data: _SupportsLenAndGetitem[_SampleOrTuple] | _SupportsLenAndGetitems[_SampleOrTuple],
        batch_size: int = 1,
        shuffle: bool = False,
        seed: int | np.random.Generator | None = None,
    ):
        """A lightweight dataloader that collates `data`.

        Args:
            data (Sequence[Tensor | int | float | Any] | Sequence[Sequence[Tensor | int | float | Any]]):
                a sequence of samples. They must either be tensors or tuples of tensors,
                or be convertable to tensors with `torch.as_tensor`, so they can be python ints and floats.

                As in pytorch dataloader, `data` can also be an object with `__getitems__(self, idxs: tuple[int, ...])` method.
                It should be equivalent to `[data[i] for i in idxs]`, but can potentially implement
                paralellization.
            batch_size (int, optional): how many samples per batch to load (default: 1).
            shuffle (bool, optional): set to True to have the data reshuffled at every epoch (default: False).
            seed (int | np.random.Generator | None, optional):
                seed for shuffling, set to None to let numpy use a random seed.
                Can also be a numpy.random.Generator. Defaults to None.
        """
        self.data: _SupportsLenAndGetitem[_SampleOrTuple] = data
        self.batch_size = batch_size
        self.shuffle = shuffle

        self._use_getitems = hasattr(self.data, "__getitems__")

        if isinstance(seed, np.random.Generator):
            self.generator = seed
            self.seed = None

        else:
            self.seed = seed
            self.generator = np.random.default_rng(seed)


    def reset_rng(self):
        if self.seed is not None: self.generator = np.random.default_rng(self.seed)

    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)

    def __iter__(self) -> Generator[_SampleOrTuple, None, None]:

        if self.shuffle: indices = self.generator.permutation(len(self.data))
        else: indices = range(len(self.data))

        for batch_indices in batched(indices, self.batch_size):
            if self._use_getitems: uncollated_batch = self.data.__getitems__(batch_indices)
            else: uncollated_batch = [self.data[i] for i in batch_indices]
            if isinstance(uncollated_batch[0], torch.Tensor):
                yield torch.stack(uncollated_batch)
            else:
                collated = list(zip(*uncollated_batch))
                yield [torch.stack(i) if isinstance(i[0], torch.Tensor) else torch.as_tensor(i) for i in collated]

