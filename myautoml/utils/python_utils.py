import json
import os
import functools
import weakref
from collections import Counter, UserDict, UserList
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, cast, overload

if TYPE_CHECKING:
    from typing import Self, TypeGuard


class HasFit(Protocol):
    def fit(self, *__args, **__kwargs) -> "Self": ...

class HasPredict(Protocol):
    def predict(self, *__args, **__kwargs) -> Any: ...

class HasTransform(Protocol):
    def transform(self, *__args, **__kwargs) -> Any: ...

class EstimatorProtocol(Protocol):
    def fit(self, *__args, **__kwargs) -> Any: ...
    def predict(self, *__args, **__kwargs) -> Any: ...

class TransformerProtocol(Protocol):
    def fit(self, *__args, **__kwargs) -> Any: ...
    def transform(self, *__args, **__kwargs) -> Any: ...

type TupleOrMapping[K1, V1] = Sequence[tuple[K1, V1]] | Mapping[K1, V1]

def totuples[K2,V2](x: TupleOrMapping[K2,V2]) -> list[tuple[K2,V2]]:
    if isinstance(x, Sequence): return [el for el in x]
    return list(x.items())

def is_sequence(x) -> "TypeGuard[Sequence]":
    if isinstance(x, str): return False
    return isinstance(x, Sequence)

def get_duplicates[T1](seq: Sequence[T1]) -> list[T1]:
    """use for small sequences"""
    c = Counter(seq)
    return [k for k in c if c[k] > 1]

def include_exclude(seq: Sequence[str], include: str | Sequence[str] | None, exclude: str | Sequence[str] | None) -> list[str]:
    if isinstance(include, str): include = (include, )
    if isinstance(exclude, str): exclude = (exclude, )
    if include is not None: seq = [el for el in seq if el in include]
    if exclude is not None: seq = [el for el in seq if el not in exclude]
    return list(seq)

type _Nested[X] = X | Iterable[X] | Iterable[_Nested[X]]

def flatiter[T](iterable:Iterable[_Nested[T]]) -> Iterator[T]:
    for item in iterable:
        if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            yield from flatiter(item)
        else:
            yield cast(T, item)

def flatten[T](x: Iterable[_Nested[T]]) -> list[T]:
    return list(flatiter(x))

def inner_reorder(s1: Sequence, s2: Sequence):
    """Reorders s1 such that all elements containing in s2 have the same order as in s2,
    while other elements order is unchanged"""
    # Map elements in s2 to their index for fast lookup
    order_map = {val: i for i, val in enumerate(s2)}

    # Extract elements from s1 that exist in s2 and sort them by s2s order
    # This maintains the count of duplicates if s1 has multiple of the same element
    subset_to_sort = [x for x in s1 if x in order_map]
    subset_to_sort.sort(key=lambda x: order_map[x])

    # put sorted elements back
    sorted_iter = iter(subset_to_sort)
    return [next(sorted_iter) if x in order_map else x for x in s1]

def weakself_cache(func):
    """``functools.cache`` with weak reference to self"""
    @functools.cache
    def _weak_func(_self_ref, *args, **kwargs):
        return func(_self_ref(), *args, **kwargs)

    @functools.wraps(func)
    def inner(self, *args, **kwargs):
        return _weak_func(weakref.ref(self), *args, **kwargs)

    return inner

class StrictDict[K,V](UserDict[K, V]):
    def __setitem__(self, key: K, value: V):
        if key in self:
            raise KeyError(f"Key '{key}' already exists and cannot be replaced")
        super().__setitem__(key, value)

def safe_dict_update_(d1_:dict, d2:dict):
    """Updates ``d1`` with ``d2``, but raises ``RuntimeError`` if there are any duplicate keys"""
    inter = set(d1_.keys()).intersection(d2.keys())
    if len(inter) > 0: raise RuntimeError(f"Duplicate keys {inter}")
    d1_.update(d2)

class UnableToFit(Exception):
    """Raised when fitter is unable to fit the ``Task``, as alternative to ``can_fit``."""


type Composable[_Return] = (
    Callable[..., _Return] |
    Iterable[Callable[..., _Return | Any]]
)

class Compose[Input, Return](UserList[Callable]):
    """Compose multiple functions into a single function."""
    @overload
    def __init__(self, __functions: Iterable[Callable | None]): ...
    @overload
    def __init__(self, *functions: Callable | None): ...
    def __init__(self, *functions):

        # passing iterable which isn't callable
        if len(functions) == 1:
            first = functions[0]
            if first is not None and not callable(first):
                functions = first

        # filter None
        functions = [f for f in functions if f is not None]
        super().__init__(functions)

    def __call__(self, x: Input, *args, **kwargs) -> Return:
        for t in self:
            x = t(x, *args, **kwargs)
        return cast(Return, x)

    @overload
    def __iadd__(self, item: Callable[[Return], Return]): ...
    @overload
    def __iadd__(self, item: Iterable[Callable[..., Return | Any]]): ...
    def __iadd__(self, item: Composable[Return]):

        # single callable
        if callable(item):
            super().append(item)

        # iterable
        else:
            super().extend(item)

    append = __iadd__

    @overload
    def prepend(self, item: Callable[[Input], Input]): ...
    @overload
    def prepend(self, item: Iterable[Callable]): ...
    def prepend(self, item: Composable):

        # single callable
        if callable(item):
            super().insert(0, item)

        # iterable
        else:
            for i,v in enumerate(item):
                super().insert(i, v)

    @overload
    def __add__[NewReturn](self, other: Callable[[Return], NewReturn]) -> "Compose[Input, NewReturn]": ...
    @overload
    def __add__[NewReturn](self, other: Iterable[Callable[..., NewReturn | Any]]) -> "Compose[Input, NewReturn]": ...
    def __add__[NewReturn](self, other: Composable[NewReturn]) -> "Compose[Input, NewReturn]":
        if callable(other): return Compose(*self, other)
        return Compose(*self, *other)

    @overload
    def __radd__[NewInput](self, other: Callable[[NewInput], Input]) -> "Compose[NewInput, Return]": ...
    @overload
    def __radd__(self, other: Iterable[Callable]) -> "Compose[Any, Return]": ...
    def __radd__[NewInput](self, other: Composable[NewInput]) -> "Compose[Input | Any, Return]":
        if callable(other): return Compose(other, *self)
        return Compose(*other, *self)


    __or__ = __add__
    __ror__ = __radd__
    __ior__ = append

class Split:
    """Split with ``n`` functions accepts and iterable of ``n`` elements and zips functions with elements."""
    __slots__ = ('functions', )

    def __init__(self, *functions: Composable | None):
        self.functions = [Compose(i) for i in functions]

    def __call__(self, x: Iterable) -> list:
        return [f(i) for f, i in zip(self.functions, x)]

    def __str__(self):
        return f"Split({', '.join(str(t) for t in self.functions)})"


def read_json(file: str | os.PathLike):
    with open(file, "r", encoding='utf-8') as f:
        return json.load(f)

def write_json(d: Mapping, file: str | os.PathLike):
    with open(file, "w", encoding='utf-8') as f:
        json.dump(d, f)
