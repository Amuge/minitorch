from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence, Tuple, Union

import numba
import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias

from .operators import prod

MAX_DIMS = 32


class IndexingError(RuntimeError):
    "Exception raised for indexing errors."
    pass


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """
    Converts a multidimensional tensor `index` into a single-dimensional position in
    storage based on strides.

    Args:
        index : index tuple of ints
        strides : tensor strides

    Returns:
        Position in storage
    """

    # TODO: Implement for Task 2.1.
    pos = 0
    for x,y in zip(index,strides):
        pos += x*y
    return pos


def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """
    Convert an `ordinal` to an index in the `shape`.
    Should ensure that enumerating position 0 ... size of a
    tensor produces every index exactly once. It
    may not be the inverse of `index_to_position`.

    Args:
        ordinal: ordinal position to convert.
        shape : tensor shape.
        out_index : return index corresponding to position.

    """
    # TODO: Implement for Task 2.1.
    # #get the 'strides'
    # len_shape = len(shape)
    # strides = strides_from_shape(shape)
    # #calculate the index of each dim
    # cur_ord = ordinal
    # for i in range(len_shape):
    #     out_index[i] = cur_ord // strides[i]
    #     cur_ord -= strides[i] * out_index[i]

    current_ordinal = ordinal + 0 # ??????

    for i in range(len(shape) - 1, -1, -1):
        #store back to out_index_list
        out_index[i] = int(current_ordinal % shape[i])#remainder
        current_ordinal = current_ordinal // shape[i]#divided answer
     
    # cur_ord = ordinal + 0 # why plus 0? why without it would cause the numba.core.errors? 
    # for i in range(len(shape) - 1, -1, -1):
    #     sh = shape[i]
    #     out_index[i] = int(cur_ord % sh)
    #     cur_ord = cur_ord // sh


def broadcast_index(
    big_index: Index, big_shape: Shape, shape: Shape, out_index: OutIndex
) -> None:
    """
    Convert a `big_index` into `big_shape` to a smaller `out_index`
    into `shape` following broadcasting rules. In this case
    it may be larger or with more dimensions than the `shape`
    given. Additional dimensions may need to be mapped to 0 or
    removed.

    Args:
        big_index : multidimensional index of bigger tensor
        big_shape : tensor shape of bigger tensor
        shape : tensor shape of smaller tensor
        out_index : multidimensional index of smaller tensor

    Returns:
        None
    """
    # TODO: Implement for Task 2.2.
    for i in range(len(shape)):
        offset = len(big_shape) - len(shape) + i
        out_index[i] = big_index[offset] if shape[i] != 1 else 0


def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """
    Broadcast two shapes to create a new union shape.

    Args:
        shape1 : first shape
        shape2 : second shape

    Returns:
        broadcasted shape

    Raises:
        IndexingError : if cannot broadcast
    """
    # TODO: Implement for Task 2.2.
    x = shape1
    y = shape2

    new_x_shape = list(x)
    new_y_shape = list(y)
    len_diff = abs(len(x) - len(y))
    if len_diff > 0:
        if(len(x) < len(y)):
            for i in range(len_diff):
                new_x_shape.insert(0,1)
        else:
            for i in range(len_diff):
                new_y_shape.insert(0,1)
 
    length = max(len(x),len(y))
    new_shape = [0] * length

    for i in range(length):
        if new_x_shape[i] == 1 or new_y_shape[i] == 1 or new_x_shape[i] == new_y_shape[i]:
            new_shape[i] = max(new_x_shape[i],new_y_shape[i])
        else:
            raise IndexingError("Broadcast failure")
    return tuple(new_shape)

    # l = max(len(shape1), len(shape2))
    # if len(shape1) > len(shape2):
    #     shape2 = [1 for i in range(l - len(shape2))] + list(shape2)
    # else:
    #     shape1 = [1 for i in range(l - len(shape1))] + list(shape1)
    # ans = []
    # for i in range(l):
    #     if shape1[i] != shape2[i] and shape1[i] != 1 and shape2[i] != 1:
    #         raise IndexingError("it cannot broadcast")
    #     ans.append(max(shape1[i], shape2[i]))
    # return tuple(ans)


def strides_from_shape(shape: UserShape) -> UserStrides:
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self) -> None:  # pragma: no cover
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """
        Check that the layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions.

        Returns:
            bool : True if contiguous
        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        if isinstance(index, int):
            aindex: Index = array([index])
        if isinstance(index, tuple):
            aindex = array(index)

        # Pretend 0-dim shape is 1-dim shape of singleton
        shape = self.shape
        if len(shape) == 0 and len(aindex) != 0:
            shape = (1,)

        # Check for errors
        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be size of {self.shape}.")
        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {aindex} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {aindex} not supported.")

        # Call fast indexing.
        # return index_to_position(array(index), self._strides)
        return index_to_position(aindex, self._strides)

    def indices(self) -> Iterable[UserIndex]:
        lshape: Shape = array(self.shape)
        out_index: Index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> UserIndex:
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, val: float) -> None:
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """
        Permute the dimensions of the tensor.

        Args:
            *order: a permutation of the dimensions

        Returns:
            New `TensorData` with the same storage and a new dimension order.
        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"
        # TODO: Implement for Task 2.1.
        def chage_dim(order:UserIndex,old_shape:UserShape,old_stride:UserStrides):
            new_shape = [0] * len(order)
            new_strides = [0] * len(order)
            for i in range(len(order)):
                new_shape[i] = old_shape[order[i]]
                new_strides[i] = old_stride[order[i]]
            return tuple(new_shape),tuple(new_strides)
                
        new_shape,new_strides = chage_dim(order,self.shape,self.strides)
        return TensorData(self._storage,new_shape,new_strides)

        # return TensorData(
        #     self._storage,
        #     tuple([self.shape[i] for i in order]),
        #     tuple([self.strides[i] for i in order]),
        # )

    def to_string(self) -> str:
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s
