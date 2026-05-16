import numpy as np
from numba import cuda

from .kernels import (
    add_inplace_kernel,
    add_kernel,
    matmul_kernel,
    relu_backward_kernel,
    relu_kernel,
)

THREADS_PER_BLOCK = 256


def blocks_for(size):
    return (size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK


def topological_sort(tensor):
    topo, visited = [], set()

    def build(t):
        if t in visited or not t.requires_grad:
            return
        visited.add(t)
        for p in t._prev:
            build(p)
        topo.append(t)

    build(tensor)
    return topo


def add_backward(t1, t2, out):
    if t1.requires_grad:
        add_inplace_kernel[blocks_for(t1.data.size), THREADS_PER_BLOCK](
            t1.grad.data, out.grad.data
        )
    if t2.requires_grad:
        add_inplace_kernel[blocks_for(t2.data.size), THREADS_PER_BLOCK](
            t2.grad.data, out.grad.data
        )


def relu_backward(tensor, out):
    if tensor.requires_grad:
        relu_backward_kernel[blocks_for(tensor.data.size), THREADS_PER_BLOCK](
            tensor.data, out.grad.data, tensor.grad.data
        )


class Tensor:
    def __init__(self, data, requires_grad=True, dtype=np.float32):
        host_data = np.asarray(data, dtype=dtype)
        self.data = cuda.to_device(host_data)
        self.requires_grad = requires_grad
        self.shape = host_data.shape
        self.dtype = host_data.dtype

        self.grad = None
        if requires_grad:
            self.zero_grad()

        self._prev = []
        self._backward = lambda: None
        self._op = ""

    @classmethod
    def from_device_array(
        cls, data, requires_grad=True, _prev=None, _backward=lambda: None, _op=""
    ):
        obj = cls.__new__(cls)
        obj.data = data
        obj.requires_grad = requires_grad
        obj.shape = tuple(data.shape)
        obj.dtype = data.dtype
        obj.grad = None

        if requires_grad:
            obj.zero_grad()

        obj._prev = [] if _prev is None else list(_prev)
        obj._backward = _backward
        obj._op = _op

        return obj

    def numpy(self):
        return self.data.copy_to_host()

    def __repr__(self):
        return f"Tensor(shape={self.shape}, dtype={self.dtype}, requires_grad={self.requires_grad})"

    def zero_grad(self):
        zeros = np.zeros(self.shape, dtype=self.dtype)
        self.grad = Tensor(zeros, requires_grad=False, dtype=self.dtype)

    def backward(self, grad=None):
        if grad is None:
            ones = np.ones(self.shape, dtype=self.dtype)
            grad = Tensor(ones, requires_grad=False)
        self.grad = grad

        topo = topological_sort(self)
        for t in reversed(topo):
            t._backward()

    def _unary_op(self, kernel):
        out = cuda.device_array(self.shape, dtype=self.dtype)

        blocks = blocks_for(out.size)
        kernel[blocks, THREADS_PER_BLOCK](self.data, out)
        cuda.synchronize()

        return Tensor.from_device_array(out, requires_grad=self.requires_grad)

    def _binary_op(self, other, kernel):
        if not isinstance(other, Tensor):
            raise TypeError("other must be a Tensor")

        if self.shape != other.shape:
            raise ValueError("shapes must be the same")

        out = cuda.device_array(self.shape, dtype=self.dtype)

        threads_per_blocks = 256
        blocks = (out.size + threads_per_blocks - 1) // threads_per_blocks

        kernel[blocks, threads_per_blocks](self.data, other.data, out)
        cuda.synchronize()

        requires_grad = self.requires_grad or other.requires_grad
        return Tensor.from_device_array(out, requires_grad=requires_grad)

    def relu(self):
        out_tensor = self._unary_op(relu_kernel)
        out_tensor._prev = [self]
        out_tensor._op = "relu"
        out_tensor._backward = lambda: relu_backward(self, out_tensor)
        return out_tensor

    def __add__(self, other):
        out_tensor = self._binary_op(other, add_kernel)
        out_tensor._prev = [self, other]
        out_tensor._op = "+"
        out_tensor._backward = lambda: add_backward(self, other, out_tensor)
        return out_tensor

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("other must be a Tensor")

        if self.shape[-1] != other.shape[0]:
            raise ValueError("shapes must be the same")

        shape = self.shape[:-1] + other.shape[1:]
        out = cuda.device_array(shape, dtype=self.dtype)

        threads_per_blocks = (16, 16)
        blocks = (
            (out.shape[0] + threads_per_blocks[0] - 1) // threads_per_blocks[0],
            (out.shape[1] + threads_per_blocks[1] - 1) // threads_per_blocks[1],
        )
        matmul_kernel[blocks, threads_per_blocks](self.data, other.data, out)
        cuda.synchronize()

        requires_grad = self.requires_grad or other.requires_grad
        return Tensor.from_device_array(out, requires_grad=requires_grad)
