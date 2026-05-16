import numpy as np
from numba import cuda


@cuda.jit
def add_kernel(a, b, out):
    i = cuda.grid(1)

    if i < out.size:
        out.flat[i] = a.flat[i] + b.flat[i]


@cuda.jit
def add_inplace_kernel(a, b):
    i = cuda.grid(1)

    if i < a.size:
        a.flat[i] += b.flat[i]


@cuda.jit
def matmul_kernel(a, b, out):
    row, col = cuda.grid(2)

    if row < out.shape[0] and col < out.shape[1]:
        s = 0
        for k in range(a.shape[1]):
            s += a[row, k] * b[k, col]
        out[row, col] = s


@cuda.jit
def relu_kernel(a, out):
    i = cuda.grid(1)

    if i < a.size:
        out.flat[i] = max(a.flat[i], np.float32(0.0))


@cuda.jit
def relu_backward_kernel(a, out_grad, a_grad):
    i = cuda.grid(1)

    if i < a.size:
        if a.flat[i] > 0:
            a_grad.flat[i] += out_grad.flat[i]
