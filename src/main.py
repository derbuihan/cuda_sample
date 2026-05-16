import numpy as np

from .tensor import Tensor


def main():
    W = Tensor(np.random.randn(100, 100))
    x = Tensor(np.random.randn(100, 1))
    y = Tensor(np.random.randn(1, 100))
    loss = y @ W @ x
    loss.backward()

    print(loss.numpy())
    print(loss.grad.numpy())


if __name__ == "__main__":
    main()
