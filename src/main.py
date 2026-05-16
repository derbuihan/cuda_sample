from .tensor import Tensor


def main():
    x = Tensor([[5.0, 6.0]])
    y = Tensor([[2.0, 1.0]])
    z = x + y
    z.backward()
    print(x.grad.numpy())
    print(y.grad.numpy())


if __name__ == "__main__":
    main()
