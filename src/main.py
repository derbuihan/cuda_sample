from .tensor import Tensor


def main():
    x = Tensor([[1, -2], [-1, 2]])
    y = Tensor([[10, 11], [12, 13]])
    z = x @ y
    w = z.relu()

    print(x)
    print(x.numpy())
    print(y)
    print(y.numpy())
    print(z)
    print(z.numpy())
    print(w)
    print(w.numpy())


if __name__ == "__main__":
    main()
