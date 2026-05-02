import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 4096)
        self.linear2 = nn.Linear(4096, 4096)
        self.linear3 = nn.Linear(4096, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x


def select_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    device = select_device()
    print(f"device: {device}")

    loader = DataLoader(
        MNIST("data", train=True, download=True, transform=ToTensor()),
        batch_size=1024,
        shuffle=True,
    )
    model = Net().to(device)

    opt = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(10):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            opt.step()

        print(epoch, loss.item())


if __name__ == "__main__":
    main()
