import numpy as np

import sys
from pathlib import Path
work_path = Path(__file__).parent.parent.parent
sys.path.append(work_path.__str__() + '\\python')
import tinytorch as ttorch
import tinytorch.nn as nn
import tinytorch.data.data_transforms as Transforms

device = ttorch.cuda()
data_path = work_path / Path('data') / Path('cifar10')


class ConvBN(ttorch.nn.Module):
    def __init__(
            self,
            in_channels, out_channels,
            kernel_size=3, stride=1, bias=True,
            norm=nn.BatchNorm2d,
            norm_eps=1e-5, norm_momentum=0.1,
            device=None, dtype="float32"
    ):
        super().__init__()
        self.conv = nn.Conv(in_channels, out_channels, kernel_size, stride, bias, device, dtype)
        self.BN = norm(out_channels, eps=norm_eps, momentum=norm_momentum, device=device, dtype=dtype)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        conv_x = self.conv(x)
        return self.relu(self.BN(conv_x))


class ResNet(ttorch.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.block1 = nn.Sequential(
            ConvBN(3, 16, 7, 4, device=device),
            ConvBN(16, 32, 3, 2, device=device)
        )
        block2 = nn.Sequential(
            ConvBN(32, 32, 3, 1, device=device),
            ConvBN(32, 32, 3, 1, device=device)
        )
        self.block2 = nn.Residual(block2)
        self.block3 = nn.Sequential(
            ConvBN(32, 64, 3, 2, device=device),
            ConvBN(64, 128, 3, 2, device=device)
        )
        block4 = nn.Sequential(
            ConvBN(128, 128, 3, 1, device=device),
            ConvBN(128, 128, 3, 1, device=device)
        )
        self.block4 = nn.Residual(block4)
        self.block5 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128, device=device),
            nn.ReLU(),
            nn.Linear(128, 10, device=device)
        )
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        out = self.block5(x4)
        return out
        ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(0)
    ### BEGIN YOUR SOLUTION
    hit, total = 0, 0
    loss_func = nn.SoftmaxLoss()
    loss_all = 0
    if opt is not None:
        model.train()
        for idx, data in enumerate(dataloader):
            x, y = data
            output = model(x)
            opt.reset_grad()
            loss = loss_func(output, y)
            loss_all += loss.numpy()[0]
            loss.backward()
            opt.step()
            hit += (y.numpy() == output.numpy().argmax(1)).sum()
            total += y.shape[0]
    else:
        model.eval()
        for idx, data in enumerate(dataloader):
            x, y = data
            output = model(x)
            loss = loss_func(output, y)
            loss_all += loss.numpy()[0]
            hit += (y.numpy() == output.numpy().argmax(1)).sum()
            total += y.shape[0]
    acc = hit / total
    return acc, loss_all / (idx + 1)
    ### END YOUR SOLUTION


def train_mnist(batch_size=100, epochs=10, optimizer=ttorch.optim.Adam,
                lr=0.01, weight_decay=0.001, data_dir="data", device=None):
    np.random.seed(0)

    transforms = [Transforms.RandomCrop(padding=4),
                  Transforms.RandomFlipHorizontal()]

    train_data = ttorch.data.CIFAR10Dataset(
        base_folder=data_dir, train=True, transforms=transforms)
    test_data = ttorch.data.CIFAR10Dataset(
        base_folder=data_dir, train=False, transforms=transforms)
    train_loader = ttorch.data.DataLoader(train_data, batch_size=batch_size, device=device, shuffle=True)
    test_loader = ttorch.data.DataLoader(test_data, batch_size=batch_size, device=device)

    model = ResNet(device=device)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    for i in range(epochs):
        train_acc, train_loss = epoch(train_loader, model, opt)
        print(f"Epoch: {i}, Train Loss: {train_loss}, Train Acc: {train_acc}")
        test_acc, test_loss = epoch(test_loader, model)
        print(f"Epoch: {i}, Test Loss: {test_loss}, Test Acc: {test_acc}")


if __name__ == "__main__":
    train_mnist(batch_size=100, data_dir=data_path.__str__(), device=device)