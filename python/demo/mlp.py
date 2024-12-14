import numpy as np

import sys
from pathlib import Path
work_path = Path(__file__).parent.parent.parent
sys.path.append(work_path.__str__() + '\\python')
import tinytorch as ttorch
import tinytorch.nn as nn

device = ttorch.cuda()
data_path = work_path / Path('data') / Path('mnist')


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1, device=None):
    ### BEGIN YOUR SOLUTION
    modules = nn.Sequential(
        nn.Linear(dim, hidden_dim, device=device),
        norm(hidden_dim, device=device),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim, device=device),   
        norm(dim, device=device)
    )
    return nn.Sequential(
        nn.Residual(modules),
        nn.ReLU()
    )
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1, device=None):
    ### BEGIN YOUR SOLUTION
    modules = [nn.Flatten(), nn.Linear(dim, hidden_dim, device=device), nn.ReLU()]
    for i in range(num_blocks):
        modules.append(ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob, device))
    modules.append(nn.Linear(hidden_dim, num_classes, device=device))
    return nn.Sequential(*modules)
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



def train_mnist(batch_size=100, epochs=2, optimizer=ttorch.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data", device=None):
    np.random.seed(0)
    ### BEGIN YOUR SOLUTION
    train_data = ttorch.data.MNISTDataset(
        data_dir + '/train-images-idx3-ubyte.gz',
        data_dir + '/train-labels-idx1-ubyte.gz'
    )
    test_data = ttorch.data.MNISTDataset(
        data_dir + '/t10k-images-idx3-ubyte.gz',
        data_dir + '/t10k-labels-idx1-ubyte.gz',
    )
    train_loader = ttorch.data.DataLoader(train_data, batch_size, device=device, shuffle=True)
    test_loader = ttorch.data.DataLoader(test_data, batch_size, device=device)
    model = MLPResNet(784, hidden_dim=hidden_dim, device=device)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    for i in range(epochs):
        train_acc, train_loss = epoch(train_loader, model, opt)
        print(f"Epoch: {i}, Train Loss: {train_loss}, Train Acc: {train_acc}")
        test_acc, test_loss = epoch(test_loader, model)
        print(f"Epoch: {i}, Test Loss: {test_loss}, Test Acc: {test_acc}")
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir=data_path.__str__(), device=device)