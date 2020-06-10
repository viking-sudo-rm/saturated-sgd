import torch.nn as nn
import torch
import torchvision

from torchvision import transforms
from torch.utils import data

import torch.optim as optim
import numpy as np
from optimizers import FloorSGD
from models import *

mnist_train = torchvision.datasets.FashionMNIST(
    "~/Documents", train=True, download=True, transform=transforms.ToTensor()
)

mnist_test = torchvision.datasets.FashionMNIST(
    "~/Documents", train=False, download=True, transform=transforms.ToTensor()
)

trainloader = data.DataLoader(mnist_train, batch_size=256, shuffle=True, num_workers=4)
testloader = data.DataLoader(mnist_test, batch_size=256, shuffle=False, num_workers=4)


@torch.no_grad()
def model_norm(model):
    norm = 0.0
    norms = []
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            norms.append(m.weight.norm().item())

    return np.mean(norms)


standard_perm = torch.arange(10)


def train_model(
    model, trainloader, testloader, optimizer, epochs=10, labelperm=standard_perm
):
    model.train()
    accs = []
    for e in range(epochs):
        for i, (data, targets) in enumerate(trainloader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, labelperm[targets])
            loss.backward()
            optimizer.step()
            if i % 20 == 0:
                print(
                    f"e{e}, i {i}, Loss: {loss.item():.4f}, Norm: {model_norm(model):.2f}",
                    end="\r",
                    flush=True,
                )

        print()

        test_acc = eval_model(model, testloader, labelperm)
        accs.append(test_acc)

    return accs


def eval_model(model, testloader, labelperm=standard_perm):
    model.eval()
    with torch.no_grad():
        correct = 0
        for i, (data, targets) in enumerate(testloader):
            predictions = model(data).argmax(dim=1, keepdim=True)
            acc1 = predictions.eq(labelperm[targets].view_as(predictions)).sum().item()
            correct += acc1

        print("Acc@1:", correct / len(testloader.dataset))

    model.train()
    return correct / len(testloader.dataset)


model = LeNet()

optimizer = optim.SGD(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

print("=> Normal SGD Training")
accs_pretr1 = train_model(model, trainloader, testloader, optimizer, epochs=40)

optimizer = FloorSGD(model.parameters(), lr=0.1)
print("=> Fine-tuning to find saturated network")
accs_pretr2 = train_model(model, trainloader, testloader, optimizer, epochs=40)
