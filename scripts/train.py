"""
This file trains the image classifier and saves the PyTorch model (.pth file).
Revised from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html.
"""


import torch
import torchvision
from torchvision import transforms
from torch import nn, optim

from model import ImageClassifier


def train():
    # Dataset and dataloader
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    # Set up device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Create a model
    net = ImageClassifier()
    net.to(device)

    # Loss function
    loss_func = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Training
    net.train()
    for epoch in range(20):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    # Save the model
    PATH = './models/image_classifier.pth'
    torch.save(net.state_dict(), PATH)


if __name__ == '__main__':
    train()
