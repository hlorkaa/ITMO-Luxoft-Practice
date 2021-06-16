#import torchvision.datasets as datasets
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils import data


TRAINING_DIR = 'data/training'
trainset = datasets.ImageFolder(root=TRAINING_DIR, transform=transforms.ToTensor())
trainloader = data.DataLoader(trainset, batch_size=40, shuffle=True)

TESTING_DIR = 'data/training'
testset = datasets.ImageFolder(root=TESTING_DIR, transform=transforms.ToTensor())
testloader = data.DataLoader(testset, batch_size=40, shuffle=True)


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = NeuralNet()
optimizer = optim.Adam(model.parameters())
for(i, l) in trainloader:
    optimizer.zero_grad()
    output = model(i)
    loss = F.nll_loss(output, l)
    loss.backward()
    optimizer.step()

model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in testloader:
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(testloader.dataset)

print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(testloader.dataset),
    100. * correct / len(testloader.dataset)))

torch.save(model, 'torch_model.h5')
