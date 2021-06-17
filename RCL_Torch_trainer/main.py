import torch.nn as nn
import torch
from torchvision import datasets, transforms
from torch.utils import data
from tqdm import tqdm

TRAINING_DIR = 'data/training'
trainset = datasets.ImageFolder(root=TRAINING_DIR, transform=transforms.ToTensor())
trainloader = data.DataLoader(trainset, batch_size=40)

TESTING_DIR = 'data/training'
testset = datasets.ImageFolder(root=TESTING_DIR, transform=transforms.ToTensor())
validloader = data.DataLoader(testset, batch_size=40) #testloader


model = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3)),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2, 2)),
    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3)),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2, 2)),
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2, 2)),
    nn.Flatten(),
    nn.Linear(in_features=69696, out_features=512),
    nn.ReLU(),
    nn.Linear(in_features=512, out_features=33),
    nn.Softmax()
)

print("Model is built!")

# Declaring Criterion and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training with Validation
epochs = 2

for e in range(epochs):
    train_loss = 0.0
    correct = 0.0
    training_loop = tqdm(enumerate(trainloader), total=len(trainloader))
    for batch_index, (data, labels) in training_loop:
        # Transfer Data to GPU if available
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()

        optimizer.zero_grad()  # Clear the gradients

        target = model(data)  # Forward Pass
        loss = criterion(target, labels)  # Find the Loss
        loss.backward()  # Calculate gradients
        optimizer.step()  # Update Weights
        train_loss = loss.item() * data.size(0)  # Calculate Loss

        #Update progress bar
        training_loop.set_description(f"Epoch [{e+1}/{epochs}], training progress")
        training_loop.set_postfix(loss=train_loss/len(trainloader))

    valid_loss = 0.0
    model.eval()  # Optional when not using Model Specific layer
    validation_loop = tqdm(enumerate(validloader), total=len(validloader))
    for batch_index, (data, labels) in validation_loop:
        # Transfer Data to GPU if available
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()

        target = model(data)  # Forward Pass
        loss = criterion(target, labels)  # Find the Loss
        valid_loss = loss.item() * data.size(0)  # Calculate Loss

        # Update progress bar
        validation_loop.set_description(f"Epoch [{e + 1}/{epochs}], validation progress")
        validation_loop.set_postfix(loss=valid_loss/len(validloader))

        # Saving State Dict
        # torch.save(model.state_dict(), 'torch_model.h5')

print("Done!")
torch.save(model.state_dict(), 'torch_model.h5')
