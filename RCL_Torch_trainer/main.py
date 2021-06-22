import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 11  # 5
num_classes = 33
batch_size = 40
learning_rate = 0.001

# MNIST dataset for alternative testing purposes
# train_dataset = torchvision.datasets.MNIST(root='../../data/',
#                                            train=True,
#                                            transform=transforms.ToTensor(),
#                                            download=True)
#
# test_dataset = torchvision.datasets.MNIST(root='../../data/',
#                                           train=False,
#                                           transform=transforms.ToTensor())

# Tried this to convert rgb images to grayscale, something went wrong
# trainTransform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
#                                      transforms.ToTensor(),
#                                      transforms.Normalize(0.5, 0.5)])

TRAINING_DIR = 'data/training'
train_dataset = torchvision.datasets.ImageFolder(root=TRAINING_DIR, transform=transforms.ToTensor())

TESTING_DIR = 'data/testing'
test_dataset = torchvision.datasets.ImageFolder(root=TESTING_DIR, transform=transforms.ToTensor())


# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Convolutional neural network (three convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(ConvNet, self).__init__()
        # 40x3x278x278, 'cause batch size is 40, number of channels is 3 (RGB) and image size is 278x278
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3)),  # took 40x3x278x278, produced 40x16x276x276
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # took 40x16x276x276, produced 40x16x138x138
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 3)),  # took 40x16x138x138, produced 40x32x136x136
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # took 40x32x136x136, produced 40x32x68x68
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3)),  # took 40x32x68x68, produced 40x64x66x66
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # took 40x64x66x66, produced 40x64x33x33
        )
        self.fc1 = nn.Linear(64*33*33, 512)  # took 40x(64*33*33)=40x69696, produced 40x512
        self.fc2 = nn.Linear(512, num_classes)  # took 40x512, produced 40x33

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


model = ConvNet(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
print_freq = 107  # for batch_size=40, 'cause then we have 1391 batches, 1391/107=13 - looks clean
total_step = len(train_loader)
for epoch in range(num_epochs):
    epoch_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % print_freq == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
            epoch_loss += loss.item()
    print('EPOCH ', epoch + 1, ' LOSS = ', epoch_loss)

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on test dataset: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model, 'model.h5')
