import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

if __name__ == '__main__':

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Hyper parameters
    num_epochs = 30  # 5
    classes = 33  # 33
    batch_size = 40
    learning_rate = 0.0001

    # MNIST dataset
    # train_dataset = torchvision.datasets.MNIST(root='../../data/',
    #                                            train=True,
    #                                            transform=transforms.ToTensor(),
    #                                            download=True)
    #
    # test_dataset = torchvision.datasets.MNIST(root='../../data/',
    #                                           train=False,
    #                                           transform=transforms.ToTensor())

    trainTransform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                         transforms.ToTensor()])

    TRAINING_DIR = 'data/training'
    train_dataset = torchvision.datasets.ImageFolder(root=TRAINING_DIR, transform=trainTransform)

    TESTING_DIR = 'data/testing'
    test_dataset = torchvision.datasets.ImageFolder(root=TESTING_DIR, transform=trainTransform)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=4)


    class ConvNet(nn.Module):
        def __init__(self, num_classes=classes):
            super(ConvNet, self).__init__()
            # 40x3x28x28, 'cause batch size is 40, number of channels is 1 (grayscale) and image size is 28x28
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(3, 3), padding=2, bias=False),  # took 40x1x28x28, produced 40x16x30x30
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)  # , stride=2)  # took 40x16x30x30, produced 40x16x15x15
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=(3, 3), padding=2, bias=False),  # took 40x16x15x15, produced 40x32x17x17
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)  # , stride=2)  # took 40x32x17x17, produced 40x32x8x8
            )
            self.layer3 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, bias=False),  # took 40x32x8x8, produced 40x64x10x10
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)  # , stride=2)  # took 40x64x10x10, produced 40x64x5x5
            )
            self.fc1 = nn.Linear(64 * 5 * 5, 512)  # took 40x(64*5*5)=40x1600, produced 40x512
            self.relu = nn.ReLU()  # TEST
            self.fc2 = nn.Linear(512, num_classes)  # took 40x512, produced 40x33

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = out.reshape(out.size(0), -1)
            out = self.fc1(out)
            out = self.relu(out)
            out = self.fc2(out)
            return out


    model = ConvNet(classes).to(device)

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
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if (i + 1) % print_freq == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                epoch_loss += loss.item()
        print('EPOCH ', epoch + 1, ' LOSS = ', epoch_loss)

    # Test the model
    model.eval()  # eval mode
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
    torch.save(model.state_dict(), 'model.h5')
