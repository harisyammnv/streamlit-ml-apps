import torch
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets


class CNN(nn.Module):

    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(64*7*7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


class NetworkParameters:
    input_size = 784
    num_class = 10
    learning_rate = 0.001
    batch_size = 64
    epochs = 10


def check_accuracy(loader, model):

    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device=device)
            y = y.to(device=device)
            #x = x.reshape(x.shape[0], -1)
            scores = model(x)
            _, predictions = scores.max(1) # output shape is 64x10 --> take max among the cols
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        acc = (float(num_correct)/float(num_samples))*100
        print(f"Got {num_correct} / {num_samples} with accuracy {acc:.2f}")

    model.train()
    return acc


device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataset = datasets.FashionMNIST(root='./dataset', train=True, transform=transforms.ToTensor(),
                                      download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=NetworkParameters.batch_size, shuffle=True)

test_dataset = datasets.FashionMNIST(root='./dataset', train=False, transform=transforms.ToTensor(),
                                     download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=NetworkParameters.batch_size, shuffle=True)

model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=NetworkParameters.learning_rate)

for epoch in tqdm(range(NetworkParameters.epochs)):

    for batch_idx, (data, targets) in enumerate(train_loader):

        data = data.to(device=device)
        targets = targets.to(device=device)

        #data = data.reshape(NetworkParameters.batch_size, -1)
        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
