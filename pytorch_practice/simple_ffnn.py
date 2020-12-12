import torch
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets


class FFNN(nn.Module):

    def __init__(self, input_size, num_classes):
        super(FFNN, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        self.fc1 = nn.Linear(self.input_size, 50)
        self.fc2 = nn.Linear(50, self.num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class NetworkParameters:
    input_size = 784
    num_class = 10
    learning_rate = 0.001
    batch_size = 60
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
            x = x.reshape(x.shape[0], -1)
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

model = FFNN(input_size=NetworkParameters.input_size, num_classes=NetworkParameters.num_class).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=NetworkParameters.learning_rate)

for epoch in tqdm(range(NetworkParameters.epochs)):

    for batch_idx, (data, targets) in enumerate(train_loader):

        data = data.to(device=device)
        targets = targets.to(device=device)

        data = data.reshape(NetworkParameters.batch_size, -1)
        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
