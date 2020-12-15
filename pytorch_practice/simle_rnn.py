import torch
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * NetworkParameters.sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out


class NetworkParameters:
    input_size = 28
    sequence_length = 28
    num_layers = 2
    hidden_size = 256
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

            x = x.to(device=device).squeeze(1)
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

model = RNN(input_size=NetworkParameters.input_size, hidden_size=NetworkParameters.hidden_size,
            num_classes=NetworkParameters.num_class, num_layers=NetworkParameters.num_layers).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=NetworkParameters.learning_rate)

for epoch in tqdm(range(NetworkParameters.epochs)):

    for batch_idx, (data, targets) in enumerate(train_loader):

        data = data.to(device=device).squeeze(1)
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
