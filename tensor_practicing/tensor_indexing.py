import torch

batch_size = 10
features = 25

x = torch.rand((batch_size, features))

print(x[0].shape)
print(x[:, 0].shape)

print(x[2, 0:10])

x = torch.rand(3, 5)
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(x[rows, cols].shape)

x = torch.arange(10)
print(x[(x < 2) | (x > 8)])

print(x[x.remainder(2) == 0])

print(torch.where(x > 5, x, x*2))

print(torch.tensor([0, 0, 1, 1, 2,2, 3, 4]).unique())

print(x.ndimension())

print(x.numel())