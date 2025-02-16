import os
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader
from catalyst import utils
from catalyst.contrib.datasets import MNIST
from torch import nn

N = 18

# utils.set_global_seed(N)
train_dataset = MNIST(root=os.getcwd(), train=True, download=True)
val_dataset = MNIST(root=os.getcwd(), train=False)
train_dataloader = DataLoader(train_dataset, batch_size=128)
val_dataloader = DataLoader(val_dataset, batch_size=128)


class Identical(nn.Module):
    def forward(self, x):
        return x


class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.size(0)
        return x.view(batch_size, -1)


activation = Identical
model = nn.Sequential(
    Flatten(),
    nn.Linear(28 * 28, 128),
    activation(),
    nn.Linear(128, 128),
    activation(),
    nn.Linear(128, 10)
)
# model = torchvision.models.resnet18(pretrained=True)
# ct = 0
# for child in model.children():
#     ct += 1
#     if ct < 4:
#         for param in child.parameters():
#             param.requires_grad = False
#
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
loaders = {"train": train_dataloader, "valid": val_dataloader}
print(loaders)
max_epochs = N
accuracy = {"train": [], "valid": []}
for epoch in range(max_epochs):
    epoch_correct = 0
    epoch_all = 0
    for k, dataloader in loaders.items():
        print(k)
        for x_batch, y_batch in dataloader:
            if k == "train":
                model.train()
                optimizer.zero_grad()
                outp = model(x_batch.float().unsqueeze(1))
            else:
                model.eval()
                with torch.no_grad():
                    outp = model(x_batch.float().unsqueeze(1))
            preds = outp.argmax(-1)
            correct = (preds == y_batch).sum()

            all = len(y_batch)
            epoch_correct += correct.item()
            epoch_all += all
            if k == "train":
                loss = criterion(outp, y_batch)
                loss.backward()
                optimizer.step()
        if k == "train":
            print(f"Epoch: {epoch + 1}")
    print(f"Loader:{k}.Accuracy: {epoch_correct / epoch_all}")
    accuracy[k].append(epoch_correct / epoch_all)
