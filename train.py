import glob
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


path = "./data/train/"

class_names = {"cat": 0, "dog": 1, "wild": 2}

img_paths, labels = [], []
for img_path in glob.glob(os.path.join(path, "*/*")):
    class_i = img_path.split(os.sep)[-2]
    if class_i not in class_names.keys():
        continue
    img_paths.append(img_path)
    labels.append(class_names[class_i])


class AnimalDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        super().__init__()
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name = self.data[index]
        label = self.labels[index]
        image = Image.open(img_name)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


transform = transforms.Compose(
    [
        torchvision.transforms.RandomRotation(15),
        torchvision.transforms.RandomPerspective(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(img_paths, labels, test_size=0.2, random_state=42)

train_data = AnimalDataset(X_train, y_train, transform)
valid_data = AnimalDataset(X_test, y_test, transform)
trainloader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
validloader = DataLoader(dataset=valid_data, batch_size=32, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = models.resnet18(pretrained=True)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 3)
net = net.to(device)
criterion = nn.CrossEntropyLoss()
params = net.parameters()
optimizer = torch.optim.Adam(params, lr=0.001)

epochs = 5
train_losses = []
valid_losses = []
train_accuracy = []
valid_accuracy = []

for epoch in tqdm(range(epochs)):
    train_loss = 0.0
    valid_loss = 0.0
    correct_train = 0
    total_train = 0
    correct_valid = 0
    total_valid = 0

    batch_count = 0

    net.train()
    for data, target in trainloader:
        batch_count += 1
        # print(batch_count)
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
        _, predicted = torch.max(output.data, 1)
        total_train += target.size(0)
        correct_train += (predicted == target).float().sum()

    val_batch = 0
    net.eval()
    for data, target in validloader:
        val_batch += 1
        # print(val_batch)
        data = data.to(device)
        target = target.to(device)
        output = net(data)
        loss = criterion(output, target)
        valid_loss += loss.item() * data.size(0)
        _, predicted = torch.max(output.data, 1)
        total_valid += target.size(0)
        correct_valid += (predicted == target).float().sum()

    train_loss = train_loss / len(trainloader.sampler)
    valid_loss = valid_loss / len(validloader.sampler)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    accuracy_train = correct_train / total_train
    accuracy_valid = correct_valid / total_valid
    train_accuracy.append(accuracy_train)
    valid_accuracy.append(accuracy_valid)

torch.save(net, "./model_1.pt")

epochs = 5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
title_fontsize = 16
axis_fontsize = 12

ax1.plot(range(1, epochs + 1), train_losses, label="Training loss")
ax1.plot(range(1, epochs + 1), valid_losses, label="Validation Loss")
ax1.legend()
ax1.set_xticks(range(1, epochs + 1, 3))
ax1.set_title("Loss", fontsize=title_fontsize)
ax1.set_xlabel("Epoch", fontsize=axis_fontsize)

ax2.plot(range(1, epochs + 1), torch.tensor(train_accuracy, device="cpu"), label="Training Accuracy")
ax2.plot(range(1, epochs + 1), torch.tensor(valid_accuracy, device="cpu"), label="Validation Accuracy")
ax2.legend()
ax2.set_xticks(range(1, epochs + 1, 3))
ax2.set_title("Accuracy", fontsize=title_fontsize)
ax2.set_xlabel("Epoch", fontsize=axis_fontsize)
