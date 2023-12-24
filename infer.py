import glob
import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


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


test_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = torch.load("./model_1.pt")
net = net.to(device)
net.eval()


hpath = "./data/val/"
class_names = {"cat": 0, "dog": 1, "wild": 2}
img_paths_test, labels_test = [], []
for img_path in glob.glob(os.path.join(hpath, "*/*")):
    class_i = img_path.split(os.sep)[-2]
    if class_i not in class_names.keys():
        continue
    img_paths_test.append(img_path)
    labels_test.append(class_names[class_i])

test_data = AnimalDataset(img_paths_test, labels_test, test_transform)
testloader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

predictions = []
with torch.no_grad():
    for images, _ in tqdm(testloader):
        images = images.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        predictions.append(predicted.item())
