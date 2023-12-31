import glob
import os

import hydra
import pandas as pd
import torch
import torchvision.transforms as transforms
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from animal_dataset import AnimalDataset


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg: DictConfig):
    """
    Inference of model
    """

    test_transform = transforms.Compose(
        [
            transforms.Resize(cfg["test_transform"]["resize"]),
            transforms.CenterCrop(cfg["test_transform"]["center_crop"]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=cfg["test_transform"]["normalize_mean"], std=cfg["test_transform"]["normalize_std"]
            ),
        ]
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = torch.load(cfg["paths"]["model"])
    net = net.to(device)
    net.eval()

    path = cfg["paths"]["dir"] + "/" + cfg["paths"]["test"]

    class_names = cfg["data"]["class_names"]
    img_paths_test, labels_test = [], []
    for img_path in glob.glob(os.path.join(path, "*/*")):
        class_i = img_path.split(os.sep)[-2]
        if class_i not in class_names.keys():
            continue
        img_paths_test.append(img_path)
        labels_test.append(class_names[class_i])

    test_data = AnimalDataset(img_paths_test, labels_test, test_transform)
    testloader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    predictions = []
    imgs = []
    with torch.no_grad():
        for images, _, image_name in tqdm(testloader):
            images = images.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.append(predicted.item())
            imgs.append(image_name[0])

    df = pd.DataFrame({"filename": imgs, "label": predictions})
    df.to_csv(cfg["paths"]["preds"], index=False)


if __name__ == "__main__":
    main()
