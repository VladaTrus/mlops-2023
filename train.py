import glob
import os
import shutil
import subprocess

import hydra
import mlflow
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from animal_dataset import AnimalDataset


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg: DictConfig):
    """
    Training of model
    """

    if os.path.exists(cfg["paths"]["dir"]):
        shutil.rmtree("data")

    os.system("dvc pull")

    # path = cfg['paths']['train']
    path = cfg["paths"]["dir"] + "/" + cfg["paths"]["train"]

    class_names = cfg["data"]["class_names"]

    img_paths, labels = [], []
    for img_path in glob.glob(os.path.join(path, "*/*")):
        class_i = img_path.split(os.sep)[-2]
        if class_i not in class_names.keys():
            continue
        img_paths.append(img_path)
        labels.append(class_names[class_i])

    transform = transforms.Compose(
        [
            torchvision.transforms.RandomRotation(cfg["train_transform"]["random_rotation"]),
            torchvision.transforms.RandomPerspective(),
            transforms.Resize(cfg["train_transform"]["resize"]),
            transforms.CenterCrop(cfg["train_transform"]["center_crop"]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=cfg["train_transform"]["normalize_mean"], std=cfg["train_transform"]["normalize_std"]
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        img_paths, labels, test_size=cfg["data"]["test_size"], random_state=cfg["data"]["random_state"]
    )

    train_data = AnimalDataset(X_train, y_train, transform)
    valid_data = AnimalDataset(X_test, y_test, transform)
    trainloader = DataLoader(dataset=train_data, batch_size=cfg["model_params"]["batch_size"], shuffle=True)
    validloader = DataLoader(dataset=valid_data, batch_size=cfg["model_params"]["batch_size"], shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = models.resnet18(weights="ResNet18_Weights.DEFAULT")
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, cfg["model_params"]["cnt_out"])
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    params = net.parameters()
    optimizer = torch.optim.Adam(params, lr=cfg["model_params"]["lr"])

    epochs = cfg["model_params"]["epochs"]
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
        for data, target, _ in trainloader:
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
        for data, target, _ in validloader:
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

    torch.save(net, cfg["paths"]["model"])

    if cfg["mlflow_params"]["run_mlflow"]:
        mlflow.set_tracking_uri(uri=cfg["mlflow_params"]["uri"])

        mlflow.set_experiment("image_classification")
        with mlflow.start_run():

            mlflow.log_params(cfg["model_params"])
            mlflow.log_params(cfg["train_transform"])
            mlflow.log_param(
                "git commit id", subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
            )

            for i in range(len(train_losses)):
                mlflow.log_metric("Train loss", train_losses[i])
                mlflow.log_metric("Validation loss", valid_losses[i])
                mlflow.log_metric("Train accuracy", train_accuracy[i])
                mlflow.log_metric("Validation accuracy", valid_accuracy[i])


if __name__ == "__main__":
    main()
