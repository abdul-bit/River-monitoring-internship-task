import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import Unet
from matplotlib import pyplot as plt
from Utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 20
NUM_WORKERS = 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 280
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "Dataset/train/SegmentationImages/"
TRAIN_MASK_DIR = "Dataset/train/SegmentationClass/"
VAL_IMG_DIR = "Dataset/val/SegmentationImages/"
VAL_MASK_DIR = "Dataset/val/SegmentationClass/"


def train_fn(train_loader, val_loader, model, optimizer, loss_fn, scaler, train_loss_plt, val_loss_plt):
    train_loop = tqdm(train_loader)
    val_loop = tqdm(val_loader)
    for batch_idx, (data, targets) in enumerate(train_loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # update tqdm loop
        train_loop.set_postfix(loss=loss.item())
    train_loss_plt.append(loss.cpu().detach().numpy())
    for val_batch_idx, (val_data, val_targets) in enumerate(val_loop):
        val_data = val_data.to(device=DEVICE)
        val_targets = val_targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            val_predictions = model(val_data)
            val_loss = loss_fn(val_predictions, val_targets)
        # update tqdm loop
        val_loop.set_postfix(val_loss=val_loss.item())
    val_loss_plt.append(val_loss.cpu().detach().numpy())


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = Unet(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()
    train_loss_plt = []
    val_loss_plt = []
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, val_loader, model, optimizer,
                 loss_fn, scaler, train_loss_plt, val_loss_plt)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )
    plt.plot(train_loss_plt)
    plt.plot(val_loss_plt)
    plt.savefig('loss_plot.png')


if __name__ == "__main__":
    main()
