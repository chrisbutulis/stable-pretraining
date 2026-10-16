"""ReSSL pretraining on CIFAR-10 using stable_pretraining.

Run from your existing project so the parent-level utils.py is importable.
Uses stable_pretraining.forward.ressl for the training forward pass.
"""

import sys
from pathlib import Path

import lightning as pl
import torch
from torch import nn
from lightning.pytorch.loggers import WandbLogger
import torchmetrics
import torchvision

import stable_pretraining as spt
from stable_pretraining.backbone.ressl import ReSSLTeacherStudentWrapper
from stable_pretraining.backbone.split_batchnorm import SplitBatchNorm
from stable_pretraining.data import transforms
from stable_pretraining.callbacks.queues import UnsortedQueue
from stable_pretraining import forward

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import get_data_dir


CIFAR10_STATS = {
    "mean": [0.4914, 0.4822, 0.4465],
    "std": [0.2023, 0.1994, 0.2010],
}

BATCH_SIZE = 256
MAX_EPOCHS = 200
WARMUP_EPOCHS = 5
NUM_SPLITS = 8

def replace_batchnorm(module: nn.Module, num_splits: int = NUM_SPLITS):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            replacement = SplitBatchNorm(
                child.num_features,
                num_splits=num_splits,
                eps=child.eps,
                momentum=child.momentum,
                affine=child.affine,
                track_running_stats=child.track_running_stats,
            )
            replacement.load_state_dict(child.state_dict())
            setattr(module, name, replacement)
        else:
            replace_batchnorm(child, num_splits)

def main():
    pl.seed_everything(42, workers=True)

    weak_transform = transforms.Compose(
        transforms.RandomResizedCrop((32, 32), scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToImage(**CIFAR10_STATS),
    )
    strong_transform = transforms.Compose(
        transforms.RandomResizedCrop((32, 32), scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8,
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.PILGaussianBlur(sigma=(0.1, 2.0), p=0.5),
        transforms.ToImage(**CIFAR10_STATS),
    )
    train_transform = transforms.MultiViewTransform([weak_transform, strong_transform])
    val_transform = transforms.Compose(transforms.ToImage(**CIFAR10_STATS))

    data_dir = get_data_dir("cifar10")
    cifar_train = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=True, download=True,
    )
    cifar_val = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=False, download=True,
    )
    train_dataset = spt.data.FromTorchDataset(
        cifar_train, names=["image", "label"], transform=train_transform,
    )
    val_dataset = spt.data.FromTorchDataset(
        cifar_val, names=["image", "label"], transform=val_transform,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, num_workers=0,
        drop_last=True, shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, num_workers=0,
    )
    steps_per_epoch = len(train_loader)
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    max_steps = MAX_EPOCHS * steps_per_epoch
    print(f"Steps/epoch: {steps_per_epoch}; warmup: {warmup_steps}; total: {max_steps}")
    data = spt.data.DataModule(train=train_loader, val=val_loader)

    base = spt.backbone.from_torchvision("resnet18", low_resolution=True)
    base.fc = nn.Identity()
    replace_batchnorm(base)
    backbone = ReSSLTeacherStudentWrapper(
        base, warm_init=True,
        base_ema_coefficient=0.99, final_ema_coefficient=0.99,
    )
    projector = ReSSLTeacherStudentWrapper(
        nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 128),
        ),
        warm_init=True,
        base_ema_coefficient=0.99, final_ema_coefficient=0.99,
    )

    # A full, randomly initialized memory bank matches the original ReSSL setup.
    # Initialize after seeding and register as a child module for checkpointing.
    queue = UnsortedQueue(max_length=4096, shape=128)
    with torch.no_grad():
        initial_keys = torch.nn.functional.normalize(
            torch.randn(4096, 128), dim=-1
        )
        queue.append(initial_keys)

    module = spt.Module(
        backbone=backbone,
        projector=projector,
        forward=forward.ressl,
        ressl_loss=spt.losses.ReSSLLoss(
            temperature_student=0.1, temperature_teacher=0.04,
        ),
        queue=queue,
        optim={
            "optimizer": {
                "type": "SGD", "exclude_bias_norm": False,
                "momentum": 0.9, "lr": 0.06 * BATCH_SIZE / 256,
                "weight_decay": 5e-4,
            },
            "scheduler": {
                "type": "LinearWarmupCosineAnnealingLR",
                "warmup_steps": warmup_steps,
                "max_steps": max_steps,
                "warmup_start_lr": 0.0,
                "eta_min": 0.0,
            },
            "interval": "step",
        },
    )

    # Ensure the queue's state is tracked by Lightning checkpoints.
    assert "queue.out" in module.state_dict(), "Queue was not registered"

    linear_probe = spt.callbacks.OnlineProbe(
        module, name="linear_probe", input="embedding", target="label",
        probe=nn.Linear(512, 10), loss=nn.CrossEntropyLoss(),
        metrics={
            "top1": torchmetrics.classification.MulticlassAccuracy(10),
            "top5": torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
        },
    )
    knn_probe = spt.callbacks.OnlineKNN(
        name="knn_probe", input="embedding", target="label",
        queue_length=20000,
        metrics={"accuracy": torchmetrics.classification.MulticlassAccuracy(10)},
        input_dim=512, k=10,
    )

    logger = WandbLogger(
        entity="stable-ssl", project="cifar10-ressl",
        name="ressl-resnet18", log_model=False,
    )
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        num_sanity_val_steps=0,
        callbacks=[
            knn_probe, linear_probe,
            spt.callbacks.TeacherStudentCallback(
                update_frequency=1, update_after_backward=False,
            ),
        ],
        precision="32-true",
        logger=logger,
        enable_checkpointing=True,
    )
    manager = spt.Manager(trainer=trainer, module=module, data=data)
    manager()


if __name__ == "__main__":
    main()
