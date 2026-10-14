"""ReSSL training on CIFAR-10."""

import lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torchvision
from lightning.pytorch.loggers import WandbLogger

import stable_pretraining as spt
from stable_pretraining import forward
from stable_pretraining.data import transforms
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir

batch_size = 256

ressl_transform = transforms.MultiViewTransform(
    [
        transforms.Compose(
            transforms.RandomResizedCrop((32, 32), scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(**spt.data.static.CIFAR10),
        ),
        transforms.Compose(
            transforms.RandomResizedCrop((32, 32), scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8
            ),
            transforms.GaussianBlur(kernel_size=3, p=0.5),
            transforms.ToImage(**spt.data.static.CIFAR10),
        ),
    ]
)

val_transform = transforms.Compose(
    transforms.RGB(),
    transforms.Resize((32, 32)),
    transforms.ToImage(**spt.data.static.CIFAR10),
)

data_dir = get_data_dir("cifar10")
cifar_train = torchvision.datasets.CIFAR10(
    root=str(data_dir), train=True, download=True
)
cifar_val = torchvision.datasets.CIFAR10(root=str(data_dir), train=False, download=True)

train_dataset = spt.data.FromTorchDataset(
    cifar_train,
    names=["image", "label"],
    transform=ressl_transform,
)
val_dataset = spt.data.FromTorchDataset(
    cifar_val,
    names=["image", "label"],
    transform=val_transform,
)

train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    num_workers=0,
    drop_last=True,
    shuffle=True,
)
val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=0,
)

warmup_epochs = 5
max_epochs = 200
steps_per_epoch = len(train_dataloader)
warmup_steps = warmup_epochs * steps_per_epoch
max_steps = max_epochs * steps_per_epoch

print(f"Steps per epoch: {steps_per_epoch}")
print(f"Warmup steps: {warmup_steps}")
print(f"Total steps: {max_steps}")

data = spt.data.DataModule(train=train_dataloader, val=val_dataloader)

base = spt.backbone.from_torchvision(
    "resnet18",
    low_resolution=True,
)
base.fc = nn.Identity()

backbone = spt.TeacherStudentWrapper(
    base,
    warm_init=True,
    base_ema_coefficient=0.99,
    final_ema_coefficient=0.99,
)

projector = spt.TeacherStudentWrapper(
    nn.Sequential(
        nn.Linear(512, 4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, 512),
    ),
    warm_init=True,
    base_ema_coefficient=0.99,
    final_ema_coefficient=0.99,
)

module = spt.Module(
    backbone=backbone,
    projector=projector,
    forward=forward.ressl,
    ressl_loss=spt.losses.ReSSLLoss(temperature_student=0.1, temperature_teacher=0.05),
    queue=torch.nn.functional.normalize(
        torch.randn(4096, 512),
        dim=1,
    ),
    queue_ptr=torch.zeros(1, dtype=torch.long),
    optim={
        "optimizer": {
            "type": "SGD",
            "exclude_bias_norm": True,
            "momentum": 0.9,
            "lr": 0.06 * batch_size / 256,
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

# TODO: REMOVE
# Register ReSSL queue and pointer as persistent buffers
queue = module.queue
queue_ptr = module.queue_ptr

del module.queue
del module.queue_ptr

module.register_buffer("queue", queue)
module.register_buffer("queue_ptr", queue_ptr)
print("Queue registered:", "queue" in dict(module.named_buffers()))
print("Pointer registered:", "queue_ptr" in dict(module.named_buffers()))

linear_probe = spt.callbacks.OnlineProbe(
    module,
    name="linear_probe",
    input="embedding",
    target="label",
    probe=torch.nn.Linear(512, 10),
    loss=torch.nn.CrossEntropyLoss(),
    metrics={
        "top1": torchmetrics.classification.MulticlassAccuracy(10),
        "top5": torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
    },
)

knn_probe = spt.callbacks.OnlineKNN(
    name="knn_probe",
    input="embedding",
    target="label",
    queue_length=20000,
    metrics={"accuracy": torchmetrics.classification.MulticlassAccuracy(10)},
    input_dim=512,
    k=10,
)

wandb_logger = WandbLogger(
    entity="stable-ssl",
    project="cifar10-ressl",
    name="ressl-resnet18",
    log_model=False,
)

ema_callback = spt.callbacks.TeacherStudentCallback(
    update_frequency=1,
    update_after_backward=False,
)

trainer = pl.Trainer(
    max_epochs=200,
    num_sanity_val_steps=0,
    callbacks=[knn_probe, linear_probe, ema_callback],
    precision="32-true",
    logger=wandb_logger,
    enable_checkpointing=False,
)

manager = spt.Manager(trainer=trainer, module=module, data=data)
manager()
