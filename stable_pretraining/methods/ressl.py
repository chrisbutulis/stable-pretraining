"""ReSSL: Relational Self-Supervised Learning.

Student-teacher self-supervised learning with relational distributions over a
queue of teacher projections.

Reference:
    Zheng et al., "ReSSL: Relational Self-Supervised Learning with
    Weak Augmentation," NeurIPS 2021.
"""

from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone.ressl import ReSSLTeacherStudentWrapper
from stable_pretraining.callbacks.queues import UnsortedQueue
from stable_pretraining.losses import ReSSLLoss


@dataclass
class ReSSLOutput(ModelOutput):
    """Structured output of the ReSSL method."""

    loss: torch.Tensor = None
    student_embedding: torch.Tensor = None
    teacher_embedding: torch.Tensor = None

@torch.no_grad()
def gather_teacher_keys(keys: torch.Tensor) -> torch.Tensor:
    """Gather teacher projections across distributed processes."""
    if not dist.is_available() or not dist.is_initialized():
        return keys

    world_size = dist.get_world_size()
    if world_size == 1:
        return keys

    gathered = [torch.empty_like(keys) for _ in range(world_size)]
    dist.all_gather(gathered, keys.contiguous())
    return torch.cat(gathered, dim=0)

class ReSSL(Module):
    """Relational Self-Supervised Learning with an EMA teacher and memory queue.

    Args:
        encoder_name: Timm backbone name or an instantiated PyTorch backbone.
        encoder_dim: Output feature dimension of the backbone. Inferred from
            num_features when available.
        temperature_teacher: Teacher similarity-distribution temperature.
        temperature_student: Student similarity-distribution temperature.
        queue_size: Number of teacher projections retained in the queue.
        ema_decay_start: Initial teacher EMA momentum.
        ema_decay_end: Final teacher EMA momentum.
        pretrained: Whether to use pretrained weights for a named backbone.
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "resnet18",
        encoder_dim: int | None = None,
        temperature_teacher: float = 0.05,
        temperature_student: float = 0.1,
        queue_size: int = 256,
        ema_decay_start: float = 0.996,
        ema_decay_end: float = 1.0,
        pretrained: bool = False,
    ):
        super().__init__()
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")

        if isinstance(encoder_name, str):
            import timm

            base = timm.create_model(
                encoder_name,
                num_classes=0,
                pretrained=pretrained,
            )
        else:
            base = encoder_name

        if encoder_dim is None:
            encoder_dim = getattr(base, "num_features", None)

        if encoder_dim is None:
            raise ValueError(
                "encoder_dim must be provided when the backbone "
                "does not expose num_features."
            )

        if not isinstance(encoder_dim, int) or encoder_dim <= 0:
            raise ValueError("encoder_dim must be a positive integer")

        self.temperature_teacher = temperature_teacher
        self.temperature_student = temperature_student

        self.backbone = ReSSLTeacherStudentWrapper(
            base,
            warm_init=True,
            base_ema_coefficient=ema_decay_start,
            final_ema_coefficient=ema_decay_end,
        )
        self.projection_head = ReSSLTeacherStudentWrapper(
            nn.Sequential(
                nn.Linear(encoder_dim, 4096),
                nn.ReLU(),
                nn.Linear(4096, 512),
            ),
            warm_init=True,
            base_ema_coefficient=ema_decay_start,
            final_ema_coefficient=ema_decay_end,
        )

        self.queue = UnsortedQueue(max_length=queue_size, shape=512)
        with torch.no_grad():
            initial_keys = F.normalize(torch.randn(queue_size, 512), dim=-1)
            self.queue.append(initial_keys)

        self.loss = ReSSLLoss(
            temperature_student=temperature_student,
            temperature_teacher=temperature_teacher,
        )

    def forward(
        self,
        weak_augmentation: torch.Tensor,
        contrastive_augmentation: torch.Tensor,
    ) -> ReSSLOutput:
        """Compute relational cross-entropy against the current teacher queue."""
        student_h = self.backbone.forward_student(contrastive_augmentation)

        if not self.training:
            return ReSSLOutput(
                loss=None,
                student_embedding=student_h,
                teacher_embedding=None,
            )

        with torch.no_grad():
            teacher_h = self.backbone.forward_teacher(weak_augmentation)
            teacher_z = self.projection_head.forward_teacher(teacher_h)
            teacher_z = F.normalize(teacher_z, dim=-1)

        student_z = self.projection_head.forward_student(student_h)
        student_z = F.normalize(student_z, dim=-1)

        # Clone because appending to the queue changes its storage in place.
        queue_snapshot = self.queue.get().detach().clone()
        loss = self.loss(student_z, teacher_z, queue_snapshot)

        if self.training:
            with torch.no_grad():
                keys = gather_teacher_keys(teacher_z.detach())
                self.queue.append(keys)

        return ReSSLOutput(
            loss=loss,
            student_embedding=student_h,
            teacher_embedding=teacher_h,
        )
