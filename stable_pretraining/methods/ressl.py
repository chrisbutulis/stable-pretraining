"""ReSSL: Relational Self-Supervised Learning.

Implements a student-teacher self-supervised learning method using
relational distributions over a queue of teacher embeddings.

Reference:
    Zheng et al. "ReSSL: Relational Self-Supervised Learning with
    Weak Augmentation." NeurIPS 2021.
"""

from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import TeacherStudentWrapper
from stable_pretraining.losses import ReSSLLoss


@dataclass
class ReSSLOutput(ModelOutput):
    """Structured output of the :class:`ReSSL` SSL method."""

    loss: torch.Tensor = None
    student_embedding: torch.Tensor = None
    teacher_embedding: torch.Tensor = None

class ReSSL(Module):
    """DINOv3: DINOv2 with register tokens + KoLeo.

    :param encoder_name: timm ViT name. Register tokens are added on top of
        the timm model via a ``Parameter``.
    :param n_register_tokens: Number of register tokens (default 4).
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "resnet18",
        temperature_teacher: float = 0.05,
        temperature_student: float = 0.1,
        queue_size: int = 256,
        ema_decay_start: float = 0.996,
        ema_decay_end: float = 1.0,
        pretrained: bool = False,
    ):
        super().__init__()
        if isinstance(encoder_name, str):
            import timm

            base = timm.create_model(
                encoder_name,
                num_classes=0,
                pretrained=pretrained,
            )
        else:
            base = encoder_name

        with torch.no_grad():
            features = base(torch.zeros(1, 3, 224, 224))
            encoder_dim = features.shape[-1]

        self.temperature_teacher = temperature_teacher
        self.temperature_student = temperature_student 

        self.backbone = TeacherStudentWrapper(
            base,
            warm_init=True,
            base_ema_coefficient=ema_decay_start,
            final_ema_coefficient=ema_decay_end,
        )
        self.projection_head = TeacherStudentWrapper(
            nn.Sequential(
                nn.Linear(encoder_dim, 4096),
                nn.ReLU(),
                nn.Linear(4096, 512),
            ),
            warm_init=True,
            base_ema_coefficient=ema_decay_start,
            final_ema_coefficient=ema_decay_end,
        )

        self.register_buffer(
            "queue",
            F.normalize(torch.randn(queue_size, 512), dim=1),
        )
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.loss = ReSSLLoss(temperature_student=temperature_student, temperature_teacher=temperature_teacher)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        ptr = int(self.queue_ptr.item())
        size = self.queue.shape[0]
        batch_size = keys.shape[0]

        if batch_size >= size:
            keys = keys[-size:]
            batch_size = size

        end = ptr + batch_size

        if end <= size:
            self.queue[ptr:end] = keys
        else:
            first = size - ptr
            self.queue[ptr:] = keys[:first]
            self.queue[:end - size] = keys[first:]

        self.queue_ptr[0] = end % size

    def forward(
        self,
        weak_augmentation: torch.Tensor,
        contrastive_augmentation: torch.Tensor,
    ) -> ReSSLOutput:
        # teacher augmentation
        with torch.no_grad(): 
            teacher_h = self.backbone.forward_teacher(weak_augmentation)
            teacher_z = self.projection_head.forward_teacher(teacher_h)
            
        student_h = self.backbone.forward_student(contrastive_augmentation)
        student_z = self.projection_head.forward_student(student_h)

        loss = self.loss(student_z, teacher_z, self.queue.detach().clone())

        self._dequeue_and_enqueue(teacher_z.detach())

        return ReSSLOutput(
            loss=loss,
            student_embedding = student_h,
            teacher_embedding = teacher_h,
        )
