"""ReSSL Loss.

This module contains losses for ___________:

Reference: __________
"""

import torch
import torch.nn.functional as F

class ReSSLLoss(torch.nn.Module):
    """Relational self-supervised learning loss.

    Computes cross-entropy between teacher and student relational
    distributions after temperature scaling.

    The teacher distribution is treated as a fixed target, while the
    student distribution is optimized to match it.

    Args:
        temperature_student (float): Temperature for the student relational
            distribution.
        temperature_teacher (float): Temperature for the teacher relational
            distribution.
    """

    def __init__(
        self,
        temperature_student: float = 0.1,
        temperature_teacher: float = 0.05,
    ):
        super().__init__()
        self.temperature_student = temperature_student
        self.temperature_teacher = temperature_teacher

    def forward(
        self,
        student_embeddings: torch.Tensor,
        teacher_embeddings: torch.Tensor,
        queue: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the ReSSL relational cross-entropy loss.

        Args:
            student_embeddings: Student relational similarity logits.
            teacher_embeddings: Teacher relational similarity logits.

        Returns:
            Scalar cross-entropy loss averaged over the batch.

        Shape:
            - student_embeddings: (B, K), where B is the batch size and K is
              the queue size.
            - teacher_embeddings: (B, K), where B is the batch size and K is
              the queue size.
            - output: scalar.
        """
        # Apply temperature-scaled log-softmax to student
        student_logits = student_embeddings @ queue.T
        student_log_probs = F.log_softmax(
            student_logits.float() / self.temperature_student, dim=-1
        )

        # Apply temperature-scaled softmax to teacher
        teacher_logits = teacher_embeddings @ queue.T
        teacher_probs = F.softmax(
            teacher_logits.detach().float() / self.temperature_teacher, dim=-1
        )

        return -torch.sum(teacher_probs * student_log_probs, dim=-1).mean()
