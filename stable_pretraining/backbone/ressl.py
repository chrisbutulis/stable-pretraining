import stable_pretraining as spt
import torch

class ReSSLTeacherStudentWrapper(spt.TeacherStudentWrapper):
    """EMA teacher that updates parameters but not BatchNorm buffers.

    ReSSL's reference training keeps teacher BatchNorm statistics
    independent of the student while updating teacher weights via EMA.
    """

    @torch.no_grad()
    def update_teacher(self):
        if not self.training or self.teacher is self.student:
            return

        momentum = float(self.ema_coefficient)
        if momentum == 1.0:
            return

        student_params = list(self.student.parameters())
        teacher_params = list(self.teacher.parameters())
        if len(student_params) != len(teacher_params):
            raise RuntimeError("Student and teacher parameter counts differ")

        for student_param, teacher_param in zip(student_params, teacher_params):
            teacher_param.lerp_(student_param, 1.0 - momentum)