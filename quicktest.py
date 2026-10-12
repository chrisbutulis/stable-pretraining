import torch

from stable_pretraining.methods.ressl import ReSSL
from stable_pretraining.losses import ReSSLLoss


def test_ressl_forward_shapes():
    """ReSSL should run end-to-end and return expected tensor shapes."""
    model = ReSSL(queue_size=16)

    weak = torch.randn(2, 3, 224, 224)
    strong = torch.randn(2, 3, 224, 224)

    out = model(weak, strong)

    assert out.loss.ndim == 0
    assert torch.isfinite(out.loss)

    assert out.student_embedding.shape == (2, 512)
    assert out.teacher_embedding.shape == (2, 512)

    assert model.queue.shape == (16, 512)


def test_ressl_queue_updates():
    """A forward pass should insert teacher projections into the queue."""
    model = ReSSL(queue_size=16)

    weak = torch.randn(2, 3, 224, 224)
    strong = torch.randn(2, 3, 224, 224)

    queue_before = model.queue.clone()
    ptr_before = model.queue_ptr.item()

    model(weak, strong)

    assert not torch.equal(queue_before, model.queue)

    expected_ptr = (
        ptr_before + weak.shape[0]
    ) % model.queue.shape[0]

    assert model.queue_ptr.item() == expected_ptr


def test_ressl_queue_wraparound():
    """The circular queue pointer should wrap around the queue boundary."""
    model = ReSSL(queue_size=4)

    keys = torch.randn(3, 512)

    model.queue_ptr[0] = 3
    model._dequeue_and_enqueue(keys)

    expected_ptr = (3 + keys.shape[0]) % model.queue.shape[0]

    assert model.queue_ptr.item() == expected_ptr
    assert model.queue_ptr.item() == 2


def test_ressl_queue_contains_new_keys():
    """New teacher embeddings should overwrite the oldest queue entries."""
    model = ReSSL(queue_size=4)

    keys = torch.randn(2, 512)

    model.queue_ptr[0] = 0
    model._dequeue_and_enqueue(keys)

    assert torch.allclose(model.queue[0], keys[0])
    assert torch.allclose(model.queue[1], keys[1])
    assert model.queue_ptr.item() == 2


def test_ressl_queue_wraparound_contents():
    """Queue values should be written correctly when insertion wraps around."""
    model = ReSSL(queue_size=4)

    keys = torch.randn(3, 512)

    model.queue_ptr[0] = 3
    model._dequeue_and_enqueue(keys)

    assert torch.allclose(model.queue[3], keys[0])
    assert torch.allclose(model.queue[0], keys[1])
    assert torch.allclose(model.queue[1], keys[2])


def test_ressl_backward():
    """The ReSSL loss should support backpropagation."""
    model = ReSSL(queue_size=16)

    weak = torch.randn(2, 3, 224, 224)
    strong = torch.randn(2, 3, 224, 224)

    out = model(weak, strong)

    out.loss.backward()

    assert out.loss.grad_fn is not None


def test_ressl_student_backbone_gets_gradients():
    """Student backbone parameters should receive gradients."""
    model = ReSSL(queue_size=16)

    weak = torch.randn(2, 3, 224, 224)
    strong = torch.randn(2, 3, 224, 224)

    out = model(weak, strong)
    out.loss.backward()

    student_grads = [
        param.grad
        for param in model.backbone.student.parameters()
        if param.requires_grad
    ]

    assert any(grad is not None for grad in student_grads)


def test_ressl_teacher_backbone_has_no_gradients():
    """Teacher backbone parameters should not receive gradients."""
    model = ReSSL(queue_size=16)

    weak = torch.randn(2, 3, 224, 224)
    strong = torch.randn(2, 3, 224, 224)

    out = model(weak, strong)
    out.loss.backward()

    teacher_grads = [
        param.grad
        for param in model.backbone.teacher.parameters()
    ]

    assert all(grad is None for grad in teacher_grads)


def test_ressl_student_projector_gets_gradients():
    """Student projection-head parameters should receive gradients."""
    model = ReSSL(queue_size=16)

    weak = torch.randn(2, 3, 224, 224)
    strong = torch.randn(2, 3, 224, 224)

    out = model(weak, strong)
    out.loss.backward()

    student_grads = [
        param.grad
        for param in model.projection_head.student.parameters()
        if param.requires_grad
    ]

    assert any(grad is not None for grad in student_grads)


def test_ressl_teacher_projector_has_no_gradients():
    """Teacher projection-head parameters should not receive gradients."""
    model = ReSSL(queue_size=16)

    weak = torch.randn(2, 3, 224, 224)
    strong = torch.randn(2, 3, 224, 224)

    out = model(weak, strong)
    out.loss.backward()

    teacher_grads = [
        param.grad
        for param in model.projection_head.teacher.parameters()
    ]

    assert all(grad is None for grad in teacher_grads)


def test_ressl_loss_scalar():
    """ReSSLLoss should return a finite scalar."""
    loss_fn = ReSSLLoss()

    student = torch.randn(4, 512, requires_grad=True)
    teacher = torch.randn(4, 512)
    queue = torch.randn(32, 512)

    loss = loss_fn(student, teacher, queue)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_ressl_loss_backward():
    """ReSSLLoss should propagate gradients to student embeddings."""
    loss_fn = ReSSLLoss()

    student = torch.randn(4, 512, requires_grad=True)
    teacher = torch.randn(4, 512)
    queue = torch.randn(32, 512)

    loss = loss_fn(student, teacher, queue)
    loss.backward()

    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_ressl_loss_does_not_backprop_to_teacher():
    """Teacher embeddings should act as fixed targets in ReSSLLoss."""
    loss_fn = ReSSLLoss()

    student = torch.randn(4, 512, requires_grad=True)
    teacher = torch.randn(4, 512, requires_grad=True)
    queue = torch.randn(32, 512)

    loss = loss_fn(student, teacher, queue)
    loss.backward()

    assert student.grad is not None
    assert teacher.grad is None