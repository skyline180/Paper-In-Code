import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class DistillationLoss(nn.Module):
    def __init__(self, temperature=3.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        # Standard cross-entropy loss
        ce = self.ce_loss(student_logits, labels)

        # Soft targets (distillation loss)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)

        kd = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.temperature ** 2)

        return self.alpha * ce + (1 - self.alpha) * kd


def train_distillation(student, teacher, dataloader, device,
                       epochs=10, lr=1e-3, temperature=3.0, alpha=0.5):

    teacher.eval()  # Teacher is fixed
    student.train()

    optimizer = Adam(student.parameters(), lr=lr)
    criterion = DistillationLoss(temperature, alpha)

    for epoch in range(epochs):
        total_loss = 0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                teacher_logits = teacher(images)

            student_logits = student(images)

            loss = criterion(student_logits, teacher_logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Accuracy: {acc:.2f}%")
    return acc
