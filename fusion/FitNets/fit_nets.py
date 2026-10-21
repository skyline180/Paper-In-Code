
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Hint Regressor
# ============================================================

class HintRegressor(nn.Module):
    """
    Maps student hint features to the teacher hint feature space.

    Input:
        student_hint: [B, Cs, Hs, Ws]
        teacher_hint: [B, Ct, Ht, Wt]

    Output:
        mapped_student: [B, Ct, Ht, Wt]
    """

    def __init__(self, student_channels, teacher_channels):
        super().__init__()

        self.regressor = nn.Conv2d(
            student_channels,
            teacher_channels,
            kernel_size=1,
            bias=True
        )

    def forward(self, student_hint, teacher_hint):
        mapped_student = self.regressor(student_hint)

        # Align spatial dimensions when necessary.
        if mapped_student.shape[2:] != teacher_hint.shape[2:]:
            mapped_student = F.interpolate(
                mapped_student,
                size=teacher_hint.shape[2:],
                mode="bilinear",
                align_corners=False
            )

        return mapped_student


# ============================================================
# 2. FitNets Loss
# ============================================================

class HintLoss(nn.Module):
    """
    Hint loss between projected student features
    and teacher features.
    """

    def __init__(self, student_channels, teacher_channels):
        super().__init__()

        self.regressor = HintRegressor(
            student_channels,
            teacher_channels
        )

    def forward(self, student_hint, teacher_hint):
        mapped_student = self.regressor(
            student_hint,
            teacher_hint
        )

        # Mean squared error between hint representations.
        loss = F.mse_loss(
            mapped_student,
            teacher_hint
        )

        return loss


# ============================================================
# 3. Model Output Convention
# ============================================================

# Both teacher and student must return:
#
#     logits, hint_features = model(inputs)
#
# logits:
#     [B, num_classes]
#
# hint_features:
#     [B, channels, height, width]
#
# Select the hint layer in each model's forward() method.
#
# Example:
#
# def forward(self, x):
#     x = self.stem(x)
#     hint = self.hint_layer(x)
#     x = self.remaining_layers(hint)
#     logits = self.classifier(self.pool(x).flatten(1))
#     return logits, hint


# ============================================================
# 4. Stage One: Hint-Based Pretraining
# ============================================================

def train_hint_stage(
    teacher,
    student,
    hint_criterion,
    train_loader,
    optimizer_hint,
    device,
    epochs=10
):
    """
    Train the student using intermediate teacher hints.

    The optimizer must include:
        - Student parameters
        - Hint regressor parameters
    """

    teacher.eval()
    teacher.requires_grad_(False)

    student.train()
    hint_criterion.train()

    for epoch in range(epochs):
        running_loss = 0.0
        num_samples = 0

        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            batch_size = inputs.size(0)

            optimizer_hint.zero_grad(set_to_none=True)

            # Teacher features are targets, not trainable outputs.
            with torch.no_grad():
                _, teacher_hint = teacher(inputs)

            # Student hint features retain their gradients.
            _, student_hint = student(inputs)

            loss = hint_criterion(
                student_hint,
                teacher_hint
            )

            loss.backward()
            optimizer_hint.step()

            running_loss += loss.item() * batch_size
            num_samples += batch_size

        average_loss = running_loss / max(num_samples, 1)

        print(
            f"Hint Stage | Epoch [{epoch + 1}/{epochs}] "
            f"| Hint Loss: {average_loss:.6f}"
        )


# ============================================================
# 5. Stage Two: Student Training
# ============================================================

def train_student_stage(
    teacher,
    student,
    train_loader,
    optimizer_student,
    device,
    epochs=50,
    temperature=4.0,
    hard_weight=0.1,
    kd_weight=0.9
):
    """
    Train the full student using hard labels and,
    optionally, teacher soft targets.

    For hard-label-only training, set:
        hard_weight=1.0
        kd_weight=0.0

    For conventional FitNets-style supervised fine-tuning,
    hard-label-only training is a useful baseline.

    Combining hard labels and KD makes this a FitNets +
    knowledge-distillation hybrid.
    """

    if hard_weight < 0 or kd_weight < 0:
        raise ValueError("Loss weights must be non-negative.")

    if hard_weight == 0 and kd_weight == 0:
        raise ValueError("At least one loss weight must be positive.")

    if temperature <= 0:
        raise ValueError("Temperature must be positive.")

    teacher.eval()
    teacher.requires_grad_(False)

    student.train()

    for epoch in range(epochs):
        running_loss = 0.0
        running_hard_loss = 0.0
        running_kd_loss = 0.0
        correct = 0
        num_samples = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).long()

            optimizer_student.zero_grad(set_to_none=True)

            # Teacher is frozen.
            with torch.no_grad():
                teacher_logits, _ = teacher(inputs)

            # Full student forward pass.
            student_logits, _ = student(inputs)

            # Hard-label classification loss.
            hard_loss = F.cross_entropy(
                student_logits,
                labels
            )

            # Knowledge-distillation loss is optional.
            if kd_weight > 0:
                soft_loss = F.kl_div(
                    F.log_softmax(
                        student_logits / temperature,
                        dim=1
                    ),
                    F.softmax(
                        teacher_logits / temperature,
                        dim=1
                    ),
                    reduction="batchmean"
                ) * (temperature ** 2)
            else:
                soft_loss = student_logits.new_zeros(())

            total_loss = (
                hard_weight * hard_loss
                + kd_weight * soft_loss
            )

            total_loss.backward()
            optimizer_student.step()

            batch_size = labels.size(0)

            running_loss += total_loss.item() * batch_size
            running_hard_loss += hard_loss.item() * batch_size
            running_kd_loss += soft_loss.item() * batch_size

            correct += (
                student_logits.argmax(dim=1) == labels
            ).sum().item()

            num_samples += batch_size

        denominator = max(num_samples, 1)

        print(
            f"Student Stage | Epoch [{epoch + 1}/{epochs}] "
            f"| Total Loss: {running_loss / denominator:.6f} "
            f"| CE: {running_hard_loss / denominator:.6f} "
            f"| KD: {running_kd_loss / denominator:.6f} "
            f"| Accuracy: {100.0 * correct / denominator:.2f}%"
        )


# ============================================================
# 6. Complete FitNets Training Pipeline
# ============================================================

def fitnets_train(
    teacher,
    student,
    train_loader,
    student_hint_channels,
    teacher_hint_channels,
    device,
    hint_epochs=10,
    student_epochs=50,
    hint_lr=1e-3,
    student_lr=1e-3,
    temperature=4.0,
    hard_weight=0.1,
    kd_weight=0.9
):
    """
    Run hint pretraining followed by student training.

    Returns:
        student, hint_criterion
    """

    teacher = teacher.to(device)
    student = student.to(device)

    teacher.eval()
    teacher.requires_grad_(False)

    # Create a trainable hint regressor.
    hint_criterion = HintLoss(
        student_channels=student_hint_channels,
        teacher_channels=teacher_hint_channels
    ).to(device)

    # Stage one: optimize student and hint regressor together.
    optimizer_hint = torch.optim.Adam(
        list(student.parameters())
        + list(hint_criterion.parameters()),
        lr=hint_lr
    )

    print("\n========== STAGE 1: HINT PRETRAINING ==========")

    train_hint_stage(
        teacher=teacher,
        student=student,
        hint_criterion=hint_criterion,
        train_loader=train_loader,
        optimizer_hint=optimizer_hint,
        device=device,
        epochs=hint_epochs
    )

    # Stage two: optimize the student, not the hint regressor.
    optimizer_student = torch.optim.Adam(
        student.parameters(),
        lr=student_lr
    )

    print("\n========== STAGE 2: STUDENT TRAINING ==========")

    train_student_stage(
        teacher=teacher,
        student=student,
        train_loader=train_loader,
        optimizer_student=optimizer_student,
        device=device,
        epochs=student_epochs,
        temperature=temperature,
        hard_weight=hard_weight,
        kd_weight=kd_weight
    )

    return student, hint_criterion


# ============================================================
# 7. Example Usage
# ============================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # TODO: Define your actual teacher and student models.
    # Both models must return (logits, hint_features).
    #
    # teacher = TeacherNet(...)
    # student = StudentNet(...)

    # TODO: Create your training DataLoader.
    # train_loader = ...

    # After defining those objects, run:
    #
    # student, hint_criterion = fitnets_train(
    #     teacher=teacher,
    #     student=student,
    #     train_loader=train_loader,
    #     student_hint_channels=64,
    #     teacher_hint_channels=256,
    #     device=device,
    #     hint_epochs=10,
    #     student_epochs=50,
    #     hint_lr=1e-3,
    #     student_lr=1e-3,
    #     temperature=4.0,
    #     hard_weight=0.1,
    #     kd_weight=0.9
    # )
    #
    # torch.save(student.state_dict(), "fitnets_student.pth")
