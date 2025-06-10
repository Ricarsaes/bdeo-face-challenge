# train_pytorch.py
"""
Trains a multi-task PyTorch model to predict age, gender, and eye coordinates
from face images using MobileNetV2 as the backbone architecture.
"""
import os

import torch
from torch import nn, optim

from tqdm import tqdm

from config import DEVICE, EPOCHS, LEARNING_RATE, MAX_AGE
from data_loader_flip import get_data_loaders
from model_pytorch import FaceMultitaskModel

# TODO - fix env variables for multi-threading issues
# I was getting errors related to the number of threads being used
NUM_THREADS = "1"
os.environ["OMP_NUM_THREADS"] = NUM_THREADS
os.environ["MKL_NUM_THREADS"] = NUM_THREADS  # If MKL might be used by NumPy/SciPy
os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS  # If OpenBLAS is used
os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_THREADS  # For macOS Accelerate framework
os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS  # For NumExpr if used

torch.set_num_threads(int(NUM_THREADS))  # For PyTorch's own intra-op parallelism


def train_one_epoch(
    model,
    data_loader,
    criterion_age,
    criterion_gender,
    criterion_eyes,
    optimizer,
    device,
    loss_weights,
):
    model.train()
    total_loss = 0.0
    total_age_loss = 0.0
    total_gender_loss = 0.0
    total_eyes_loss = 0.0

    # Metrics
    age_mae_sum = 0.0
    gender_correct_sum = 0
    gender_total_sum = 0
    eyes_mae_sum = 0.0
    num_samples = 0

    for images, (ages_true, genders_true, eyes_true) in tqdm(
        data_loader, desc="Training"
    ):
        if images is None:
            continue

        images = images.to(device)
        ages_true = ages_true.to(device).unsqueeze(1)
        genders_true = genders_true.to(device)
        eyes_true = eyes_true.to(device)

        optimizer.zero_grad()

        ages_pred, genders_pred, eyes_pred = model(images)

        loss_age = criterion_age(ages_pred, ages_true)
        loss_gender = criterion_gender(genders_pred, genders_true)
        loss_eyes = criterion_eyes(eyes_pred, eyes_true)

        # Weighted sum of losses
        loss = (
            loss_weights["age"] * loss_age
            + loss_weights["gender"] * loss_gender
            + loss_weights["eyes"] * loss_eyes
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_age_loss += loss_age.item() * images.size(0)
        total_gender_loss += loss_gender.item() * images.size(0)
        total_eyes_loss += loss_eyes.item() * images.size(0)

        # Metrics calculation
        num_samples += images.size(0)
        age_mae_sum += torch.abs(ages_pred - ages_true).sum().item()

        preds_gender_binary = (torch.sigmoid(genders_pred) > 0.5).float()
        gender_correct_sum += (preds_gender_binary == genders_true).sum().item()
        gender_total_sum += genders_true.size(0)

        eyes_mae_sum += torch.abs(eyes_pred - eyes_true).sum().item() / 4

    avg_loss = total_loss / num_samples
    avg_age_loss = total_age_loss / num_samples
    avg_gender_loss = total_gender_loss / num_samples
    avg_eyes_loss = total_eyes_loss / num_samples

    avg_age_mae = age_mae_sum / num_samples
    avg_gender_acc = (
        gender_correct_sum / gender_total_sum if gender_total_sum > 0 else 0
    )
    avg_eyes_mae = eyes_mae_sum / num_samples

    return (
        avg_loss,
        avg_age_loss,
        avg_gender_loss,
        avg_eyes_loss,
        avg_age_mae,
        avg_gender_acc,
        avg_eyes_mae,
    )


def evaluate_model(
    model,
    data_loader,
    criterion_age,
    criterion_gender,
    criterion_eyes,
    device,
    loss_weights,
):
    model.eval()
    total_loss = 0.0
    total_age_loss = 0.0
    total_gender_loss = 0.0
    total_eyes_loss = 0.0

    # Metrics
    age_mae_sum = 0.0
    gender_correct_sum = 0
    gender_total_sum = 0
    eyes_mae_sum = 0.0
    num_samples = 0

    with torch.no_grad():
        for images, (ages_true, genders_true, eyes_true) in tqdm(
            data_loader, desc="Evaluating"
        ):
            if images is None:
                continue

            images = images.to(device)
            ages_true = ages_true.to(device).unsqueeze(1)
            genders_true = genders_true.to(device)
            eyes_true = eyes_true.to(device)

            ages_pred, genders_pred, eyes_pred = model(images)

            loss_age = criterion_age(ages_pred, ages_true)
            loss_gender = criterion_gender(genders_pred, genders_true)
            loss_eyes = criterion_eyes(eyes_pred, eyes_true)

            loss = (
                loss_weights["age"] * loss_age
                + loss_weights["gender"] * loss_gender
                + loss_weights["eyes"] * loss_eyes
            )

            total_loss += loss.item() * images.size(0)
            total_age_loss += loss_age.item() * images.size(0)
            total_gender_loss += loss_gender.item() * images.size(0)
            total_eyes_loss += loss_eyes.item() * images.size(0)

            num_samples += images.size(0)
            age_mae_sum += torch.abs(ages_pred - ages_true).sum().item()

            preds_gender_binary = (torch.sigmoid(genders_pred) > 0.5).float()
            gender_correct_sum += (preds_gender_binary == genders_true).sum().item()
            gender_total_sum += genders_true.size(0)

            eyes_mae_sum += torch.abs(eyes_pred - eyes_true).sum().item() / 4

    avg_loss = total_loss / num_samples
    avg_age_loss = total_age_loss / num_samples
    avg_gender_loss = total_gender_loss / num_samples
    avg_eyes_loss = total_eyes_loss / num_samples

    avg_age_mae = age_mae_sum / num_samples
    avg_gender_acc = (
        gender_correct_sum / gender_total_sum if gender_total_sum > 0 else 0
    )
    avg_eyes_mae = eyes_mae_sum / num_samples

    return (
        avg_loss,
        avg_age_loss,
        avg_gender_loss,
        avg_eyes_loss,
        avg_age_mae,
        avg_gender_acc,
        avg_eyes_mae,
    )


def main():
    print(f"Using device: {DEVICE}")

    train_loader, val_loader = get_data_loaders()
    model = FaceMultitaskModel().to(DEVICE)

    # Loss functions
    criterion_age = nn.L1Loss()
    criterion_gender = nn.BCEWithLogitsLoss()
    criterion_eyes = nn.MSELoss()

    loss_weights = {"age": 1.0, "gender": 1.0, "eyes": 5.0}

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5, factor=0.2, min_lr=1e-6
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0
    patience_early_stop = 10

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        train_loss, tr_age_l, tr_gen_l, tr_eye_l, tr_age_mae, tr_gen_acc, tr_eye_mae = (
            train_one_epoch(
                model,
                train_loader,
                criterion_age,
                criterion_gender,
                criterion_eyes,
                optimizer,
                DEVICE,
                loss_weights,
            )
        )
        print(
            f"Train: Loss={train_loss:.4f} | AgeL={tr_age_l:.4f} GenL={tr_gen_l:.4f} EyeL={tr_eye_l:.4f}"
        )
        print(
            f"Train Metrics: AgeMAE={tr_age_mae*MAX_AGE:.2f} GenACC={tr_gen_acc:.4f} EyeMAE_norm={tr_eye_mae:.4f}"
        )

        val_loss, vl_age_l, vl_gen_l, vl_eye_l, vl_age_mae, vl_gen_acc, vl_eye_mae = (
            evaluate_model(
                model,
                val_loader,
                criterion_age,
                criterion_gender,
                criterion_eyes,
                DEVICE,
                loss_weights,
            )
        )
        print(
            f"Val:   Loss={val_loss:.4f} | AgeL={vl_age_l:.4f} GenL={vl_gen_l:.4f} EyeL={vl_eye_l:.4f}"
        )
        print(
            f"Val Metrics:   AgeMAE={vl_age_mae*MAX_AGE:.2f} GenACC={vl_gen_acc:.4f} EyeMAE_norm={vl_eye_mae:.4f}"
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "face_multitask_model_pytorch_best.pth")
            print(f"Best model saved with val_loss: {best_val_loss:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience_early_stop:
            print(
                f"Early stopping triggered after {epochs_no_improve} epochs with no improvement."
            )
            break

    print("\nTraining complete.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("Loading best model for final evaluation examples...")
    model.load_state_dict(torch.load("face_multitask_model_pytorch_best.pth"))

    # Example predictions with the best model
    model.eval()
    print("\nExample predictions from validation set (best model):")
    with torch.no_grad():
        for images, (ages_true, genders_true, eyes_true) in val_loader:
            if images is None:
                continue
            images_dev = images.to(DEVICE)
            pred_ages, pred_genders, pred_eyes = model(images_dev)

            pred_ages_actual = pred_ages.cpu().numpy() * MAX_AGE
            ages_true_actual = ages_true.cpu().numpy() * MAX_AGE
            pred_genders_prob = torch.sigmoid(pred_genders).cpu().numpy()
            pred_genders_binary = (pred_genders_prob > 0.5).astype(int)

            for i in range(min(5, images.size(0))):
                print(f"\nSample {i+1}:")
                print(
                    f"  True Age (Actual): {ages_true_actual[i]:.1f}, Predicted Age (Actual): {pred_ages_actual[i][0]:.1f}"
                )
                print(
                    f"  True Gender: {genders_true[i].item():.0f}, Predicted Gender Prob: {pred_genders_prob[i][0]:.2f} (Pred: {pred_genders_binary[i][0]})"
                )
                print(f"  True Eyes (Normalized): {eyes_true[i].numpy()}")
                print(f"  Predicted Eyes (Normalized): {pred_eyes[i].cpu().numpy()}")
            break  # Show only one batch


if __name__ == "__main__":
    main()
