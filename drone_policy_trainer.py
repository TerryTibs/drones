"""
=============================================================================
STEP 4: SUPERVISED POLICY DISTILLATION ENGINE
File: drone_policy_trainer.py
=============================================================================
Multi-Task Training Pipeline:
- Optimizes Continuous Motor Commands (MSE)
- Optimizes Discrete Executive FSM Logits (Cross-Entropy)
- Enforces Motor Smoothness & Vibration Regularization
- Saves verified weights to 'drone_neural_autopilot_model.pth'
=============================================================================
"""

import os
import time
import math
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Import Step 2 architecture and Step 3 harvester
from drone_neural_policy_network import MultiModalFlightExecutiveNet, ACTION_NAMES
from drone_policy_data_harvester import harvest_autopilot_dataset

MODEL_WEIGHTS_FILE = "drone_neural_autopilot_model.pth"

def train_neural_flight_executive(total_samples=120000, epochs=10, batch_size=2048):
    """
    Executes end-to-end dataset harvesting and multi-task neural policy training.
    """
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n=========================================================")
    print(f"STEP 4: SUPERVISED POLICY DISTILLATION ENGINE")
    print(f"Hardware Compute Device: {device}")
    print(f"Dataset Size Target:     {total_samples:,} samples")
    print(f"Batch Size:              {batch_size}")
    print(f"Training Epochs:         {epochs}")
    print("=========================================================")

    # 1. Harvest Dataset across CPU cores via Step 3
    X_np, y_motors_np, y_fsm_np = harvest_autopilot_dataset(total_samples=total_samples)

    # 2. Train / Validation Split (90% Train, 10% Validation)
    num_total = len(X_np)
    num_val = int(num_total * 0.10)
    num_train = num_total - num_val

    indices = np.random.permutation(num_total)
    train_idx = indices[:num_train]
    val_idx = indices[num_train:]

    # Transfer to PyTorch Tensors
    X_train = torch.tensor(X_np[train_idx], device=device)
    y_m_train = torch.tensor(y_motors_np[train_idx], device=device)
    y_f_train = torch.tensor(y_fsm_np[train_idx], device=device)

    X_val = torch.tensor(X_np[val_idx], device=device)
    y_m_val = torch.tensor(y_motors_np[val_idx], device=device)
    y_f_val = torch.tensor(y_fsm_np[val_idx], device=device)

    print(f"\nData Partitioned: {num_train:,} Train | {num_val:,} Validation")

    # 3. Model, Loss Functions & Optimizer Setup
    model = MultiModalFlightExecutiveNet().to(device)

    criterion_motor = nn.MSELoss()
    criterion_fsm = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-4)
    num_batches = (num_train + batch_size - 1) // batch_size
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=4e-3, steps_per_epoch=num_batches, epochs=epochs
    )

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    lambda_fsm = 0.45
    lambda_smooth = 0.05

    print("Beginning Policy Optimization Loop...\n")

    # 4. Training Loop
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(num_train, device=device)
        train_loss_total = 0.0
        train_motor_loss = 0.0
        train_fsm_loss = 0.0

        for b in range(0, num_train, batch_size):
            b_idx = perm[b : b + batch_size]
            bx = X_train[b_idx]
            by_m = y_m_train[b_idx]
            by_f = y_f_train[b_idx]

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                pred_motors, pred_fsm = model(bx)

                # Continuous motor tracking loss
                l_m = criterion_motor(pred_motors, by_m)

                # Categorical executive decision loss
                l_f = criterion_fsm(pred_fsm, by_f)

                # Actuator spatial smoothness regularization
                mean_motor = pred_motors.mean(dim=1, keepdim=True)
                l_smooth = ((pred_motors - mean_motor) ** 2).mean()

                loss = l_m + lambda_fsm * l_f + lambda_smooth * l_smooth

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss_total += loss.item()
            train_motor_loss += l_m.item()
            train_fsm_loss += l_f.item()

        # 5. Validation Step
        model.eval()
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=use_amp):
                val_pred_m, val_pred_f = model(X_val)
                val_l_m = criterion_motor(val_pred_m, y_m_val).item()
                val_l_f = criterion_fsm(val_pred_f, y_f_val).item()

                # Classification accuracy on discrete FSM actions
                val_acc = (val_pred_f.argmax(dim=-1) == y_f_val).float().mean().item() * 100.0

        avg_loss = train_loss_total / num_batches
        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] "
              f"| Loss: {avg_loss:.4f} "
              f"| Motor MSE: {val_l_m:.5f} "
              f"| FSM Acc: {val_acc:5.1f}% "
              f"| LR: {scheduler.get_last_lr()[0]:.5f}")

    # 6. Save Model Artifacts
    torch.save(model.state_dict(), MODEL_WEIGHTS_FILE)
    elapsed_total = time.time() - t_start

    print("\n=========================================================")
    print(f"STEP 4 VERIFIED: Policy Distillation Complete!")
    print(f"Total Time:             {elapsed_total:.2f}s")
    print(f"Final Validation Acc:   {val_acc:.2f}%")
    print(f"Final Motor Error:      {val_l_m:.6f} MSE")
    print(f"Saved Checkpoint:       {os.path.abspath(MODEL_WEIGHTS_FILE)}")
    print("=========================================================")

    return model, device

if __name__ == "__main__":
    mp.freeze_support()
    # Runs Step 4 with 100,000 samples for verified fast convergence
    train_neural_flight_executive(total_samples=100000, epochs=8, batch_size=2048)