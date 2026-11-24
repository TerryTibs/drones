"""
=============================================================================
STEP 2: MULTI-MODAL DUAL-HEAD NEURAL FLIGHT EXECUTIVE POLICY
File: drone_neural_policy_network.py
=============================================================================
Deep Neural Policy featuring:
1. Continuous Motor Output Head (4x normalized rotor commands via Sigmoid)
2. Discrete Operational Executive Head (6x flight phase logits via CrossEntropy)
3. Shared LayerNorm + SiLU latent trunk with residual feature extraction.
=============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OBSERVATION_DIM = 32
NUM_MOTORS = 4
NUM_EXECUTIVE_STATES = 6

# Categorical Executive Action Tokens
ACTION_NAMES = [
    "TRANSIT",             # 0: Flying en-route to dropzone along spline
    "HOVER_LOCK",          # 1: Precision position-hold over target basket
    "WINCH_LOWER",         # 2: Motorized deployment of tether line
    "TOUCHDOWN_RELEASE",   # 3: Ground contact sensed, package unlatched
    "WINCH_RETRACT",       # 4: Empty hook reeling back to airframe belly
    "AUTO_DISPATCH"        # 5: Hook stowed, selecting next delivery target
]

class MultiModalFlightExecutiveNet(nn.Module):
    """
    Dual-Head Physics & Mission Executive Neural Policy:
    Maps 32-dimensional normalized sensory state -> (Motor_Commands, Executive_Logits)
    """
    def __init__(self, in_features=OBSERVATION_DIM):
        super().__init__()

        # --- 1. Shared Multi-Modal Representation Backbone ---
        self.shared_backbone = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.SiLU()
        )

        # Residual projection layer to preserve gradient flow
        self.skip_proj = nn.Linear(in_features, 128)

        # --- 2. Continuous Motor Actuation Head ---
        # Outputs 4 normalized motor commands in [0.0, 1.0] (RPM / MAX_RPM)
        self.motor_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, NUM_MOTORS),
            nn.Sigmoid()  # Guarantees non-negative, bounded motor throttle
        )

        # --- 3. Discrete Executive FSM Decision Head ---
        # Outputs 6 logits corresponding to macro mission phases
        self.executive_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, NUM_EXECUTIVE_STATES)
        )

    def forward(self, x):
        """
        Forward pass:
        - x: Tensor of shape (Batch_Size, 32)
        Returns:
        - motor_cmds: Tensor of shape (Batch_Size, 4) in [0.0, 1.0]
        - executive_logits: Tensor of shape (Batch_Size, 6) unnormalized logits
        """
        latent = self.shared_backbone(x) + self.skip_proj(x)

        motor_cmds = self.motor_head(latent)
        executive_logits = self.executive_head(latent)

        return motor_cmds, executive_logits

    def predict_action(self, obs_np, device="cpu"):
        """
        Inference helper for real-time 60 Hz simulation loop:
        Converts single numpy observation vector -> (motor_rpms, active_phase_token, confidence)
        """
        self.eval()
        with torch.no_grad():
            t_in = torch.from_numpy(obs_np).float().unsqueeze(0).to(device)
            motor_norm, logits = self(t_in)

            motor_cmds_out = motor_norm[0].cpu().numpy()

            probs = F.softmax(logits[0], dim=-1).cpu().numpy()
            token_idx = int(np.argmax(probs))
            confidence = float(probs[token_idx])

        return motor_cmds_out, token_idx, ACTION_NAMES[token_idx], confidence


# ==========================================
# VERIFICATION UNIT TEST
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalFlightExecutiveNet().to(device)

    dummy_batch = torch.randn(16, OBSERVATION_DIM, device=device)
    motors, logits = model(dummy_batch)

    dummy_loss = motors.sum() + logits.sum()
    dummy_loss.backward()

    total_params = sum(p.numel() for p in model.parameters())

    print("=========================================================")
    print("STEP 2 VERIFIED: Multi-Modal Flight Executive Architecture Operational")
    print(f"Device: {device}")
    print(f"Input Shape:            {dummy_batch.shape}")
    print(f"Motor Head Shape:       {motors.shape} (Range: [{motors.min():.3f}, {motors.max():.3f}])")
    print(f"Executive Head Shape:   {logits.shape} (6 Action Classes)")
    print(f"Trainable Parameters:   {total_params:,}")
    print(f"Backward Autograd:      PASSED (Gradients computed successfully)")
    print("=========================================================")