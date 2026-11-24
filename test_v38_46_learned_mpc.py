"""Focused synthetic contract tests for V38.46 learned-transition offline MPC.

These tests check software contracts and fail-closed behavior only. They do not
measure the user's trained checkpoint or establish physical predictive accuracy.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import torch

SOURCE = Path(__file__).with_name("Survival_Core_V38_46_Learned_Transition_Offline_MPC.py")
spec = importlib.util.spec_from_file_location("survival_v3846_test_module", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import source for tests: {SOURCE}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SyntheticTransitionModel(torch.nn.Module):
    """Deterministic mock head used to test rollout plumbing, not model quality."""
    _action_conditioned_transition_head_checkpoint_trained = True
    _environment_observation_transition_checkpoint_trained = True
    _cargo_dynamics_head_checkpoint_trained = True

    def __init__(self, invalid_transition: bool = False):
        super().__init__()
        self.invalid_transition = invalid_transition

    def forward(self, observations, **kwargs):
        batch, device = observations.shape[0], observations.device
        transition = torch.zeros((batch, module.ACTION_TRANSITION_OUTPUT_DIM), device=device)
        if self.invalid_transition:
            transition.fill_(100.0)
        return (
            torch.zeros((batch, 3), device=device),
            torch.zeros((batch, module.NUM_EXECUTIVE_STATES), device=device),
            torch.zeros((batch, 1), device=device),
            torch.ones((batch, 1), device=device),
            torch.ones((batch, 1), device=device),
            torch.zeros((batch, 1), device=device),
            torch.zeros((batch, 6), device=device),
            torch.zeros((batch, 5), device=device),
            transition,
            torch.zeros((batch, module.PHYSICS_UNCERTAINTY_DIM), device=device),
            torch.zeros((batch, module.ENVIRONMENT_TRANSITION_DIM), device=device),
        )


def main() -> None:
    templates = module._learned_mpc_candidate_templates()
    assert len(templates) == 11
    assert templates[0]["candidate_id"] == "NOMINAL"
    assert len({row["candidate_id"] for row in templates}) == len(templates)

    action = module._learned_mpc_make_action_condition(np.zeros(3, dtype=np.float32), 0, 0.0)
    assert action.shape == (module.ACTION_TRANSITION_ACTION_DIM,)
    assert int(np.argmax(action[3:3 + module.NUM_EXECUTIVE_STATES])) == 0
    assert float(action[-1]) == 0.0

    for invalid_action in (
        (np.zeros(3, dtype=np.float32), 6, 0.0),
        (np.asarray([1.1, 0.0, 0.0], dtype=np.float32), 0, 0.0),
    ):
        try:
            module._learned_mpc_make_action_condition(*invalid_action)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid/untrained action was not rejected")

    assert module._learned_mpc_decode_transition(np.zeros(module.ACTION_TRANSITION_OUTPUT_DIM)).shape == (
        module.ACTION_TRANSITION_OUTPUT_DIM,)
    try:
        module._learned_mpc_decode_transition(np.full(module.ACTION_TRANSITION_OUTPUT_DIM, 100.0))
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-envelope transition was not rejected")

    obs = np.zeros(module.OBSERVATION_DIM, dtype=np.float32)
    frame = obs.reshape(module.HISTORY_LEN, module.BASE_OBS_DIM)[-1]
    frame[3:6] = np.asarray([0.0, 0.0, 12.0 / 200.0], dtype=np.float32)
    frame[9:18] = np.eye(3, dtype=np.float32).reshape(-1)
    frame[58] = 0.0
    assert module._learned_mpc_state_feasibility(obs, active_cargo=False)["feasible"]
    low = obs.copy()
    low.reshape(module.HISTORY_LEN, module.BASE_OBS_DIM)[-1, 5] = 0.1 / 200.0
    assert not module._learned_mpc_state_feasibility(low, active_cargo=False)["feasible"]

    action = module._learned_mpc_make_action_condition(np.zeros(3, dtype=np.float32), 0, 0.0)
    candidate = module._learned_mpc_candidate_sequence(
        SyntheticTransitionModel(), torch.device("cpu"), obs, action, templates[0], 3, dt_s=0.016)
    assert candidate["feasible_for_offline_model_envelope"]
    assert candidate["scored_steps"] == 3
    assert candidate["deployment_or_safety_feasible"] is False
    assert candidate["score"] is not None and np.isfinite(candidate["score"])

    bad = module._learned_mpc_candidate_sequence(
        SyntheticTransitionModel(invalid_transition=True), torch.device("cpu"), obs,
        action, templates[0], 3, dt_s=0.016)
    assert not bad["feasible_for_offline_model_envelope"]
    assert bad["scored_steps"] == 0
    assert bad["reason"] and "outside_transition_target_envelope" in bad["reason"]

    state = module._learned_mpc_decode_observation(obs)
    cost = module._learned_mpc_cost_terms(state, state["position_m"], np.zeros(3), np.zeros(3), np.zeros(3))
    assert np.isfinite(cost["total"]) and cost["total"] >= 0.0
    print("[V38.46 FOCUSED TEST PASS] action contract, FSM rejection, transition bounds, state envelope, 3-step synthetic recursive rollout, and fail-closed prediction rejection.")
    print("[LIMIT] Synthetic zero-head tests validate software plumbing only; no trained-checkpoint accuracy or physical safety is established.")


if __name__ == "__main__":
    main()
