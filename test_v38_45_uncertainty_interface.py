"""Focused tests for the V38.45 uncertainty-aware offline safety interface.

Run: python test_v38_45_uncertainty_interface.py
Set SURVIVAL_V38_45_RUN_FULL_SELF_TEST=1 to additionally run the source's full suite.
"""
from __future__ import annotations
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

SOURCE = Path(__file__).with_name("Survival_Core_V38_45_Uncertainty_Aware_Offline_Safety.py")
if not SOURCE.is_file():
    raise FileNotFoundError(f"V38.45 source not found: {SOURCE}")
spec = importlib.util.spec_from_file_location("survival_core_v38_45_uncertainty_test_module", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not load V38.45 source")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

# Covariance integrity is diagnostic and must not mutate the provided matrix.
cov = np.diag([0.01] * 3 + [0.04] * 3 + [0.01] * 3 + [0.001] * 6).astype(np.float64)
cov_copy = cov.copy()
assert module._covariance_integrity_report(cov)["valid"]
assert np.array_equal(cov, cov_copy)
bad_asym = cov.copy(); bad_asym[0, 1] = 0.1
assert not module._covariance_integrity_report(bad_asym)["valid"]
bad_psd = cov.copy(); bad_psd[0, 0] = -0.1
assert not module._covariance_integrity_report(bad_psd)["valid"]
bad_nan = cov.copy(); bad_nan[0, 0] = np.nan
assert not module._covariance_integrity_report(bad_nan)["valid"]
assert not module._covariance_integrity_report([['not-a-number']])["valid"]
# Real EKF15 object reports covariance health without mutating/replacing P.
ekf = module.ErrorStateEKF15([0.0, 0.0, 10.0])
for _ in range(20):
    ekf.predict(np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
    ekf.update_gps_position_adaptive(np.array([0.0, 0.0, 10.0], dtype=np.float32))
assert ekf.covariance_health()["valid"]

# Margins must increase monotonically with state covariance and forecast-error allowance.
base = module._offline_uncertainty_clearance_margin(cov, 0.5, 0.05, 0.5, 4.0)
cov_larger = cov.copy(); cov_larger[:6, :6] *= 4.0
higher_cov = module._offline_uncertainty_clearance_margin(cov_larger, 0.5, 0.05, 0.5, 4.0)
higher_error = module._offline_uncertainty_clearance_margin(cov, 0.5, 0.2, 0.5, 4.0)
higher_horizon = module._offline_uncertainty_clearance_margin(cov, 1.0, 0.05, 0.5, 4.0)
assert base["valid"] and base["clearance_feasible"]
assert higher_cov["margin_m"] >= base["margin_m"]
assert higher_horizon["margin_m"] >= base["margin_m"]
assert higher_error["margin_m"] > base["margin_m"]
assert not module._offline_uncertainty_clearance_margin(bad_nan, 0.5, 0.05, 0.5, 4.0)["valid"]
assert not module._offline_uncertainty_clearance_margin(cov, "bad-horizon", 0.05, 0.5, 4.0)["valid"]
capped = module._offline_uncertainty_clearance_margin(cov_larger, 0.5, 0.05, 0.5, 4.0, margin_cap_m=0.5)
assert not capped["valid"] and capped["margin_capped"] and not capped["clearance_feasible"]

# Calibration diagnostics expose coverage *and* sharpness/error distributions.
pred = np.zeros((40, 2), dtype=np.float64)
logvar = np.full((40, 2), np.log(0.25), dtype=np.float64)
target = np.column_stack((np.linspace(-0.1, 0.1, 40), np.zeros(40), np.ones(40)))
stats = module._numpy_uncertainty_calibration_stats(pred, logvar, target, 2)
assert stats["count"] == 40
assert stats["sigma_p95_normalized"] > 0.0
assert "standardized_abs_error_p95" in stats
labels = {"fsm": np.array(["A"] * 20 + ["B"] * 20)}
grouped = module._uncertainty_grouped_calibration_audit(pred, logvar, target, 2, labels, min_group_count=16)
assert grouped["groups"]["fsm"]["A"]["count"] == 20
assert grouped["groups"]["fsm"]["B"]["interpretation_status"].startswith("REPORTABLE")

# MPC uncertainty options are explicit/offline and fail closed if incomplete or capped.
start = np.array([0.0, 0.0, 12.0])
state = module._offline_mpc_initial_state(start, np.zeros(3))
incomplete = module.evaluate_offline_predictive_action(
    state, start, np.array([2.0, 0.0, 12.0]), 4.0, 0.0, horizon_s=0.25, dt_s=0.05,
    state_covariance_15x15=cov)
assert incomplete["status"] == "NO_FEASIBLE_CANDIDATE" and incomplete["feasible_candidates"] == 0
capped_mpc = module.evaluate_offline_predictive_action(
    state, start, np.array([2.0, 0.0, 12.0]), 4.0, 0.0, horizon_s=0.25, dt_s=0.05,
    state_covariance_15x15=cov_larger, empirical_position_error_quantile_m=0.05,
    process_accel_sigma_mps2=0.5, uncertainty_margin_cap_m=0.5)
assert capped_mpc["status"] == "NO_FEASIBLE_CANDIDATE"
assert capped_mpc["live_control_authorized"] is False

print("[V38.45 FOCUSED TEST PASS] covariance integrity, monotone margins, capped/invalid fail-closed behaviour, calibration sharpness/grouping, and offline MPC integration")

if os.environ.get("SURVIVAL_V38_45_RUN_FULL_SELF_TEST", "0") == "1":
    env = os.environ.copy()
    env["SURVIVAL_SELF_TEST"] = "1"
    subprocess.run([sys.executable, str(SOURCE)], check=True, env=env)
