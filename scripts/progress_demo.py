"""demo script for TrainingProgress — simulates a training loop with smooth fake metrics."""

import math
import random
import sys
import time

sys.path.insert(0, "src")

from naime_hybrid.training.progress import TrainingProgress

_NOISE = 0.15


def _drift(val, scale, lo=float("-inf"), hi=float("inf")):
    val += random.uniform(-scale, scale) * _NOISE
    return max(lo, min(hi, val))


def simulate_run(architecture: str, total_steps: int = 500, eval_every: int = 80):
    progress = TrainingProgress(total_steps, architecture=architecture)

    lm = 7.5 if "v5" in architecture else 6.0
    entropy = 1.08
    alpha = 0.35
    grad = 1.2
    slot_cos = -0.05
    slot_div = 0.0003
    state_pred = 0.012
    slot_conf = 0.20
    slot_dlt = 0.080
    slot_w_ent = 1.00
    slot_w_max = 0.60
    slot_gate = 0.40
    mix_a = 0.35
    mix_c = 0.38
    mix_s = 0.27
    best_lm = 99.0

    print("=" * 60)
    print(f"  NAIME Training Progress Demo: {architecture}")
    print(f"  steps={total_steps}  eval_every={eval_every}")
    print("=" * 60)

    for step in range(1, total_steps + 1):
        time.sleep(0.12)

        decay = 1.0 - step / total_steps
        lm = max(1.0, lm - 0.0028 + _drift(0, 0.008, hi=0.02))
        entropy = _drift(entropy, 0.01, lo=0.50)
        alpha = _drift(alpha, 0.005, lo=0.10, hi=0.80)
        grad = _drift(grad, 0.10, lo=0.10)
        slot_cos = min(0.60, max(-0.10, slot_cos + 0.0012 + _drift(0, 0.005)))
        slot_conf = _drift(slot_conf, 0.02, lo=0.0, hi=0.50)
        slot_dlt = _drift(slot_dlt, 0.005, lo=0.010)
        slot_w_ent = _drift(slot_w_ent, 0.02, lo=0.50)
        slot_w_max = _drift(slot_w_max, 0.02, lo=0.20, hi=0.90)
        slot_gate = _drift(slot_gate, 0.02, lo=0.0)
        mix_a = _drift(mix_a, 0.005, lo=0.10)
        mix_c = _drift(mix_c, 0.005, lo=0.10)
        mix_s = 1.0 - mix_a - mix_c
        state_pred = _drift(state_pred, 0.0005, lo=0.001)

        payload = {
            "step": step,
            "lm": lm,
            "ppl": min(9999.0, math.exp(min(20.0, lm))),
            "alpha_downstream_mean": alpha,
            "router_entropy": entropy,
            "grad_norm": grad,
            "lr": 3e-4 * max(0.1, decay),
            "lambda_sparse_effective": 0.012,
            "v5_slot_confidence": slot_conf,
            "v5_slot_cosine": slot_cos,
            "v5_slot_read_entropy": _drift(0.90, 0.02, lo=0.50),
            "v5_state_pred": state_pred,
            "v5_slot_diversity": max(0.0, slot_div),
            "v5_slot_stability": 0.0,
            "v5_slot_delta": slot_dlt,
            "v5_slot_update_gate": slot_gate,
            "v5_slot_write_entropy": slot_w_ent,
            "v5_slot_write_max": slot_w_max,
            "gate_mix_alpha_weight": mix_a,
            "gate_mix_clean_weight": mix_c,
            "gate_mix_state_weight": mix_s,
            "v4_state_confidence": slot_conf,
            "v4_state_delta": slot_dlt,
            "v4_memory_norm": _drift(1.30, 0.05, lo=0.5),
            "v4_memory_read_strength": _drift(0.32, 0.02, lo=0.1),
            "v4_memory_novelty": _drift(0.30, 0.02, lo=0.0),
            "tok_s": _drift(13700, 300, lo=5000),
            "tokens_per_step": 3072,
        }

        progress.render_step(payload)

        if step % eval_every == 0:
            val_lm = lm + random.uniform(-0.15, 0.10)
            best_lm = min(best_lm, val_lm)
            eval_payload = {
                **payload,
                "val_lm_loss": val_lm,
                "val_ppl_val": math.exp(min(20.0, val_lm)),
                "val_alpha_downstream_mean": alpha + random.uniform(-0.01, 0.01),
                "val_router_entropy": entropy + random.uniform(-0.02, 0.02),
                "best_val_lm_loss": best_lm,
                "structural_gap": 0.0,
            }
            progress.render_eval(eval_payload)
            time.sleep(1.2)

    progress.finalize()
    print()


if __name__ == "__main__":
    arch = sys.argv[1] if len(sys.argv) > 1 else "naime_v5_world_state_moe"
    simulate_run(arch)
