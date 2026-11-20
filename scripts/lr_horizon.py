"""Horizon correction for the searched learning rate (instruct / chat).

The LR search (lr_search.py) scores candidate LRs on short probes (S1: tens to ~100 optimizer steps) and the
winner is then used for a run that is 20-60x longer. Picking the LR that minimises loss after a short horizon
is biased towards LRs that are too small for the full run: short-horizon bias (Wu, Ren, Liao, Grosse 2018,
"Understanding Short-Horizon Bias in Stochastic Meta-Optimization", arXiv 1803.02021). Large-LR runs pay a
transient cost early that a long run recovers, so the long-horizon optimum sits closer to the stability edge
than the short-horizon one ("Stepping on the Edge", NeurIPS 2024; Cohen et al. 2021, arXiv 2103.00065).

The correction grows with how much longer the real run is than the probe that chose the LR:
    lr <- lr * clamp((run_steps / probe_steps) ** GAMMA, 1, CAP)
Measured (official evaluator, crown paired test, tools/paired.py): see NOTES 20-21 Nov 2026.
"""
import math

GAMMA = 0.096        # 1.4x at a 33x longer run (calibration: r_3d21b45dq 4090 / a15b846a H100 A/B)
CAP = 1.6


def factor(run_steps, probe_steps) -> float:
    try:
        r = float(run_steps) / float(probe_steps)
    except (TypeError, ValueError, ZeroDivisionError):
        return 1.0
    if not math.isfinite(r) or r <= 1.0:
        return 1.0
    return float(min(CAP, max(1.0, r ** GAMMA)))


def correct(lr: float, run_steps, probe_steps, log=print) -> float:
    f = factor(run_steps, probe_steps)
    if f != 1.0:
        log(f"[sn56][horizon] lr {lr:.2e} x {f:.3f} = {lr * f:.2e} (run ~{float(run_steps):.0f} steps vs probe "
            f"{int(probe_steps)} steps; short-horizon bias correction)")
    return float(lr) * f
