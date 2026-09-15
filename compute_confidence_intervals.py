"""
Compute Wilson score 95% confidence intervals for all key proportions
in the PubTabNet consistency study (n=100 tables, token-enabled run).

Wilson score interval: more reliable than normal approximation at small n,
especially near 0 or 1.
"""
import math

def wilson_ci(k, n, z=1.96):
    """Wilson score interval for k successes in n trials."""
    if n == 0:
        return (0.0, 0.0)
    p_hat = k / n
    denominator = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denominator
    margin = (z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denominator
    lo = max(0.0, centre - margin)
    hi = min(1.0, centre + margin)
    return lo, hi

def one_sided_upper_wilson(k, n, z=1.645):
    """One-sided 95% upper bound on a proportion (for the '0 successes' case)."""
    p_hat = k / n
    denominator = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denominator
    margin = (z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denominator
    return min(1.0, centre + margin)

def fmt(lo, hi):
    return f"[{100*lo:.1f}%–{100*hi:.1f}%]"

n = 100  # total tables measured

print("=" * 65)
print("PubTabNet H1 Pilot: 95% Wilson Score Confidence Intervals")
print(f"n = {n} tables, apoidea/pubtabnet-html, validation split")
print("=" * 65)

# 1. Tables with raw geometric overlaps
k_ov = 57
lo, hi = wilson_ci(k_ov, n)
print(f"\n[1] Tables with ≥1 raw geometric overlap")
print(f"    Point estimate: {k_ov}/{n} = {100*k_ov/n:.1f}%")
print(f"    95% CI (Wilson):  {fmt(lo, hi)}")

# 2. Catastrophic collapse rate
k_cat = 5
lo, hi = wilson_ci(k_cat, n)
print(f"\n[2] Catastrophic collapse (n_overlaps > IQR fence=24)")
print(f"    Point estimate: {k_cat}/{n} = {100*k_cat/n:.1f}%")
print(f"    95% CI (Wilson):  {fmt(lo, hi)}")
print(f"    (Interpretation: between 1 in {round(1/(hi),1)} and 1 in {round(1/(lo),1)} tables if lo>0)")

# 3. OTSL squareness (100%) — trivially 100%, CI shows floor
k_sq = 100
lo, hi = wilson_ci(k_sq, n)
print(f"\n[3] OTSL grid valid (otsl_square = True)")
print(f"    Point estimate: {k_sq}/{n} = 100.0%")
print(f"    95% CI (Wilson):  {fmt(lo, hi)}")

# 4. Bbox sync (100%)
k_sync = 100
lo, hi = wilson_ci(k_sync, n)
print(f"\n[4] Bbox count matches OTSL cell count (bbox_sync = True)")
print(f"    Point estimate: {k_sync}/{n} = 100.0%")
print(f"    95% CI (Wilson):  {fmt(lo, hi)}")

# 5. Near-duplicate IoU > 0.90 in ordinary tables — CONFIRMED: 3/95
print(f"\n[5a] Near-duplicate cells (max pair IoU > 0.90) — PRELIMINARY (n=25 ordinary tables)")
k_nd_prelim = 0
n_nd_prelim = 25
upper_prelim = one_sided_upper_wilson(k_nd_prelim, n_nd_prelim)
print(f"    Observed: {k_nd_prelim}/{n_nd_prelim} = 0.0%")
print(f"    One-sided 95% upper bound: {100*upper_prelim:.1f}%  <- was overstated as clean zero")

print(f"\n[5b] Near-duplicate cells (max pair IoU > 0.90) — CONFIRMED (n=95 ordinary tables)")
k_nd = 3
n_nd = 95
lo, hi = wilson_ci(k_nd, n_nd)
print(f"    Observed: {k_nd}/{n_nd} = {100*k_nd/n_nd:.1f}%")
print(f"    95% CI (Wilson):  {fmt(lo, hi)}")
print(f"    Flagged: pubtabnet_tok_597425 (max_iou=0.946), pubtabnet_tok_689997 (0.933), pubtabnet_tok_722429 (0.968)")
print(f"    Note: ALL 3 had ordinary overlap counts (7, 3, 21 resp.) — near-dup is a subset of ordinary overlaps.")
print(f"    max pair IoU > 0.95: 1/95 = 1.1% (Wilson 95% CI: [0.0%–5.8%])")

# 6. Exact bbox duplicates (b1 == b2) — 0/100
print(f"\n[6] Exact bbox-coordinate duplicates (b1 == b2, both runs)")
k_dup = 0
n_dup = 100
upper2 = one_sided_upper_wilson(k_dup, n_dup)
print(f"    Observed: {k_dup}/{n_dup} = 0.0%")
print(f"    One-sided 95% upper bound: {100*upper2:.1f}%")

print()
print("Note: Wilson score is preferred over normal approximation at edges (p near 0/1).")
print("For the ordinary-overlap mean (3.2), a t-interval on the full distribution")
print("would require the per-table values; that can be computed from the CSV.")
