"""
Anomaly Correlation Coefficient (ACC) vs lead day, for the 3 models.

Anomalies are taken w.r.t. a climatology field (the per-cell seasonal mean of
the chosen truth source). For each model and lead day we pool over all land
cells and all forecast cases:

    ACC = Σ f' o' / sqrt( Σ f'^2 · Σ o'^2 ),   f' = fc - clim,  o' = obs - clim

ACC = 1 perfect anomaly pattern, 0 no correlation. The forecast is the ensemble
mean (deterministic models = their single member).

Usage (truth / climatology source defaults to CHIRPS):
    python acc_lead_curves.py            # CHIRPS
    python acc_lead_curves.py tamsat     # TAMSAT
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

import ea_common as C

TRUTH = (sys.argv[1] if len(sys.argv) > 1 else "chirps").lower()
LEADS = [1, 3, 5, 7]

preds = C.load_predictions()
obs_2d, LAT, LON = C.load_truth(TRUTH)

clim_field = np.nanmean(np.stack(list(obs_2d.values())), axis=0)   # (lat, lon) seasonal mean


def acc(model, lead):
    """Pooled anomaly correlation over land cells and cases."""
    fc_all, ob_all = C.gather_pairs(preds, obs_2d, model, lead)
    fa = fc_all.mean(axis=1) - clim_field[None]                   # forecast anomaly (C, lat, lon)
    oa = ob_all - clim_field[None]                               # obs anomaly
    ok = np.isfinite(fa) & np.isfinite(oa)
    fa, oa = fa[ok], oa[ok]
    denom = np.sqrt(np.sum(fa ** 2) * np.sum(oa ** 2))
    return float(np.sum(fa * oa) / denom) if denom > 0 else np.nan


ACC = {m: [acc(m, ld) for ld in LEADS] for m in C.MODELS}
for m in C.MODELS:
    print(f"{m:12s} ACC  " + "  ".join(f"LD{ld}:{a:+.3f}" for ld, a in zip(LEADS, ACC[m])))

# ── plot: ACC vs lead day ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
for m in C.MODELS:
    ax.plot(LEADS, ACC[m], color=C.COLORS[m], lw=2.4, marker="o", ms=8,
            mec="white", mew=1.0, label=C.LABELS[m])

ax.axhline(0.6, color="#999999", ls="--", lw=1, alpha=0.8)
ax.text(LEADS[-1], 0.6, "  ACC = 0.6\n  (useful-skill guide)", va="center",
        ha="left", fontsize=9, color="#666666")
ax.set_xticks(LEADS)
ax.set_xlabel("Lead day", fontsize=12)
ax.set_ylabel("Anomaly Correlation Coefficient", fontsize=12)
ax.set_ylim(min(0.0, min(min(v) for v in ACC.values()) - 0.05), 1.0)
ax.grid(alpha=0.3)
ax.legend(fontsize=11, loc="best", framealpha=0.95)
ax.set_title(f"Anomaly correlation vs lead day, MAM 2024  (truth = {TRUTH.upper()})",
             fontsize=14, fontweight="bold")
plt.tight_layout()
out = f"mam2024_analysis_outputs/acc_lead_curves_{TRUTH}.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print("saved →", out)
plt.show()
