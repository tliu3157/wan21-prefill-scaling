"""Reproduce the four figures in the write-up from data/results.csv.

Usage:  python scripts/make_figures.py
Writes figures/fig{1..4}_*.png and .pdf.
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
df = pd.read_csv(os.path.join(ROOT, "data", "results.csv"))
FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)

# Wan2.1-T2V-1.3B DiT config and Tesla T4 constants (as printed by the notebook)
LAYERS, D, FFN, HEADS, TEXT_LEN, BYTES = 30, 1536, 8960, 12, 512, 2
PEAK_TFLOPS, TOTAL_GB, WEIGHTS_GB = 65.0, 15.6, 2.85

def flops_total(N):
    N = np.asarray(N, dtype=float)
    return LAYERS * (N * (12 * D**2 + 4 * D * FFN) + 4 * N**2 * D + 4 * N * TEXT_LEN * D + 4 * TEXT_LEN * D**2)
def act_fused_gb(N): return BYTES * np.asarray(N, float) * (FFN + 6 * D) / 1e9
def act_math_gb(N):  return act_fused_gb(N) + 2 * HEADS * np.asarray(N, float)**2 * BYTES / 1e9

BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.6, "legend.frameon": False,
                     "figure.facecolor": "white", "axes.facecolor": "white", "lines.linewidth": 2, "lines.markersize": 7})

main = df[(df.sweep == "main") & (df.backend == "default")]
A = main[main.run == "A"].sort_values("N"); B = main[main.run == "B"].sort_values("N")
M = df[(df.sweep == "main") & (df.backend == "math") & (~df.oom)].sort_values("N")
Ngrid = np.geomspace(100, 60000, 200)
calib = B.lat_med_s.iloc[-1] / flops_total(B.N.iloc[-1])      # one constant, fitted at the largest N

def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=200)
    plt.close(fig)

# Fig 1: latency
fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.loglog(Ngrid, flops_total(Ngrid) * calib, ":", color=INK, lw=1.5, label="FLOP model (scaled to match at largest N)")
ax.loglog(Ngrid, flops_total(Ngrid) / (PEAK_TFLOPS * 1e12), "--", color=MUTED, lw=1.2, label="Lower bound at 65 TFLOP/s peak")
ax.loglog(B.N, B.lat_med_s, "s-", color=ORANGE, mec="white", mew=1, lw=1.5, label="Fused attention, run B (14 sizes)")
ax.loglog(A.N, A.lat_med_s, "o", mfc="none", mec=BLUE, mew=1.8, ms=10, label="Fused attention, run A (11 sizes)")
ax.loglog(M.N, M.lat_med_s, "^-", color=AQUA, mec="white", mew=1, lw=1.5, label="Naive (math) attention, 1 run")
ax.axvline(14136, color=MUTED, lw=0.8); ax.text(14136 * 1.07, 0.012, "attention = linear FLOPs\n(N = 14,136)", fontsize=8, color=MUTED, va="bottom")
ax.set(xlabel="N (video latent tokens)", ylabel="Latency of one DiT forward pass (s)", ylim=(0.01, 300))
ax.grid(True, which="major"); ax.legend(fontsize=8, loc="upper left")
save(fig, "fig1_latency_vs_tokens")

# Fig 2: memory
fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.loglog(Ngrid, act_fused_gb(Ngrid), ":", color=BLUE, lw=1.5, label="Model, fused: 36.4 kB per token")
ax.loglog(Ngrid[Ngrid < 16000], act_math_gb(Ngrid[Ngrid < 16000]), ":", color=AQUA, lw=1.5, label=r"Model, naive: fused + 48$N^2$ bytes")
ax.loglog(B.N, B.peak_act_gb, "s-", color=BLUE, mec="white", mew=1, lw=1.5, label="Measured, fused attention (identical in runs A and B)")
ax.loglog(M.N, M.peak_act_gb, "^-", color=AQUA, mec="white", mew=1, lw=1.5, label="Measured, naive (math) attention")
lim = TOTAL_GB - WEIGHTS_GB
ax.axhline(lim, color=MUTED, lw=1, ls="--"); ax.text(110, lim * 1.12, f"T4 memory left after weights ({lim:.1f} GB)", fontsize=8, color=MUTED)
ax.plot([12480], [lim], "x", color="#e34948", ms=9, mew=2.5); ax.text(12480 * 1.2, lim * 0.42, "naive: out of memory\nat N = 12,480", fontsize=8, color=INK)
ax.set(xlabel="N (video latent tokens)", ylabel="Peak activation memory (GB)", ylim=(0.0005, 40))
ax.grid(True, which="major"); ax.legend(fontsize=8, loc="lower right")
save(fig, "fig2_memory_vs_tokens")

# Fig 3: utilisation
fig, ax = plt.subplots(figsize=(6.4, 4.0))
for d, c, m, lab in ((A, BLUE, "o", "Run A"), (B, ORANGE, "s", "Run B")):
    ax.semilogx(d.N, flops_total(d.N) / d.lat_med_s / 1e12 / PEAK_TFLOPS * 100, m + "-", color=c, mec="white", mew=1, lw=1.5, label=lab)
ax.set(xlabel="N (video latent tokens)", ylabel="Achieved FLOP rate (% of 65 TFLOP/s peak)", ylim=(0, 30))
ax.grid(True, which="major"); ax.legend(fontsize=9)
save(fig, "fig3_utilisation_vs_tokens")

# Fig 4: run-to-run variation
p = main.pivot(index="N", columns="run", values="lat_med_s").dropna()
diff = (p.A - p.B).abs() / p.mean(axis=1) * 100
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.semilogx(diff.index, diff.values, "o-", color=BLUE, mec="white", mew=1, lw=1.5)
for n, v in diff.items():
    ax.annotate(f"{v:.1f}", (n, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color=INK)
ax.set(xlabel="N (video latent tokens)", ylabel="|run A - run B| / mean latency (%)", ylim=(0, 50))
ax.grid(True, which="major")
save(fig, "fig4_run_to_run_variation")
print("wrote", sorted(os.listdir(FIG)))
