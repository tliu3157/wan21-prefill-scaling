# Wan2.1-T2V-1.3B: does cost grow the way the FLOP count says?

A benchmark of one forward pass of the Wan2.1-T2V-1.3B diffusion transformer (DiT) on a Tesla T4, measuring how latency and peak memory scale with the number of video latent tokens `N`, and comparing both against an analytical FLOP and memory model.

**Research question:** how closely do prefill latency and memory scale with prompt length, compared with theoretical FLOP and memory predictions?

For a video diffusion model, "prompt length" is taken to be `N`, the number of video latent tokens the DiT sees (the text prompt is fixed at 512 tokens). "Prefill" is one DiT forward pass, i.e. one denoising step.

## Findings

| | Prediction | Measured |
|---|---|---|
| Latency slope, N = 1,560 to 12,480 | 1.24 | 1.29 |
| Latency slope, N = 12,480 to 56,160 | 1.65 | 2.05 |
| Point where attention FLOPs equal the rest | N ≈ 14,100 | curve bends near there |
| Fused-attention memory | 36.4 kB per token, linear | 49.4 kB per token, linear (slope 0.98) |
| Naive-attention memory | quadratic | slope 1.89, about 2.1x the formula, out of memory at N = 12,480 |
| Same N, different (frames, height, width) | same cost | 0.2% latency spread, identical memory |

The FLOP model gets the shape right up to about 12,000 tokens. Beyond that the measured cost grows faster than the FLOPs: achieved throughput falls from about 20% to 11% of the T4's peak, so a model fitted on short clips underestimates the longest one by nearly 2x. One step of a standard 81-frame 480p clip (N = 32,760) took 28.6 s.

The full write-up is in [`report/main.pdf`](report/main.pdf) (LaTeX source: [`report/main.tex`](report/main.tex)).

## Repository layout

```
notebook/wan21_prefill_scaling.ipynb   the benchmark, with the outputs of the run reported here
data/results.csv                       raw results (see "About the data")
scripts/build_csv.py                   how data/results.csv was assembled from the notebook outputs
scripts/make_figures.py                redraws all four figures from data/results.csv
figures/                               fig1..fig4 as PNG and PDF
report/main.tex, report/main.pdf       the write-up
```

## Setup of the reported run

| | |
|---|---|
| Model | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, `transformer` subfolder (30 layers, d = 1536, 12 heads, FFN 8960, 1.42 B parameters) |
| Model revision | `main` branch at the time of the run. The commit hash was not pinned or logged. See "Recording the model revision" below. |
| Hardware | Google Colab, Tesla T4, 15.6 GB |
| Precision | fp16 (the T4 does not support bf16) |
| Software | PyTorch 2.14.1+cu130, latest `diffusers` / `transformers` / `accelerate` from pip |
| Timing | 2 warm-up passes, then 5 timed passes per size; the median is reported. Timed with `torch.cuda.Event` |
| Memory | `torch.cuda.max_memory_allocated()` minus the memory held by the weights (2.85 GB) |
| Held constant | batch size 1, 512 text tokens, timestep 500, random latents and text embeddings, `torch.inference_mode` |
| Independent variable | N from 104 to 56,160: latent frames {1, 2, 3, 4, 6, 8, 11, 14, 17, 21, 28, 36} at 60x104, plus two smaller single-frame shapes |

Three sweeps are run:

1. **Fused attention** (PyTorch's default scaled-dot-product attention).
2. **Naive attention** (`SDPBackend.MATH`), which materialises the N x N score matrix. Stops at the first out-of-memory error.
3. **Same N, different shape**: (4, 60, 104), (8, 30, 104) and (16, 30, 52) all give N = 6,240.

## About the data

`data/results.csv` has one row per (sweep, backend, run, shape).

| Column | Meaning |
|---|---|
| `sweep` | `main` or `sameN` |
| `backend` | `default` (fused) or `math` (naive) |
| `run` | `A` or `B` |
| `F`, `H`, `W` | latent frames, height, width |
| `N` | video latent tokens, `F * (H/2) * (W/2)` |
| `oom` | True if the size ran out of memory |
| `lat_med_s` | median latency of 5 timed forward passes, seconds |
| `peak_act_gb` | peak activation memory, GB |

Things to know before using it:

- The fused sweep was executed twice in one Colab session. **Run A** is the earlier pass and covers the 11 sizes up to N = 26,520. **Run B** covers all 14 sizes. The three largest sizes (32,760, 43,680 and 56,160 tokens) therefore have a single run.
- The naive-attention sweep and the same-N check have one run each.
- The CSV was transcribed from the outputs saved in the notebook (the results table in section 6 and the sweep logs in section 5), because the session's own `results.csv` was not kept. Values carry the precision shown there: 4 decimal places for fused latency and memory, 0.1 ms for naive latency, 0.01 GB for naive memory. `scripts/build_csv.py` holds the transcription.
- The notebook also computes the standard deviation over the 5 timed passes (`lat_std`), but it was not displayed, so it is not in this file.

## Rerunning the benchmark

1. Open `notebook/wan21_prefill_scaling.ipynb` in [Google Colab](https://colab.research.google.com/) (File > Upload notebook, or File > Open notebook > GitHub).
2. Runtime > Change runtime type > **T4 GPU**. Other GPUs work; L4 and A100 use bf16 and have different peak-FLOP constants, which the notebook sets automatically.
3. Runtime > Run all. No Hugging Face token is needed. The whole notebook takes roughly 35 minutes on a T4, most of it the three largest sizes.
4. Results are written to `wan_prefill_results/results.csv` and `wan_prefill_results/wan_prefill_scaling.png`, and downloaded by the last cell.
5. For additional trials, rerun only the sweep cells in section 5. Results accumulate in the same `RESULTS` list and CSV. Rerunning the harness cell in section 4 clears them.

To shorten a run, trim `FRAMES` in the config cell. Dropping 28 and 36 cuts the fused sweep to about 10 minutes.

### Recording the model revision

The reported run did not record which commit of the model it downloaded. Add this cell after the model loads to print it:

```python
from huggingface_hub import model_info
print(model_info(MODEL_ID).sha)
```

To pin a run to that commit, pass `revision="<sha>"` to `WanTransformer3DModel.from_pretrained`.

## Reproducing the figures

```bash
git clone https://github.com/tliu3157/wan21-prefill-scaling.git
cd wan21-prefill-scaling
pip install numpy pandas matplotlib
python scripts/make_figures.py
```

This reads `data/results.csv` and writes four figures to `figures/`:

| File | Shows |
|---|---|
| `fig1_latency_vs_tokens` | latency (s) vs N for both fused runs and the naive backend, with the FLOP model and the compute lower bound |
| `fig2_memory_vs_tokens` | peak activation memory (GB) vs N for both backends, with both memory formulas |
| `fig3_utilisation_vs_tokens` | achieved FLOP rate as % of the T4's 65 TFLOP/s peak, per run |
| `fig4_run_to_run_variation` | difference between run A and run B as % of their mean |

To plot your own rerun, replace `data/results.csv` with a file in the same column layout.

To rebuild the PDF: `cd report && pdflatex main.tex && pdflatex main.tex`.

## The analytical model

With d = 1536, f = 8960, L = 512 text tokens, 30 layers and 2 FLOPs per multiply-add, per layer:

| Component | FLOPs |
|---|---|
| Self-attention Q, K, V, O projections | 8 N d² |
| Self-attention scores and weighted sum | 4 N² d |
| Cross-attention Q, O projections | 4 N d² |
| Cross-attention scores and weighted sum | 4 N L d |
| Cross-attention K, V projection of text | 4 L d² |
| Feed-forward network | 4 N d f |

Total: F(N) ≈ 2.60e9 N + 1.84e5 N² + 1.45e11 FLOPs. Normalisation layers, the patch embedding and the timestep MLP are ignored.

Memory (inference only, 2 bytes per value): fused ≈ 2 N (f + 6d) bytes; naive adds 2 x heads x N² x 2 bytes.

## Limitations

- Two runs for 11 sizes and one run for the three largest, where the model fits worst.
- One GPU type, one precision, and an unpinned model revision.
- The 65 TFLOP/s T4 peak is a nominal figure and the FLOP count was not checked against a profiler. Slopes do not depend on either; utilisation percentages do.
- Memory is allocated, not reserved, memory.
- Only a bare DiT step with random inputs is timed: no text encoder, VAE or classifier-free guidance.

## Reference

Wan Team, Alibaba Group. *Wan: Open and Advanced Large-Scale Video Generative Models.* arXiv:2503.20314, 2025.
