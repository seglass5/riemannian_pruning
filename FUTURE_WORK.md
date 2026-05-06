# Future Work

Possible directions for extending the Riemannian pruning investigation,
roughly ordered by expected scientific payoff.

---

## 1. Scale within causal attention — GPT-2 Medium / Large

**Status**: ✅ implemented (`--model gpt2-medium` / `--model gpt2-large`)

**Question**: does the Ricci advantage grow with model depth and width?

GPT-2 Medium (355M, 24 layers × 16 heads = 384 heads) and Large (774M,
36 layers × 20 heads = 720 heads) have more heterogeneous attention patterns
than GPT-2 base (117M, 12 layers × 12 heads). Deeper causal models should
produce stronger curvature variation across heads, making the geometric signal
more discriminating.

A scaling curve (117M → 355M → 774M) was run and shows the Ricci advantage
degrades monotonically: +12.3 pp over Random at base → +3.2 pp at Medium →
−12.5 pp at Large. The scaling hypothesis is reversed — the concentration of
head importance at scale makes Random the best pruner for GPT-2 Large.

---

## 2. BERT-base as a bidirectional bridge

**Status**: ✅ implemented (`--model bert`)

**Question**: does the directional inversion hold for full BERT, or is it
specific to DistilBERT's distillation?

BERT-base (110M, 12 layers × 12 heads = 144 heads, same count as GPT-2 base)
uses bidirectional attention but was not distilled. If Ricci_inv also wins for
BERT, the inversion is a property of bidirectional attention per se, not of
the distillation process.

---

## 3. Mechanism: which heads get pruned?

**Status**: ✅ implemented (`experiments/head_heatmap.py`)

**Question**: are the Ricci-pruned (and Ricci_inv-pruned) heads structurally
interpretable?

Layer × head heatmaps comparing the prune sets of forward vs inverted Ricci
across architectures would show whether the curvature signal targets specific
layers or head types (positional, syntactic, copy heads etc.). Cross-referencing
with the known attention-head interpretability literature (Clark et al. 2019,
Voita et al. 2019) would connect the geometric finding to functional roles.

**What was built**: `experiments/head_heatmap.py` — five-panel figure:
(1) |Δκ| score heatmap with prune-set overlay, (2) Magnitude score heatmap,
(3) 4-category comparison map (Ricci-only / Magnitude-only / Both / Neither),
(4) mean score per layer line plot, (5) pruned-count per layer bar chart.
Supports all five model architectures and `--invert-ricci` for bidirectional models.

---

## 4. Additive modulation mode

**Question**: can a single modulation strategy recover a consistent Ricci
direction across both causal and bidirectional attention, removing the
architecture-dependent sign flip?

The current scorer uses multiplicative modulation:
`Ã[i,j] = A[i,j] × |∂L/∂A[i,j]|`.
Additive modulation `Ã[i,j] = A[i,j] + α|∂L/∂A[i,j]|` shifts probability
mass rather than scaling it, which may behave differently in the symmetric
bidirectional case and could recover a consistent high-|Δκ| = important
interpretation for both architectures.

**Implementation**: `RicciPruner(modulation="additive")` already exists as a
config option; run a sweep comparison of `multiplicative` vs `additive` on
both GPT-2 and DistilBERT.

---

## 5. Iterative pruning

**Question**: does Ricci's structural signal hold across multiple prune-retrain
rounds?

One-shot scoring at 50% sparsity is aggressive. Iterative pruning (prune 10%
→ fine-tune briefly → re-score → prune another 10% → …) typically recovers
several accuracy points and would show whether the curvature delta remains
informative as the model adapts to earlier pruning decisions.

**Implementation**: add `--iterative` flag and `n_rounds` parameter to
`run_sweep`. Each round calls `fine_tune` for a small number of steps then
re-scores.

---

## 6. Auto-detect correct pruning direction from variance

**Question**: can the correct Ricci direction be selected without knowing the
architecture type in advance?

The wrong direction always produces higher variance across seeds (±0.164 for
GPT-2 Ricci_inv; ±0.081 for DistilBERT forward Ricci). A two-seed probe run
could select the direction that gives lower variance before committing to the
full sweep, making the method architecture-agnostic in practice.

**Implementation**: add `--auto-direction` flag that runs both forward and
inverted Ricci on 2 seeds, picks the lower-variance direction, then runs the
full multi-seed sweep with the selected direction.

---

## 7. DistilBERT Ricci_inv on CoLA

**Question**: does the directional inversion also recover the Ricci signal
for bidirectional models on a harder task, or does the weak-gradient problem
(model barely above majority class) dominate?

Running DistilBERT on CoLA with `--invert-ricci` would test whether the
failure mode for CoLA (GPT-2, weak task signal) and the failure mode for
DistilBERT SST-2 (wrong direction) are separable or interact.

---

## Summary table

| Direction | Status | Priority |
|-----------|--------|----------|
| GPT-2 Medium/Large scaling | ✅ implemented | — |
| BERT-base bidirectional bridge | ✅ implemented | — |
| Head prune-set visualisation | ✅ implemented | — |
| Additive modulation | not started | medium |
| Iterative pruning | not started | medium |
| Auto-detect direction | not started | medium |
| DistilBERT Ricci_inv on CoLA | not started | low |
