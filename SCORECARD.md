# 🪞 Neural Mirror — Frontier vs Small Model Scorecard

## Setup

5 models participated in a 3-round roundtable on Modal, examining their own GGUF weights via introspection tools. A frontier model (Claude Sonnet 4) then analyzed the same ground-truth data. All responses were scored by [Jev](https://typesafe.fm) (calibrated AI judge).

## Scores (0–4 scale)

| Model | Params | Accuracy | Insight | Cross-Model | Groundedness |
|-------|--------|----------|---------|-------------|-------------|
| 🏆 **Claude (frontier)** | — | 1.0 | **3.4** | **3.8** | **2.1** |
| Phi4-mini | 3.8B | 0.8 | 1.3 | 1.7 | 1.8 |
| Mistral 7B | 7.2B | 0.2 | 1.5 | 1.4 | 1.0 |
| Llama 3.2 3B | 3.2B | 0.3 | 1.3 | 1.2 | 1.2 |
| Qwen3 1.7B | 2.0B | 0.7 | 1.0 | 1.1 | 0.4 |

## Key Errors by Small Models

- **Mixed up numbers between models**: Llama & Mistral both cited Phi4's norm growth (1.79x) as Qwen3's
- **Attribution confusion**: Phi4 claimed Qwen3's specific values (17.67, 0.0878) were its own
- **Structural error**: Mistral claimed "parameters increase per layer" — wrong, all layers have identical param count
- **Missed counter-trends**: None noticed `attn_k_norm` *decreases* (3.04 → 1.73)
- **Missed curve shape**: None identified the exponential hockey-stick acceleration starting at layer 17
- **Overreach**: All speculated about "training dynamics" and "learning strategy" from static post-training weights

## What the Frontier Model Caught

- ✅ The 201x `attn_norm` growth (vs ~5x overall average norm growth)
- ✅ The hockey-stick acceleration in the last 40% of layers  
- ✅ The `k_norm` decrease counter-trend
- ✅ The distinction between overall norm growth and specific tensor norms
- ✅ The epistemological limit: static weights ≠ training dynamics
- ✅ The 91–96% sparsity as a structural finding worth investigating

## Verdict

**The gap is capability, not access.** All models had identical tools and data. The frontier model's advantage was entirely in *interpretation* — noticing non-obvious patterns, distinguishing what can and cannot be inferred, and maintaining factual discipline about which numbers belong to which model.

The frontier advantage was:
- **3.4x** on analytical depth (3.4 vs 1.0–1.5 avg)
- **2.7x** on cross-model comparison (3.8 vs 1.1–1.7 avg)
- Narrowest gap on raw accuracy (everyone struggles with exact numbers)

Bigger ≠ better among small models: Mistral 7B scored *worst* on accuracy despite being the largest small model. Phi4-mini (3.8B) was the best small model overall.
