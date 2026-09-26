# 🪞 Neural Mirror

**An LLM self-introspection framework.** Give a language model tools to inspect its own weights — then watch it reason about what it finds.

> The model IS the weights. Now it gets a camera pointed at them.

## What is this?

Standard LLMs use their weights to generate every response, but they can't *see* those weights. It's like a pianist who plays music through billions of microscopic screws inside the piano — but can't examine any individual screw.

Neural Mirror changes that. It:

1. **Parses the GGUF model file** directly — reading tensor metadata, shapes, quantization types
2. **Dequantizes weight samples** to float32 for inspection (supports Q4_0, Q4_K, Q6_K, Q8_0, F16, F32)
3. **Exposes introspection tools** the model can call via Ollama's tool-calling API
4. **Lets the model reason** about its own architecture, weight distributions, and layer-by-layer structure

The result is a recursive loop:

```
┌───────────────┐
│      LLM      │
└───────┬───────┘
        │
  "inspect layer 8"
        │
        ▼
┌───────────────┐
│    runtime    │
└───────┬───────┘
        │
   reads model's
   own GGUF file
        │
        ▼
  weight statistics
        │
        ▼
┌───────────────┐
│      LLM      │
│ reasons about │
│    itself     │
└───────────────┘
```

## Tools

| Tool | Description |
|------|-------------|
| `inspect_self` | Architecture overview: name, params, layers, quantization |
| `inspect_layer(n)` | All tensors in a transformer block with weight statistics |
| `inspect_tensor(name)` | Deep-dive: distribution, raw samples, detailed stats |
| `list_tensors(filter)` | Browse all tensor names |
| `compare_layers(a, b)` | Side-by-side weight statistics between two layers |
| `weight_fingerprint` | Per-layer mean/std across the full network depth |

## Quick Start

```bash
# Requires: Ollama running with a GGUF model
pip install gguf

# Interactive mode — chat with the model while it introspects
python introspect.py

# Auto mode — give it a prompt and watch it explore
python introspect.py --auto "Examine yourself. What do you find?"
```

### Fast Modal consensus

The Modal roundtable uses a two-phase conference protocol by default:

1. The host reads each model's GGUF and creates a compact, citable measurement packet.
2. Models produce constrained evidence briefs, then vote `ACCEPT C1` or `REJECT C1` on one host-derived descriptive motion.

This avoids optional Ollama tool-call incompatibilities and quadratic all-to-all transcripts. Each model sees at most six peer briefs. A requested third round uses one elected chair rather than repeating synthesis across every participant.

```bash
# Diverse three-family smoke conference
MODAL_PROFILE=salus modal run modal_roundtable.py \
  --rounds 2 \
  --models 'granite3.3:8b,command-r7b:7b,olmo2:7b'

# Curated core roster
MODAL_PROFILE=salus modal run modal_roundtable.py --preset core --rounds 2

# Validate model availability without inference
MODAL_PROFILE=salus modal run modal_roundtable.py --preset wide --profile-only
```

Progress is atomically checkpointed after every round in `roundtable_transcript.json`. Consensus is only recorded from explicit, machine-parsed ballots; malformed or missing ballots are abstentions.

### Cross-family mechanism study

`mechanism_study.py` replaces incomparable raw-weight means with a matched functional intervention. It multiplies the output of homologous pre-attention normalization modules by the same dimensionless factors at early, middle, and late relative depths. Every model answers the same deterministic ARC-Challenge sample by option-label likelihood.

```bash
MODAL_PROFILE=salus modal run mechanism_study.py \
  --items 32 \
  --models 'qwen3:4b,phi4-mini,mistral:7b'
```

Primary outcomes are accuracy change, prediction flips, and absolute choice-margin change divided by each model's baseline margin standard deviation. A scale-1 hook is an exact sham control, and confidence intervals bootstrap benchmark examples within model before weighting model families equally.

The initial preregistered three-family run found a shared causal **choice-margin sensitivity** to halving late normalization output, but rejected a universal late-depth mechanism:

| Model | Late ×0.5 standardized margin response (95% CI) | Mean late−early sensitivity |
|---|---:|---:|
| Qwen3 4B | 0.273 [0.239, 0.305] | +0.036 |
| Phi-4 Mini | 0.303 [0.235, 0.382] | -0.610 |
| Mistral 7B | 0.192 [0.159, 0.223] | -0.268 |

The pooled late-minus-early 95% CI was `[-0.339, -0.225]`. Thus, functional effect magnitudes are comparable under this operational definition, but raw GGUF tensor means are not. The result establishes a shared response for these checkpoints and this benchmark—not an identical internal algorithm, training-time cause, or universal transformer mechanism. Full evidence is in `mechanism_results.json`.

### Strict computer-braille admission

`braille_literacy.py` is a zero-tolerance admission test for the project's custom Unicode-braille byte protocol. It tests exact encoding, decoding, and malformed-frame rejection on two unseen seeds. A model must score 28/28; untested and inconclusive models are barred alongside failures.

```bash
MODAL_PROFILE=salus modal run braille_literacy.py \
  --models 'qwen3:4b,phi4-mini,mistral:7b'
```

Initial base-model results admitted no models: Qwen scored 5/28, Phi 3/28, and Mistral 3/28. GPT-OSS 120B is marked inconclusive because its scored run did not complete. A seed-isolated synthetic curriculum then trained a Qwen3 4B LoRA. Curriculum v1 scored 17/28, v2 scored 27/28, and v3 passed the unchanged seeds 17 and 29 at 28/28. Therefore `qwen3:4b+qwen3-4b-braille-literacy-v3` is admitted, while the unadapted `qwen3:4b` remains excluded.

```bash
MODAL_PROFILE=salus modal run --detach braille_curriculum.py --model 'qwen3:4b'
```

LoRA admission is checkpoint-specific: the Qwen adapter cannot be applied to arbitrary architectures or model sizes. Other families require separately trained adapters and independent exams. Fleet training produced 12 operational 28/28 candidates, but several were refined after seeing the original exam and therefore were not treated as independent scientific evidence.

A fresh one-shot certification was preregistered in Git before inference (`braille_certification_manifest.json`, task hash `8ec060936d917d4611238e6a4db29609361dc4171f242491a74442bb3e0e2766`). Ten of eleven frozen candidates passed new seeds 17017 and 29029 at 28/28: Falcon, Gemma, GLM, Granite 3.3, Llama, Mistral, OLMo, Qwen, Solar, and Yi. Phi scored 27/28 and was not certified; it will not be trained or retried on those seeds. The scientific roster is `braille_certified_admission.json`; `braille_admission.json` retains the broader engineering history.

### Paired base-versus-adapter evaluation

`paired_adapter_eval.py` toggles each certified LoRA off and on inside the same frozen Hugging Face checkpoint. Across ten families and 1,920 paired ARC-Challenge, HellaSwag, and MMLU examples, the macro general-accuracy change was **+0.312 percentage points**. Its fixed-family item-bootstrap 95% CI was `[-1.042, 1.615]` points, and its family-and-item bootstrap CI was `[-2.292, 2.708]`; therefore this experiment does not establish a general-capability improvement or decline.

The specialization effect was large: fresh diagnostic protocol accuracy changed from 0/280 to 277/280, and exact structured-JSON output changed from 112/160 to 144/160. All deterministic disabled-adapter sham repeats passed. Exploratory family-level results were heterogeneous—Qwen improved, Mistral declined, and most estimates included zero—so adapters should be evaluated per checkpoint rather than assumed beneficial or harmless. Full paired rows, choice-distribution shifts, bootstrap intervals, and limitations are in `paired_adapter_eval_results.json` and `paired_adapter_eval_summary.json`.

### Strict braiding

`braille_protocol.py` defines a 14-cell versioned braid frame with sender, round, operation, subject, evidence ID, relation, confidence, caveat flags, a numeric value, and CRC-8. `strict_braid.py` rejects invalid checksums, unsupported evidence IDs, duplicate frames, invalid participants, and out-of-order rounds.

An attempted LoRA extension that made Qwen and Llama calculate the new frames scored 0/16 exact for both and degraded their original literacy to 26/28 and 21/28. This established that protocol extensions were not automatically compositional and that exact transport training caused interference. The failed adapters remain separate from the admitted literacy adapters.

The working architecture assigns semantic field selection to models and byte encoding, evidence validation, and CRC calculation to deterministic host code. `evidence_registry.py` combines immutable base-GGUF, LoRA, controlled-intervention, and certification records into a hashed registry with stable uint16 IDs, provenance hashes, and explicit limitations. The current registry contains 99 records and has SHA-256 `443ae8a5f9b2ed6b423bcc8b90b480c3d73eddce125cc3ebec7087c6e0c5a0d0`.

`modal_strict_braid.py` completed a live three-frame Qwen↔Llama exchange bound to registry evidence ID `63485`, the measured Qwen late-normalization ×0.5 standardized margin response. Qwen observed the intervention, Llama challenged overgeneralization, and Qwen accepted replication. Every frame had canonical evidence and a valid checksum; mutated values, unsupported evidence, and checksum corruption fail closed. Models did not calculate integrity bytes.

Tokenizer measurements across eleven families also reject the claim that Unicode braille is inherently token-optimal. For one identical 14-byte frame, mean costs were 37.0 tokens for braille, 57.7 for compact JSON, 21.9 for hex, and 13.9 for base64. Braille is retained as a fixed visual wire representation and validation type—not as the most token-efficient transport. Full results are in `transport_benchmark.json`.

This gate measures competence with Neural Mirror's custom protocol—not general literary-braille knowledge—and it does not make semantically unsupported claims true.

## Example Output

```
you> Examine yourself.

   🔍 inspect_self({})
   ← (1,115 chars, 0.0s)
   🔍 compare_layers({"layer_a": 0, "layer_b": 27})
   ← (4,037 chars, 0.0s)
   🔍 inspect_tensor({"tensor_name": "blk.0.attn_norm.weight"})
   ← (953 chars, 0.0s)
   🔍 inspect_tensor({"tensor_name": "blk.27.attn_norm.weight"})
   ← (1,009 chars, 0.0s)

🪞> The attention norms grow 201x from layer 0 (mean 0.087) to layer 27 
   (mean 17.4). Early layers emphasize precision with tight distributions,
   while later layers prioritize range and diversity...
```

## What the model discovers about itself

When Qwen3 1.7B examines its own weights, it finds:

- **28 transformer blocks** with 2.03B total parameters
- **Attention norm weights grow ~200x** from first to last layer — the model's "volume knob" increases dramatically with depth
- **91%+ of quantized attention weights are near-zero** — extreme sparsity in the attention mechanism
- **Layer 0 is precise and constrained** (std ~0.05) while **Layer 27 is broad and diverse** (std ~4.8)

## Architecture

```
introspect.py
├── GGUFModel          — Zero-dependency GGUF parser
│   ├── _parse()       — Reads metadata + tensor info
│   ├── read_raw_bytes — Raw tensor data access
│   └── dequant_f32    — Q4_0/Q4_K/Q6_K/Q8_0/F16/F32 → float32
├── Tools              — Self-introspection functions
│   ├── inspect_self
│   ├── inspect_layer
│   ├── inspect_tensor
│   ├── list_tensors
│   ├── compare_layers
│   └── weight_fingerprint
└── Chat Loop          — Streaming Ollama integration with tool calling
```

## The idea

> Being something isn't the same as having an internal representation of that thing.

A calculator is made of transistors but doesn't know how many it has. An LLM is made of weights but can't normally inspect them. The weights constitute the model, but aren't an input available to the model.

Neural Mirror makes them available. It's a first step toward models that can:

1. Inspect their own parameters
2. Form hypotheses about their own structure
3. Compare their layers and identify patterns
4. Eventually: modify weights, reload, and evaluate the change

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai) with a GGUF model (tested with `qwen3:1.7b`)
- `gguf` package (optional, for metadata only — the parser is built-in)

## License

MIT
