"""Curated, resource-aware registry for the Neural Mirror model swarm.

The goal is family/architecture diversity, not many near-identical fine-tunes.
Every entry is an Ollama model that can expose a GGUF for introspection.
"""

MODEL_SPECS = [
    # Core dense transformer families
    {"family": "Qwen", "model": "qwen3:4b", "tier": "small", "lineage": "Qwen3", "preset": "core"},
    {"family": "Phi", "model": "phi4-mini", "tier": "small", "lineage": "Phi-4", "preset": "core"},
    {"family": "Mistral", "model": "mistral:7b", "tier": "medium", "lineage": "Mistral", "preset": "core"},
    {"family": "Gemma", "model": "gemma3:4b", "tier": "small", "lineage": "Gemma 3", "preset": "core"},
    {"family": "Llama", "model": "llama3.2:3b", "tier": "small", "lineage": "Llama 3", "preset": "core"},
    {"family": "DeepSeek", "model": "deepseek-v2:16b", "tier": "large", "lineage": "DeepSeek V2 MoE", "preset": "core"},
    {"family": "GPT-OSS", "model": "gpt-oss:120b", "tier": "xlarge", "lineage": "OpenAI GPT-OSS MoE", "preset": "core", "role": "research_lead"},

    # Wide family-diversity roster
    {"family": "Granite", "model": "granite3.3:8b", "tier": "medium", "lineage": "IBM Granite", "preset": "wide"},
    {"family": "Command-R", "model": "command-r7b:7b", "tier": "medium", "lineage": "Cohere Command-R", "preset": "wide"},
    {"family": "OLMo", "model": "olmo2:7b", "tier": "medium", "lineage": "AI2 OLMo 2", "preset": "wide"},
    {"family": "Falcon", "model": "falcon3:7b", "tier": "medium", "lineage": "TII Falcon 3", "preset": "wide"},
    {"family": "GLM", "model": "glm4:9b", "tier": "medium", "lineage": "THUDM GLM-4", "preset": "wide"},
    {"family": "Aya", "model": "aya-expanse:8b", "tier": "medium", "lineage": "Cohere Aya", "preset": "wide"},
    {"family": "EXAONE", "model": "exaone3.5:7.8b", "tier": "medium", "lineage": "LG EXAONE", "preset": "wide"},
    {"family": "StableLM", "model": "stablelm2:1.6b", "tier": "small", "lineage": "Stability AI StableLM", "preset": "wide"},
    {"family": "Yi", "model": "yi:6b", "tier": "medium", "lineage": "01.AI Yi", "preset": "wide"},
    {"family": "InternLM", "model": "internlm2:7b", "tier": "medium", "lineage": "InternLM 2", "preset": "wide"},
    {"family": "SmolLM", "model": "smollm2:1.7b", "tier": "small", "lineage": "Hugging Face SmolLM 2", "preset": "wide"},
    {"family": "Solar", "model": "solar:10.7b", "tier": "large", "lineage": "Upstage Solar", "preset": "wide"},
    {"family": "StarCoder", "model": "starcoder2:7b", "tier": "medium", "lineage": "BigCode StarCoder 2", "preset": "wide", "specialty": "code"},
    {"family": "Mixtral", "model": "mixtral:latest", "tier": "large", "lineage": "Mistral Mixtral MoE", "preset": "wide"},
    {"family": "Granite-Hybrid", "model": "granite4:3b", "tier": "small", "lineage": "IBM Granite 4 Hybrid", "preset": "wide"},
    {"family": "Nemotron", "model": "nemotron-3-nano:4b", "tier": "small", "lineage": "NVIDIA Nemotron Hybrid MoE", "preset": "wide"},
]

# Optional variants are useful for within-family studies but excluded from the
# default wide run because they add cost without adding a new family.
VARIANT_SPECS = [
    {"family": "DeepSeek-R1", "model": "deepseek-r1:8b", "tier": "medium", "lineage": "Qwen-distilled reasoning model"},
    {"family": "Qwen-MoE", "model": "qwen3:30b-a3b", "tier": "large", "lineage": "Qwen3 MoE"},
    {"family": "GPT-OSS", "model": "gpt-oss:20b", "tier": "large", "lineage": "OpenAI GPT-OSS MoE"},
]

TIER_ORDER = {"small": 0, "medium": 1, "large": 2, "xlarge": 3}


def specs_for_preset(preset="wide"):
    """Return one model per family for core or wide runs."""
    if preset == "core":
        return [spec.copy() for spec in MODEL_SPECS if spec["preset"] == "core"]
    if preset == "wide":
        return [spec.copy() for spec in MODEL_SPECS]
    if preset == "variants":
        return [spec.copy() for spec in MODEL_SPECS + VARIANT_SPECS]
    raise ValueError(f"Unknown preset {preset!r}; choose core, wide, or variants")


def spec_for_model(model):
    for spec in MODEL_SPECS + VARIANT_SPECS:
        if spec["model"] == model:
            return spec.copy()
    # User-supplied models default to the safe medium tier.
    return {"family": model.split(":", 1)[0], "model": model, "tier": "medium", "lineage": "custom"}


def registry_summary():
    return {
        "families": len({spec["family"] for spec in MODEL_SPECS}),
        "models": len(MODEL_SPECS),
        "tiers": {
            tier: sum(spec["tier"] == tier for spec in MODEL_SPECS)
            for tier in TIER_ORDER
        },
    }
