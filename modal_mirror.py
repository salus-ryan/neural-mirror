"""
Neural Mirror on Modal — Run self-introspection on large models with GPU.

Downloads GGUF models, parses their weights, runs Ollama for inference,
and has each model examine its own structure.

Usage:
  modal run modal_mirror.py  # Run all models
  modal run modal_mirror.py --model "qwen3:8b"  # Specific model
"""

import modal
import json, time, os, sys

app = modal.App("neural-mirror")

# Volume for caching downloaded models
model_cache = modal.Volume.from_name("neural-mirror-models", create_if_missing=True)

# Image with Ollama + our introspection code
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "wget", "procps", "zstd")
    .run_commands(
        "curl -fsSL https://ollama.com/install.sh | sh",
    )
    .pip_install("gguf")
    .add_local_file("introspect.py", "/app/introspect.py")
)

# Models to test — ranging from small to large
MODELS = [
    # name, size hint, needs GPU
    ("qwen3:0.6b", "small", False),
    ("qwen3:1.7b", "small", False),
    ("qwen3:4b", "medium", False),
    ("qwen3:8b", "medium", True),
    ("gemma3:4b", "medium", False),
    ("llama3.2:3b", "medium", False),
    ("phi4-mini", "medium", False),
    ("qwen3:14b", "large", True),
    ("gemma3:12b", "large", True),
    ("llama3.1:8b", "medium", True),
    ("mistral:7b", "medium", True),
    ("deepseek-r1:8b", "medium", True),
]


def start_ollama():
    """Start Ollama server in background."""
    import subprocess
    env = os.environ.copy()
    env["OLLAMA_MODELS"] = "/cache/ollama"
    proc = subprocess.Popen(
        ["ollama", "serve"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Wait for it to be ready
    for _ in range(30):
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
            return proc
        except:
            time.sleep(1)
    raise RuntimeError("Ollama failed to start")


def pull_model(model_name):
    """Pull a model via Ollama."""
    import subprocess
    env = os.environ.copy()
    env["OLLAMA_MODELS"] = "/cache/ollama"
    result = subprocess.run(
        ["ollama", "pull", model_name],
        env=env,
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to pull {model_name}: {result.stderr}")
    return True


def find_gguf_path(model_name):
    """Find the GGUF blob path for an Ollama model."""
    import subprocess
    env = os.environ.copy()
    env["OLLAMA_MODELS"] = "/cache/ollama"
    result = subprocess.run(
        ["ollama", "show", model_name, "--modelfile"],
        env=env,
        capture_output=True, text=True,
    )
    for line in result.stdout.split('\n'):
        if line.startswith('FROM /'):
            path = line[5:].strip()
            if os.path.isfile(path):
                return path
    return None


@app.function(
    image=image,
    volumes={"/cache": model_cache},
    timeout=1200,
    memory=16384,
    cpu=4,
)
def introspect_model_cpu(model_name: str) -> dict:
    """Parse GGUF + run introspection on CPU (for models ≤ 4B)."""
    sys.path.insert(0, "/app")
    from introspect import GGUFModel, load_model, tool_inspect_self, tool_inspect_layer, tool_compare_layers, tool_weight_fingerprint, tool_inspect_tensor, run_auto
    import introspect

    print(f"\n{'='*60}")
    print(f"  🪞 {model_name} — CPU introspection")
    print(f"{'='*60}\n")

    # Start Ollama and pull model
    proc = start_ollama()
    print(f"Ollama started, pulling {model_name}...")
    pull_model(model_name)
    print(f"Model pulled.")

    # Find GGUF
    gguf_path = find_gguf_path(model_name)
    if not gguf_path:
        return {"model": model_name, "error": "Could not find GGUF"}

    print(f"GGUF: {gguf_path}")
    file_size = os.path.getsize(gguf_path)
    print(f"Size: {file_size / 1e9:.2f} GB")

    # Parse
    introspect._model = None
    m = load_model(gguf_path)
    print(f"Parsed: {m.n_tensors} tensors")

    # Static analysis
    self_info = tool_inspect_self()
    print(json.dumps(self_info['identity'], indent=2))
    print(json.dumps(self_info['scale'], indent=2))

    # Layer comparison
    n_layers = self_info['scale']['layer_count']
    # Find actual max layer
    max_layer = 0
    for t in m.tensors:
        if 'blk.' in t:
            parts = t.split('.')
            for i, p in enumerate(parts):
                if p == 'blk' and i + 1 < len(parts):
                    try:
                        max_layer = max(max_layer, int(parts[i + 1]))
                    except:
                        pass

    compare = tool_compare_layers(0, max_layer)
    fingerprint = tool_weight_fingerprint()

    # Norm growth
    curve = fingerprint.get('fingerprint', {})
    if curve:
        first_key = f"layer_0"
        last_key = f"layer_{max_layer}"
        first_abs = curve.get(first_key, {}).get('abs_mean', 0)
        last_abs = curve.get(last_key, {}).get('abs_mean', 0)
        norm_growth = round(last_abs / first_abs, 2) if first_abs else None
    else:
        norm_growth = None

    # Warm up model for inference
    print(f"\nWarming up {model_name} for inference...")
    import urllib.request
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=json.dumps({"model": model_name, "prompt": "hi", "stream": False}).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=120)
        print("Model warm.")
    except Exception as e:
        print(f"Warmup issue: {e}")

    # Run self-introspection
    print(f"\n🧠 Running self-introspection loop...")
    introspect._model = None
    load_model(gguf_path)

    try:
        response = run_auto(
            model=model_name,
            prompt=(
                "You have tools to inspect your own weights. "
                "Call inspect_self (no arguments), then compare_layers with 0 and your last layer. "
                "Then explain: what is unique about YOUR weight structure? "
                "What do the numbers tell you about how you process information? Be specific and concise."
            ),
            stream=False,
        )
    except Exception as e:
        response = f"Inference error: {e}"

    proc.terminate()

    return {
        "model": model_name,
        "identity": self_info.get('identity', {}),
        "scale": self_info.get('scale', {}),
        "architecture": self_info.get('architecture_details', {}),
        "quantization": self_info.get('quantization', {}),
        "norm_growth": norm_growth,
        "fingerprint_sample": {k: v for k, v in list(curve.items())[:5] + list(curve.items())[-5:]},
        "self_reflection": response,
    }


@app.function(
    image=image,
    volumes={"/cache": model_cache},
    timeout=1200,
    gpu="A10G",
    memory=32768,
)
def introspect_model_gpu(model_name: str) -> dict:
    """Parse GGUF + run introspection on GPU (for models > 4B)."""
    # Same logic, just with GPU
    return introspect_model_cpu.local(model_name)


@app.local_entrypoint()
def main(model: str = None, all_models: bool = False, skip_inference: bool = False):
    """Run Neural Mirror on Modal."""

    print("═" * 65)
    print("  🪞 Neural Mirror on Modal — Large Model Self-Introspection")
    print("═" * 65)
    print()

    if model:
        # Single model
        targets = [(model, "medium", True)]
    elif all_models:
        targets = MODELS
    else:
        # Default: interesting subset
        targets = [
            ("qwen3:1.7b", "small", False),
            ("qwen3:8b", "medium", True),
            ("gemma3:4b", "medium", False),
            ("llama3.2:3b", "medium", False),
            ("phi4-mini", "medium", False),
            ("mistral:7b", "medium", True),
        ]

    print(f"Running {len(targets)} models:\n")
    for name, size, gpu in targets:
        print(f"  {'🔥' if gpu else '💻'} {name:<25s} ({size}, {'GPU' if gpu else 'CPU'})")
    print()

    # Launch all in parallel
    futures = []
    for name, size, needs_gpu in targets:
        if needs_gpu:
            futures.append((name, introspect_model_gpu.spawn(name)))
        else:
            futures.append((name, introspect_model_cpu.spawn(name)))

    # Collect results
    results = []
    for name, future in futures:
        print(f"\n⏳ Waiting for {name}...")
        try:
            result = future.get()
            results.append(result)
            print(f"  ✅ {name}: {result.get('scale', {}).get('total_parameters_human', '?')}")
            if result.get('self_reflection'):
                preview = result['self_reflection'][:300].replace('\n', ' ')
                print(f"  💬 {preview}{'...' if len(result.get('self_reflection',''))>300 else ''}")
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            results.append({"model": name, "error": str(e)})

    # Summary
    print(f"\n{'═' * 65}")
    print(f"  📋 Cross-Model Summary")
    print(f"{'═' * 65}\n")

    print(f"  {'Model':<25s} {'Params':>8s} {'Layers':>7s} {'Norm Δ':>8s} {'Quant':>20s}")
    print(f"  {'─'*25} {'─'*8} {'─'*7} {'─'*8} {'─'*20}")

    for r in sorted(results, key=lambda x: x.get('scale', {}).get('total_parameters', 0)):
        if 'error' in r and 'identity' not in r:
            print(f"  {r['model']:<25s} {'ERROR':>8s}")
            continue
        name = r.get('identity', {}).get('name', r.get('model', '?'))
        params = r.get('scale', {}).get('total_parameters_human', '?')
        layers = r.get('scale', {}).get('layer_count', '?')
        ng = r.get('norm_growth', '?')
        quant = list(r.get('quantization', {}).keys())[:2]
        quant_str = ', '.join(quant) if quant else '?'
        print(f"  {name:<25s} {str(params):>8s} {str(layers):>7s} {str(ng):>8s} {quant_str:>20s}")

    # Save results
    output_path = os.path.expanduser("~/neural-mirror/results.json")
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to {output_path}")
    except:
        print("\n  (results printed above)")

    # Print each model's self-reflection
    print(f"\n{'═' * 65}")
    print(f"  🪞 What Each Model Said About Itself")
    print(f"{'═' * 65}")

    for r in results:
        name = r.get('identity', {}).get('name', r.get('model', '?'))
        reflection = r.get('self_reflection', '')
        if reflection and not reflection.startswith('Error'):
            print(f"\n  {'─' * 60}")
            print(f"  {name}")
            print(f"  {'─' * 60}")
            # Print with wrapping
            for line in reflection.split('\n'):
                if line.strip():
                    print(f"  {line}")
            print()
