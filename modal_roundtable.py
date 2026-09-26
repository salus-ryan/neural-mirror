"""
Neural Mirror — Roundtable on Modal 🪞🔄

Co-learning between models, running on cloud GPUs.
Each round, models see what others found and build on it.

Usage:
  modal run modal_roundtable.py
  modal run modal_roundtable.py --rounds 4
"""

import modal
import json, time, os, sys

from swarm_registry import spec_for_model, specs_for_preset, registry_summary

app = modal.App("neural-mirror-roundtable")

model_cache = modal.Volume.from_name("neural-mirror-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "zstd", "procps")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    .pip_install("gguf")
    .add_local_file("introspect.py", "/app/introspect.py")
    # modal_roundtable.py is imported from /root in Modal containers, so this
    # top-level dependency must live on Python's import path as well.
    .add_local_file("swarm_registry.py", "/root/swarm_registry.py")
)


def start_ollama(tier="small"):
    import subprocess
    env = os.environ.copy()
    env["OLLAMA_MODELS"] = "/cache/ollama"
    if tier != "small":
        env["OLLAMA_FLASH_ATTENTION"] = "1"
        env["OLLAMA_KV_CACHE_TYPE"] = "q8_0"
        env["OLLAMA_KEEP_ALIVE"] = "30m"
    # Never pipe an unread Ollama log stream: enough output can fill the pipe
    # and freeze an otherwise healthy inference worker.
    proc = subprocess.Popen(
        ["ollama", "serve"], env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    import urllib.request
    for _ in range(90):
        if proc.poll() is not None:
            raise RuntimeError(f"Ollama exited during startup ({proc.returncode})")
        try:
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
            return proc
        except Exception:
            time.sleep(1)
    proc.terminate()
    raise RuntimeError("Ollama failed to start within 90 seconds")


def pull_model(name):
    import subprocess
    env = os.environ.copy()
    env["OLLAMA_MODELS"] = "/cache/ollama"
    tier = spec_for_model(name).get("tier", "medium")
    timeout = 3600 if tier in ("large", "xlarge") else 1200
    r = subprocess.run(["ollama", "pull", name], env=env,
                      capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"Pull failed: {r.stderr}")


def find_gguf(name):
    import subprocess
    env = os.environ.copy()
    env["OLLAMA_MODELS"] = "/cache/ollama"
    r = subprocess.run(["ollama", "show", name, "--modelfile"],
                      env=env, capture_output=True, text=True)
    for line in r.stdout.split('\n'):
        if line.startswith('FROM /'):
            path = line[5:].strip()
            if os.path.isfile(path):
                return path
    return None


def profile_gguf(path):
    sys.path.insert(0, "/app")
    from introspect import GGUFModel
    m = GGUFModel(path)
    meta = m.metadata
    arch = meta.get('general.architecture', '?')
    name = meta.get('general.name', '?')
    params = sum(t['n_elements'] for t in m.tensors.values())
    layers = set()
    for t in m.tensors:
        if 'blk.' in t:
            ps = t.split('.')
            for i, p in enumerate(ps):
                if p == 'blk' and i+1 < len(ps):
                    try: layers.add(int(ps[i+1]))
                    except: pass
    # Quick norm growth
    first_norms, last_norms = [], []
    max_layer = max(layers) if layers else 0
    for tname, tinfo in m.tensors.items():
        if 'blk.0.' in tname and 'norm' in tname:
            vals = m.dequant_f32_sample(tname, max_elements=128)
            if vals: first_norms.append(sum(vals)/len(vals))
        if f'blk.{max_layer}.' in tname and 'norm' in tname:
            vals = m.dequant_f32_sample(tname, max_elements=128)
            if vals: last_norms.append(sum(vals)/len(vals))
    fn = sum(first_norms)/len(first_norms) if first_norms else 0
    ln = sum(last_norms)/len(last_norms) if last_norms else 0
    ng = round(ln/fn, 2) if fn else None

    return {
        'model_name': name, 'arch': arch,
        'params': f"{params/1e9:.2f}B",
        'n_layers': len(layers), 'max_layer': max_layer,
        'norm_growth': ng,
        'first_norm': round(fn, 4), 'last_norm': round(ln, 4),
    }


def _compact_evidence_packet(local_model, call_tool):
    """Build a bounded, citation-friendly packet from host-executed tools."""
    self_info = call_tool("inspect_self", {})
    layer_numbers = []
    for tensor_name in local_model.tensors:
        if "blk." not in tensor_name:
            continue
        try:
            layer_numbers.append(int(tensor_name.split("blk.", 1)[1].split(".", 1)[0]))
        except (ValueError, IndexError):
            pass
    last_layer = max(layer_numbers) if layer_numbers else 0
    comparison = call_tool("compare_layers", {"layer_a": 0, "layer_b": last_layer})

    # Keep only the largest measured first-to-last changes. Sending every
    # tensor made context grow quickly and slowed agreement without adding a
    # stronger decision signal.
    deltas = comparison.get("deltas", {})
    strongest = sorted(
        deltas.items(),
        key=lambda item: max(abs(float(v)) for v in item[1].values()),
        reverse=True,
    )[:6]
    first = comparison.get("stats", {}).get("layer_0", {})
    last = comparison.get("stats", {}).get(f"layer_{last_layer}", {})
    evidence = {}
    for index, (tensor, delta) in enumerate(strongest, 1):
        evidence[f"DELTA.{index}"] = {
            "tensor": tensor,
            "layer_0": first.get(tensor),
            f"layer_{last_layer}": last.get(tensor),
            "delta": delta,
        }

    return {
        "source": "host-executed GGUF inspection",
        "method": "512-value dequantized samples per tensor; first vs last block",
        "warning": "preliminary quantized samples support descriptive, not causal, claims",
        "identity": self_info.get("identity", {}),
        "scale": self_info.get("scale", {}),
        "quantization": self_info.get("quantization", {}),
        "first_layer": 0,
        "last_layer": last_layer,
        "evidence": evidence,
    }


def _ground_model_response(content, packet):
    """Replace free-form MEASURED text with one canonical host measurement."""
    evidence = packet.get("evidence", {})
    top = evidence.get("DELTA.1")
    if not top:
        return content

    delta = top.get("delta") or {}
    metric_delta, delta_value = max(
        delta.items(), key=lambda item: abs(float(item[1]))
    )
    metric = metric_delta.removesuffix("_delta")
    first_key = f"layer_{packet.get('first_layer', 0)}"
    last_key = f"layer_{packet.get('last_layer', 0)}"
    first_value = (top.get(first_key) or {}).get(metric)
    last_value = (top.get(last_key) or {}).get(metric)
    canonical = (
        f"MEASURED [DELTA.1]: {top.get('tensor')} {metric} changed from "
        f"{first_value} at block {packet.get('first_layer', 0)} to {last_value} "
        f"at block {packet.get('last_layer', 0)} (delta {delta_value}; "
        "512 sampled dequantized values per tensor)."
    )

    # A model may still provide useful labeled interpretation, but its own
    # MEASURED line is not allowed to overwrite the canonical observation.
    lines = [
        line for line in content.splitlines()
        if not line.strip().upper().startswith("MEASURED:")
        and not line.strip().upper().startswith("MEASURED [")
    ]
    recognized_labels = (
        "HYPOTHESIS:", "CAVEAT:", "PROPOSAL:",
        "VOTE:", "REASON:", "DISSENT:", "NEXT_TEST:",
        "CONSENSUS:", "LIMITATION:", "DECISION:",
    )
    labeled = [
        line.strip() for line in lines
        if line.strip().upper().startswith(recognized_labels)
    ]
    # Thinking checkpoints may narrate their work before the requested answer.
    # Once labeled fields exist, retain only those fields.
    return "\n".join([canonical, *(labeled or lines)]).strip()


def _conference_schema(user_prompt):
    """Return the constrained response schema for the current conference phase."""
    if "VOTE: ACCEPT C1" in user_prompt:
        properties = {
            "vote": {"type": "string", "enum": ["ACCEPT C1", "REJECT C1"]},
            "reason": {"type": "string"},
            "dissent": {"type": "string"},
            "next_test": {"type": "string"},
        }
    elif "Act as conference chair" in user_prompt:
        properties = {
            "consensus": {"type": "string"},
            "dissent": {"type": "string"},
            "limitation": {"type": "string"},
            "decision": {"type": "string"},
        }
    else:
        properties = {
            "hypothesis": {"type": "string"},
            "caveat": {"type": "string"},
            "proposal": {"type": "string"},
        }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _render_conference_json(content):
    """Turn constrained JSON into compact human-readable labeled fields."""
    import re

    try:
        data = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        if not str(content).lstrip().startswith("{"):
            return content
        # Never let a truncated/degenerate JSON generation flood peer context.
        # Recover only complete string fields; otherwise expose a short marker.
        data = {}
        for key in (
            "hypothesis", "caveat", "proposal", "vote", "reason",
            "dissent", "next_test", "consensus", "limitation", "decision",
        ):
            match = re.search(
                rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', str(content)
            )
            if match:
                try:
                    data[key] = json.loads(f'"{match.group(1)}"')
                except json.JSONDecodeError:
                    pass
        if not data:
            return "FORMAT_ERROR: invalid structured response"
    labels = {
        "hypothesis": "HYPOTHESIS",
        "caveat": "CAVEAT",
        "proposal": "PROPOSAL",
        "vote": "VOTE",
        "reason": "REASON",
        "dissent": "DISSENT",
        "next_test": "NEXT_TEST",
        "consensus": "CONSENSUS",
        "limitation": "LIMITATION",
        "decision": "DECISION",
    }
    return "\n".join(
        f"{labels[key]}: {value}"
        for key, value in data.items()
        if key in labels and str(value).strip()
    )


def run_model_turn(model_name, gguf_path, system_prompt, user_prompt, host="http://localhost:11434"):
    """One bounded turn using host-generated evidence, with raw fallback.

    Host-side inspection is both faster and more portable than asking every
    model family to negotiate Ollama's optional tool-call wire format. The LLM
    still reasons about its own GGUF; the host merely performs the reads.
    """
    import re
    import urllib.error
    import urllib.request

    sys.path.insert(0, "/app")
    from introspect import GGUFModel, call_tool
    import introspect

    local_model = GGUFModel(gguf_path)
    introspect._model = local_model
    packet = _compact_evidence_packet(local_model, call_tool)
    packet_json = json.dumps(packet, separators=(",", ":"), default=str)
    evidence_user = (
        f"{user_prompt}\n\nSELF EVIDENCE (cite IDs exactly):\n{packet_json}\n\n"
        "Use only supplied measurements. Keep hypotheses explicitly labeled."
    )
    # Qwen3's template-level switch is more reliable than the generic Ollama
    # think flag for forcing a short direct answer on Thinking checkpoints.
    if model_name.split(":", 1)[0] == "qwen3":
        evidence_user = "/no_think\n" + evidence_user
    tool_log = ["host_evidence_packet"]

    def clean(text):
        text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
        text = text.replace("<think>", "").replace("</think>", "")
        # Some raw/chat templates continue by echoing the supplied packet.
        # Keep the answer and discard any copied prompt payload.
        for marker in (
            "\nSELF EVIDENCE:",
            "\nHOST OBSERVATION TABLE:",
            "\nPEER CLAIM BOARD:",
            "\nMOTION:",
        ):
            text = text.split(marker, 1)[0]
        return text.strip()

    def request_json(endpoint, payload, timeout=600):
        req = urllib.request.Request(
            f"{host}{endpoint}", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:1200]
            raise RuntimeError(f"Ollama HTTP {exc.code}: {body}") from exc

    response_schema = _conference_schema(user_prompt)
    prediction_budget = 500 if model_name.split(":", 1)[0] == "qwen3" else 220
    options = {"num_ctx": 4096, "num_predict": prediction_budget, "temperature": 0.0}
    try:
        result = request_json("/api/chat", {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": evidence_user},
            ],
            "stream": False,
            "think": False,
            "format": response_schema,
            "options": options,
        })
        message = result.get("message", {})
        content = _render_conference_json(clean(message.get("content", "")))
        hidden_thinking = clean(message.get("thinking", ""))
        if content:
            tool_log.append("plain_chat")
            return _ground_model_response(content, packet), tool_log
        if hidden_thinking:
            tool_log.append(f"plain_chat_only_thinking({len(hidden_thinking)} chars)")
        else:
            tool_log.append("plain_chat_empty")
    except Exception as chat_exc:
        tool_log.append(f"plain_chat_fallback({str(chat_exc)[:160]})")

    # Base/code models and models lacking a chat template still participate.
    result = request_json("/api/generate", {
        "model": model_name,
        "prompt": f"{system_prompt}\n\n{evidence_user}",
        "stream": False,
        "think": False,
        "format": response_schema,
        "options": options,
    })
    content = _render_conference_json(clean(result.get("response", "")))
    hidden_thinking = clean(result.get("thinking", ""))
    tool_log.append("raw_generate_fallback")
    if not content:
        detail = f"; hidden thinking: {len(hidden_thinking)} chars" if hidden_thinking else ""
        raise RuntimeError(f"Ollama returned no content from chat or generate{detail}")
    return _ground_model_response(content, packet), tool_log


def _candidate_consensus(prior_findings):
    """Derive a narrow C1 motion from canonical Round-1 measurements."""
    import re

    pattern = re.compile(
        r"MEASURED \[DELTA\.1\]: (\S+) (\w+) changed from "
        r"([-+0-9.eE]+) at block \d+ to ([-+0-9.eE]+) at block \d+ "
        r"\(delta ([-+0-9.eE]+);"
    )
    observations = []
    for model, finding in prior_findings.items():
        match = pattern.search(str(finding))
        if match:
            tensor, metric, first, last, delta = match.groups()
            observations.append({
                "model": model,
                "tensor": tensor,
                "metric": metric,
                "first": float(first),
                "last": float(last),
                "delta": float(delta),
            })

    if len(observations) < 2:
        return (
            "C1: The available canonical measurements are insufficient for a "
            "cross-model directional claim."
        ), observations

    positive = sum(item["delta"] > 0 for item in observations)
    negative = sum(item["delta"] < 0 for item in observations)
    if positive == len(observations):
        direction = "positive"
    elif negative == len(observations):
        direction = "negative"
    else:
        return (
            f"C1: DELTA.1 direction is heterogeneous across {len(observations)} "
            "participants, so no shared first-to-last direction is established."
        ), observations

    return (
        f"C1: Each of {len(observations)}/{len(observations)} independent canonical "
        f"briefs has a {direction} sign for its own host-selected DELTA.1 "
        "first-to-last-block statistic. C1 asserts only those independent signs; "
        "it asserts no uniform magnitude, shared tensor, common mechanism, trend, "
        "or cross-model relationship. Selection bias, architecture, and "
        "quantization prevent direct comparison."
    ), observations


def _tally_consensus(candidate, responses, eligible_models=None):
    """Count C1 ballots from models with canonical evidence."""
    import re

    eligible = set(eligible_models or responses)
    votes = {}
    for model, response in responses.items():
        if model not in eligible:
            continue
        match = re.search(
            r"^VOTE:\s*(ACCEPT|REJECT)\s+C1\b",
            str(response),
            flags=re.IGNORECASE | re.MULTILINE,
        )
        votes[model] = match.group(1).upper() if match else "ABSTAIN"
    accepts = sum(vote == "ACCEPT" for vote in votes.values())
    rejects = sum(vote == "REJECT" for vote in votes.values())
    abstains = sum(vote == "ABSTAIN" for vote in votes.values())
    if accepts >= 2 and rejects == 0 and abstains == 0:
        status = "unanimous"
    elif accepts > rejects and accepts >= 2:
        status = "majority"
    else:
        status = "no_consensus"
    return {
        "motion": candidate,
        "status": status,
        "accept": accepts,
        "reject": rejects,
        "abstain": abstains,
        "votes": votes,
        "excluded": sorted(set(responses) - eligible),
    }


def _run_round_for_model_impl(
    model_name: str,
    round_num: int,
    profile: dict,
    roster: str,
    prior_findings: dict,
) -> dict:
    """Run one model's turn. Decorated wrappers select its compute tier."""

    spec = spec_for_model(model_name)
    tier = spec.get("tier", "medium")
    is_lead = spec.get("role") == "research_lead"
    proc = start_ollama(tier=tier)

    # Pull model
    pull_model(model_name)
    gguf_path = find_gguf(model_name)
    if not gguf_path:
        return {"model": model_name, "error": "No GGUF found"}

    # Warm up
    import urllib.request
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=json.dumps({
                "model": model_name, "prompt": "Reply OK.", "stream": False,
                "options": {"num_predict": 2, "temperature": 0.0},
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=600 if tier in ("large", "xlarge") else 240)
    except:
        pass

    s = profile
    role_instruction = (
        "You are the swarm's senior research lead. Separate measurements from interpretation, "
        "challenge unsupported causal claims, and synthesize the strongest peer evidence."
        if is_lead else
        "You are a peer researcher. Report measurements faithfully and avoid causal claims the data cannot support."
    )
    base_system = f"""You are {s['model_name']} ({s['params']}, {s['arch']} architecture, {s['n_layers']} layers).
{role_instruction}

You have introspection tools to examine your own weights:
- inspect_self() — no arguments
- inspect_layer(layer_num) — 0 to {s['max_layer']}
- compare_layers(layer_a, layer_b)
- weight_fingerprint() — no arguments
- inspect_tensor(tensor_name)
- list_tensors(filter_str)

{roster}

RULES:
- Host-executed introspection evidence is authoritative for this turn
- Cite supplied evidence IDs exactly; do not invent evidence or tensor names
- Layer depth is NOT training time or a training stage
- Never infer learning progress, focus, refinement, or performance from weights alone
- Cross-architecture magnitudes are not directly comparable
- No <think> tags — respond directly
- Stay under 180 words
- Separate measurements, hypotheses, and caveats"""

    if round_num == 1:
        prompt = (
            "Create a compact evidence brief for the conference. Return exactly four labeled lines:\n"
            "MEASURED: strongest self finding with one or more evidence IDs and numbers\n"
            "HYPOTHESIS: one cautious interpretation\n"
            "CAVEAT: the most important limitation\n"
            "PROPOSAL: one claim peers should test"
        )
    elif round_num == 2:
        # Bounded, deterministic peer neighborhoods avoid O(models²) prompt
        # growth while ensuring each family considers diverse outside claims.
        own = str(prior_findings.get(model_name, ""))[:300]
        peers = [
            (name, str(finding)) for name, finding in prior_findings.items()
            if name != model_name and not str(finding).startswith("Error:")
        ]
        peers.sort(key=lambda item: sum(ord(c) for c in f"{model_name}|{item[0]}"))
        peer_board = "\n".join(
            f"CLAIM {index} [{name}]: {finding[:320]}"
            for index, (name, finding) in enumerate(peers[:6], 1)
        )
        candidate, observations = _candidate_consensus(prior_findings)
        observation_table = json.dumps(observations, separators=(",", ":"))
        prompt = (
            f"YOUR BRIEF:\n{own}\n\nPEER CLAIM BOARD:\n{peer_board}\n\n"
            f"HOST OBSERVATION TABLE:\n{observation_table}\n\nMOTION:\n{candidate}\n\n"
            "Evaluate only whether C1 accurately describes the observation table; do not vote on a causal story. "
            "Return exactly four labeled lines:\n"
            "VOTE: ACCEPT C1 or REJECT C1\n"
            "REASON: one sentence grounded in the table\n"
            "DISSENT: one material limitation beyond those already in C1, or NONE\n"
            "NEXT_TEST: one controlled experiment that would add causal evidence"
        )
    else:
        briefs = []
        for r, findings in sorted(prior_findings.items(), key=lambda item: str(item[0])):
            if not isinstance(findings, dict):
                continue
            for name, finding in findings.items():
                if not str(finding).startswith("Error:"):
                    briefs.append(f"R{r} [{name}]: {str(finding)[:260]}")
        prompt = (
            "CONFERENCE RECORD:\n" + "\n".join(briefs[:12]) + "\n\n"
            "Act as conference chair. Return exactly four labeled lines:\n"
            "CONSENSUS: strongest claim supported across participants\n"
            "DISSENT: material unresolved disagreement, or NONE\n"
            "LIMITATION: why the consensus remains preliminary\n"
            "DECISION: the single next controlled experiment"
        )

    response, tools_used = run_model_turn(model_name, gguf_path, base_system, prompt)

    proc.terminate()

    return {
        "model": model_name,
        "model_name": s['model_name'],
        "round": round_num,
        "response": response,
        "tools_used": tools_used,
        "role": "research_lead" if is_lead else "peer",
        "family": spec.get("family"),
        "tier": tier,
    }


@app.function(
    image=image,
    volumes={"/cache": model_cache},
    timeout=600,
    memory=16384,
    cpu=4,
)
def run_round_for_model(
    model_name: str,
    round_num: int,
    profile: dict,
    roster: str,
    prior_findings: dict,
) -> dict:
    return _run_round_for_model_impl(
        model_name, round_num, profile, roster, prior_findings
    )


@app.function(
    image=image,
    volumes={"/cache": model_cache},
    gpu="L4",
    timeout=1800,
    memory=24576,
    cpu=4,
)
def run_round_for_medium_model(
    model_name: str,
    round_num: int,
    profile: dict,
    roster: str,
    prior_findings: dict,
) -> dict:
    return _run_round_for_model_impl(
        model_name, round_num, profile, roster, prior_findings
    )


@app.function(
    image=image,
    volumes={"/cache": model_cache},
    gpu="A100",
    timeout=2400,
    memory=32768,
    cpu=8,
)
def run_round_for_large_model(
    model_name: str,
    round_num: int,
    profile: dict,
    roster: str,
    prior_findings: dict,
) -> dict:
    return _run_round_for_model_impl(
        model_name, round_num, profile, roster, prior_findings
    )


@app.function(
    image=image,
    volumes={"/cache": model_cache},
    gpu="H100",
    timeout=3600,
    memory=32768,
    cpu=8,
)
def run_round_for_xlarge_model(
    model_name: str,
    round_num: int,
    profile: dict,
    roster: str,
    prior_findings: dict,
) -> dict:
    return _run_round_for_model_impl(
        model_name, round_num, profile, roster, prior_findings
    )


@app.function(
    image=image,
    volumes={"/cache": model_cache},
    timeout=1800,
    memory=8192,
)
def profile_model_remote(model_name: str) -> dict:
    """Pull and profile a model without loading it for generation."""
    spec = spec_for_model(model_name)
    proc = start_ollama(tier=spec.get("tier", "medium"))
    pull_model(model_name)
    path = find_gguf(model_name)
    if not path:
        proc.terminate()
        return {"model": model_name, "error": "No GGUF"}
    p = profile_gguf(path)
    if p.get('model_name') in (None, '', '?'):
        p['model_name'] = f"{spec.get('family', model_name)} ({model_name})"
    p.update({
        'ollama_name': model_name,
        'family': spec.get('family'),
        'lineage': spec.get('lineage'),
        'tier': spec.get('tier'),
        'role': spec.get('role', 'peer'),
    })
    proc.terminate()
    return p


@app.function(
    image=image,
    volumes={"/cache": model_cache},
    gpu="H100",
    timeout=3600,
    memory=32768,
    cpu=8,
)
def profile_large_model_remote(model_name: str) -> dict:
    """Pull/profile an xlarge model on the same worker class used for inference."""
    spec = spec_for_model(model_name)
    proc = start_ollama(tier="xlarge")
    try:
        pull_model(model_name)
        path = find_gguf(model_name)
        if not path:
            return {"model": model_name, "error": "No GGUF"}
        p = profile_gguf(path)
        if p.get('model_name') in (None, '', '?'):
            p['model_name'] = f"{spec.get('family', model_name)} ({model_name})"
        p.update({
            'ollama_name': model_name,
            'family': spec.get('family'),
            'lineage': spec.get('lineage'),
            'tier': spec.get('tier'),
            'role': spec.get('role', 'peer'),
        })
        return p
    finally:
        proc.terminate()


def _write_transcript(path, payload):
    """Atomically checkpoint progress so interrupted runs retain all rounds."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


@app.local_entrypoint()
def main(
    rounds: int = 2,
    models: str = None,
    preset: str = "wide",
    max_models: int = 0,
    profile_only: bool = False,
):
    """Run a core/wide family roster, or a comma-separated custom roster."""

    specs = (
        [spec_for_model(name.strip()) for name in models.split(',') if name.strip()]
        if models else specs_for_preset(preset)
    )
    if max_models > 0:
        specs = specs[:max_models]
    MODELS = [spec["model"] for spec in specs]
    spec_map = {spec["model"]: spec for spec in specs}
    run_label = "custom" if models else preset

    summary = registry_summary()
    print("━" * 72)
    print("  🪞 Neural Mirror — Model-Family Swarm on Modal")
    print(f"  Selection: {run_label} | selected: {len(MODELS)} models / {summary['families']} registered families")
    print("━" * 72)
    for spec in specs:
        lead = " [research lead]" if spec.get("role") == "research_lead" else ""
        print(f"  {spec['family']:<16s} {spec['model']:<24s} {spec['tier']:<7s}{lead}")
    print()

    # Phase 0: Pull and profile all GGUFs in parallel. Xlarge models use the
    # H100 path; profiling all other tiers does not need a loaded GPU model.
    print("📋 Profiling models...")
    profile_futures = [
        (profile_large_model_remote if spec_map[m].get("tier") == "xlarge" else profile_model_remote).spawn(m)
        for m in MODELS
    ]
    profiles = {}
    for m, fut in zip(MODELS, profile_futures):
        try:
            p = fut.get()
            if 'error' not in p:
                profiles[m] = p
                ng = p.get('norm_growth', '?')
                print(f"  ✅ {p['model_name']:<25s} {p['params']:>6s} {p['arch']:<8s} {p['n_layers']}L  norm:{ng}x")
            else:
                print(f"  ❌ {m}: {p['error']}")
        except Exception as e:
            print(f"  ❌ {m}: {e}")

    active_models = list(profiles.keys())

    catalog = {
        "selection": run_label,
        "requested": specs,
        "available": {model: profiles[model] for model in active_models},
        "failed": [model for model in MODELS if model not in profiles],
    }
    catalog_path = os.path.expanduser("~/neural-mirror/swarm_catalog.json")
    with open(catalog_path, "w") as handle:
        json.dump(catalog, handle, indent=2)
    print(f"\n  Catalog: {catalog_path}")

    if profile_only:
        print(f"  Profile-only complete: {len(active_models)}/{len(MODELS)} models available")
        return
    if len(active_models) < 2:
        print("Need at least 2 models!")
        return

    # Build roster
    roster = "Models at this roundtable:\n"
    for m, p in profiles.items():
        roster += f"  - {p['model_name']} ({p['params']}, {p['arch']}, {p['n_layers']}L, norm growth: {p.get('norm_growth','?')}x)\n"

    print(f"\n  Roster: {len(active_models)} models ready\n")

    round_findings = {}  # round_num -> {model_name: response}
    conference_consensus = None
    t_start = time.time()
    transcript_path = os.path.expanduser('~/neural-mirror/roundtable_transcript.json')

    for round_num in range(1, rounds + 1):
        print("━" * 65)
        labels = {1: "Self-Examination", 2: "Cross-Pollination", 3: "Synthesis"}
        label = labels.get(round_num, f"Round {round_num}")
        print(f"  Round {round_num}: {label}")
        print("━" * 65)
        print()

        # Build prior findings for this round
        if round_num == 1:
            prior = {}
        elif round_num == 2:
            prior = round_findings.get(1, {})
        else:
            # Flatten all prior rounds
            prior = {}
            for r in range(1, round_num):
                prior[str(r)] = round_findings.get(r, {})

        # The first two rounds are parallel briefs and ballots. If a third
        # round is requested, one elected chair synthesizes rather than paying
        # for every model to repeat nearly identical summaries.
        round_models = active_models
        if round_num >= 3:
            tier_rank = {"small": 0, "medium": 1, "large": 2, "xlarge": 3}
            research_leads = [
                m for m in active_models if spec_map[m].get("role") == "research_lead"
            ]
            chair = research_leads[0] if research_leads else max(
                active_models, key=lambda m: tier_rank.get(spec_map[m].get("tier"), 0)
            )
            round_models = [chair]
            print(f"  Chair: {profiles[chair]['model_name']} (single synthesis turn)\n")

        # Launch this round's participants in parallel.
        futures = {}
        runners = {
            "small": run_round_for_model,
            "medium": run_round_for_medium_model,
            "large": run_round_for_large_model,
            "xlarge": run_round_for_xlarge_model,
        }
        for m in round_models:
            runner = runners.get(spec_map[m].get("tier", "medium"), run_round_for_medium_model)
            fut = runner.spawn(m, round_num, profiles[m], roster, prior)
            futures[m] = fut

        # Collect results
        this_round = {}
        for m, fut in futures.items():
            p = profiles[m]
            try:
                result = fut.get()
                response = result.get('response', '')
                tools = result.get('tools_used', [])
                this_round[m] = response

                print(f"  {'─' * 60}")
                mname = p['model_name']
                print(f"  🪞 {mname} ({p['params']}, {p['arch']})")
                if tools:
                    print(f"     🔍 {', '.join(tools[:5])}")
                print()
                # Print response with wrapping
                for line in response.split('\n'):
                    line = line.strip()
                    if line:
                        wrapped = line[:120]
                        print(f"     {wrapped}")
                        if len(line) > 120:
                            for i in range(120, len(line), 120):
                                print(f"     {line[i:i+120]}")
                print()

            except Exception as e:
                print(f"  ❌ {m}: {e}")
                this_round[m] = f"Error: {e}"

        round_findings[round_num] = this_round
        if round_num == 2:
            motion, observations = _candidate_consensus(round_findings.get(1, {}))
            eligible_models = [item["model"] for item in observations]
            conference_consensus = _tally_consensus(
                motion, this_round, eligible_models=eligible_models
            )
            print("  🗳️  Conference motion result")
            print(f"     {conference_consensus['motion']}")
            print(
                "     "
                f"{conference_consensus['status'].upper()}: "
                f"{conference_consensus['accept']} accept, "
                f"{conference_consensus['reject']} reject, "
                f"{conference_consensus['abstain']} abstain"
            )
            if conference_consensus['excluded']:
                print(
                    "     Excluded (no canonical brief): "
                    + ", ".join(conference_consensus['excluded'])
                )

        _write_transcript(transcript_path, {
            'selection': run_label,
            'status': 'in_progress',
            'completed_rounds': round_num,
            'requested_rounds': rounds,
            'registry_specs': [spec_map[m] for m in active_models],
            'participants': {m: profiles[m] for m in active_models},
            'rounds': {str(r): findings for r, findings in round_findings.items()},
            'conference_consensus': conference_consensus,
            'elapsed': round(time.time() - t_start, 1),
        })
        print(f"  ✅ Round {round_num} complete (checkpoint saved)\n")

    elapsed = time.time() - t_start

    # Final summary
    print("━" * 65)
    print(f"  🪞 Roundtable Complete — {elapsed:.0f}s, {rounds} rounds, {len(active_models)} models")
    print("━" * 65)
    print()

    # Print the synthesis round highlights
    if rounds >= 3 and 3 in round_findings:
        print("  💡 Synthesis Highlights:")
        print()
        for m, resp in round_findings[3].items():
            p = profiles.get(m, {})
            print(f"  {p.get('model_name', m)}:")
            preview = resp[:300].replace('\n', ' ').strip()
            print(f"  {preview}...")
            print()

    # Save transcript
    transcript = {
        'selection': run_label,
        'status': 'complete',
        'completed_rounds': rounds,
        'requested_rounds': rounds,
        'registry_specs': [spec_map[m] for m in active_models],
        'participants': {m: profiles[m] for m in active_models},
        'rounds': {str(r): f for r, f in round_findings.items()},
        'conference_consensus': conference_consensus,
        'elapsed': round(elapsed, 1),
    }
    _write_transcript(transcript_path, transcript)
    print(f"  Transcript: {transcript_path}")
