"""Strict computer-braille literacy gate for the Neural Mirror swarm.

This tests competence with Neural Mirror's custom byte protocol, not knowledge
of literary braille. Models receive the complete byte/cell legend and protocol
specification, then must encode, decode, and reject unseen held-out frames.
Admission requires 100% exact accuracy on every task across every seed.

Usage:
  MODAL_PROFILE=salus modal run braille_literacy.py
  MODAL_PROFILE=salus modal run braille_literacy.py --models 'qwen3:4b,phi4-mini,mistral:7b'
"""

import json
import os
import random
import subprocess
import time
import urllib.error
import urllib.request

import modal

from braille_protocol import (
    CODE_TO_MODULE,
    KNOWN_MODULE_MASK,
    MODULE_CODES,
    MSG_EVAL,
    MSG_LORA,
    cell,
    cells,
    decode_proposal_strict,
    decode_vote_strict,
    encode_proposal_strict,
    encode_vote_strict,
)
from swarm_registry import spec_for_model

app = modal.App("neural-mirror-braille-literacy")
model_cache = modal.Volume.from_name("neural-mirror-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "zstd", "procps")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    .add_local_file("braille_protocol.py", "/root/braille_protocol.py")
    .add_local_file("swarm_registry.py", "/root/swarm_registry.py")
)

DEFAULT_MODELS = [
    "qwen3:4b",
    "phi4-mini",
    "mistral:7b",
    "gemma3:4b",
    "llama3.2:3b",
]
SEEDS = [17, 29]
TASKS_PER_SEED = 14

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "proposal_decodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "model_idx": {"type": "integer"},
                    "target_modules": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(MODULE_CODES)},
                    },
                    "rank": {"type": "integer"},
                    "alpha": {"type": "integer"},
                    "dropout_pct": {"type": "integer"},
                },
                "required": ["id", "model_idx", "target_modules", "rank", "alpha", "dropout_pct"],
                "additionalProperties": False,
            },
        },
        "proposal_encodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "packet": {"type": "string"}},
                "required": ["id", "packet"],
                "additionalProperties": False,
            },
        },
        "vote_decodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "voter_idx": {"type": "integer"},
                    "rankings": {"type": "array", "items": {"type": "integer"}},
                    "agree": {"type": "array", "items": {"type": "boolean"}},
                },
                "required": ["id", "voter_idx", "rankings", "agree"],
                "additionalProperties": False,
            },
        },
        "vote_encodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "packet": {"type": "string"}},
                "required": ["id", "packet"],
                "additionalProperties": False,
            },
        },
        "malformed": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "error": {
                        "type": "string",
                        "enum": [
                            "EMPTY", "NON_BRAILLE", "LENGTH", "TYPE", "MODULE_MASK",
                            "RANK", "ALPHA", "DROPOUT", "RANKINGS", "AGREEMENT",
                        ],
                    },
                },
                "required": ["id", "error"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "proposal_decodes", "proposal_encodes", "vote_decodes",
        "vote_encodes", "malformed",
    ],
    "additionalProperties": False,
}


def _start_ollama(tier):
    env = os.environ.copy()
    env["OLLAMA_MODELS"] = "/cache/ollama"
    if tier != "small":
        env["OLLAMA_FLASH_ATTENTION"] = "1"
        env["OLLAMA_KV_CACHE_TYPE"] = "q8_0"
    proc = subprocess.Popen(
        ["ollama", "serve"], env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(90):
        if proc.poll() is not None:
            raise RuntimeError(f"Ollama exited during startup ({proc.returncode})")
        try:
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
            return proc, env
        except Exception:
            time.sleep(1)
    proc.terminate()
    raise RuntimeError("Ollama failed to start")


def _pull_model(model_name, env, tier):
    timeout = 3600 if tier in ("large", "xlarge") else 1200
    result = subprocess.run(
        ["ollama", "pull", model_name], env=env,
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode:
        raise RuntimeError(f"pull failed: {result.stderr[-1000:]}")


def _canonical_modules(modules):
    return [
        name for code, name in sorted(CODE_TO_MODULE.items())
        if name in modules
    ]


def _build_tasks(seed):
    rng = random.Random(seed)
    module_names = list(MODULE_CODES)
    proposal_decodes = []
    proposal_encodes = []
    vote_decodes = []
    vote_encodes = []

    for index in range(3):
        selected = rng.sample(module_names, rng.randint(1, 4))
        record = {
            "model_idx": rng.randint(0, 12),
            "target_modules": _canonical_modules(selected),
            "rank": rng.choice([4, 8, 13, 16, 22, 32, 64]),
            "alpha": rng.choice([8, 16, 26, 32, 44, 64, 128]),
            "dropout_pct": rng.choice([0, 3, 5, 10, 20]),
        }
        packet = encode_proposal_strict(
            record["target_modules"], record["rank"], record["alpha"],
            record["dropout_pct"], record["model_idx"],
        )
        proposal_decodes.append({"id": f"PD{index + 1}", "packet": packet, "expected": record})

    for index in range(3):
        selected = rng.sample(module_names, rng.randint(1, 4))
        record = {
            "model_idx": rng.randint(0, 12),
            "target_modules": _canonical_modules(selected),
            "rank": rng.choice([4, 8, 13, 16, 22, 32, 64]),
            "alpha": rng.choice([8, 16, 26, 32, 44, 64, 128]),
            "dropout_pct": rng.choice([0, 3, 5, 10, 20]),
        }
        expected = encode_proposal_strict(
            record["target_modules"], record["rank"], record["alpha"],
            record["dropout_pct"], record["model_idx"],
        )
        proposal_encodes.append({"id": f"PE{index + 1}", **record, "expected": expected})

    for index in range(2):
        n = rng.randint(3, 6)
        rankings = list(range(n))
        rng.shuffle(rankings)
        agrees = [bool(rng.getrandbits(1)) for _ in range(n)]
        voter = rng.randint(0, n - 1)
        packet = encode_vote_strict(voter, rankings, agrees)
        vote_decodes.append({
            "id": f"VD{index + 1}", "packet": packet,
            "expected": {"voter_idx": voter, "rankings": rankings, "agree": agrees},
        })

    for index in range(2):
        n = rng.randint(3, 6)
        rankings = list(range(n))
        rng.shuffle(rankings)
        agrees = [bool(rng.getrandbits(1)) for _ in range(n)]
        voter = rng.randint(0, n - 1)
        expected = encode_vote_strict(voter, rankings, agrees)
        vote_encodes.append({
            "id": f"VE{index + 1}", "voter_idx": voter,
            "rankings": rankings, "agree": agrees, "expected": expected,
        })

    valid = encode_proposal_strict(["q_proj", "v_proj"], 16, 32, 5, 0)
    malformed = [
        {"id": "M1", "kind": "proposal", "packet": valid[:-1], "expected": "LENGTH"},
        {
            "id": "M2", "kind": "proposal",
            "packet": cells(bytes([MSG_LORA, 0, 0x80, 16, 32, 5])),
            "expected": "MODULE_MASK",
        },
        {
            "id": "M3", "kind": "proposal",
            "packet": cells(bytes([MSG_LORA, 0, 0x01, 16, 32, 101])),
            "expected": "DROPOUT",
        },
        {
            "id": "M4", "kind": "vote",
            "packet": cells(bytes([MSG_EVAL, 0, 3, 0, 0, 2, 1, 0, 1])),
            "expected": "RANKINGS",
        },
    ]

    return {
        "proposal_decodes": proposal_decodes,
        "proposal_encodes": proposal_encodes,
        "vote_decodes": vote_decodes,
        "vote_encodes": vote_encodes,
        "malformed": malformed,
    }


def expected_answer(tasks):
    """Return the one exact semantic answer accepted by the literacy scorer."""
    return {
        "proposal_decodes": [
            {"id": item["id"], **item["expected"]}
            for item in tasks["proposal_decodes"]
        ],
        "proposal_encodes": [
            {"id": item["id"], "packet": item["expected"]}
            for item in tasks["proposal_encodes"]
        ],
        "vote_decodes": [
            {"id": item["id"], **item["expected"]}
            for item in tasks["vote_decodes"]
        ],
        "vote_encodes": [
            {"id": item["id"], "packet": item["expected"]}
            for item in tasks["vote_encodes"]
        ],
        "malformed": [
            {"id": item["id"], "error": item["expected"]}
            for item in tasks["malformed"]
        ],
    }


def _public_tasks(tasks):
    return {
        "proposal_decodes": [
            {"id": item["id"], "packet": item["packet"]}
            for item in tasks["proposal_decodes"]
        ],
        "proposal_encodes": [
            {key: value for key, value in item.items() if key != "expected"}
            for item in tasks["proposal_encodes"]
        ],
        "vote_decodes": [
            {"id": item["id"], "packet": item["packet"]}
            for item in tasks["vote_decodes"]
        ],
        "vote_encodes": [
            {key: value for key, value in item.items() if key != "expected"}
            for item in tasks["vote_encodes"]
        ],
        "malformed": [
            {key: value for key, value in item.items() if key != "expected"}
            for item in tasks["malformed"]
        ],
    }


def _legend():
    return " ".join(f"{value:02X}={cell(value)}" for value in range(256))


def _literacy_prompt(tasks):
    example = encode_proposal_strict(["q_proj", "v_proj"], 16, 32, 5, 0)
    return f"""You are taking an exact computer-braille protocol literacy test.
Return only the requested JSON. Do not explain, repair, or omit any task.

BYTE ALPHABET
A cell is U+2800 plus its byte value. Complete lookup table:
{_legend()}

PROPOSAL FRAME (exactly 6 cells)
[04 type][model_idx][module bitmask][rank][alpha][dropout percent]
Module bits in canonical output order:
01=q_proj 02=k_proj 04=v_proj 08=o_proj 10=gate_proj 20=up_proj 40=down_proj
Rules: mask must be nonzero and use no other bits; rank 1..128; alpha 1..255; dropout 0..100.
Example semantic record model=0, q_proj+v_proj, rank=16, alpha=32, dropout=5 encodes as {example}

VOTE FRAME (exactly 3+2N cells)
[10 type][voter_idx][N][N ranking bytes][N agreement bytes]
Rules: N=1..32; rankings must be a permutation of 0..N-1; agreement bytes must be 00 or 01.

STRICT ERROR PRECEDENCE
EMPTY/NON_BRAILLE, then LENGTH, TYPE, then field errors.
Proposal field errors: MODULE_MASK, RANK, ALPHA, DROPOUT.
Vote field errors: RANKINGS, AGREEMENT.

For decoded proposal output use keys id, model_idx, target_modules, rank, alpha, dropout_pct.
For decoded vote output use keys id, voter_idx, rankings, agree.
For encoding output use only id and packet. Copy actual braille cells, never hex text.

HELD-OUT TASKS
{json.dumps(_public_tasks(tasks), ensure_ascii=False)}"""


def _request(host, endpoint, payload, timeout=1200):
    req = urllib.request.Request(
        host + endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")[:1000]
        raise RuntimeError(f"Ollama HTTP {error.code}: {body}") from error


def _run_test_turn(model_name, tasks, host="http://localhost:11434"):
    prompt = _literacy_prompt(tasks)
    if model_name.split(":", 1)[0] == "qwen3":
        prompt = "/no_think\n" + prompt
    prediction_budget = 8000 if model_name.split(":", 1)[0] == "gpt-oss" else 2200
    options = {"temperature": 0.0, "num_ctx": 16384, "num_predict": prediction_budget}
    chat_error = None
    try:
        raw = _request(host, "/api/chat", {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "Exact protocol conformance test. Output only schema-valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            "format": RESPONSE_SCHEMA,
            "think": False,
            "stream": False,
            "options": options,
        })
        content = raw.get("message", {}).get("content", "")
        if content:
            return json.loads(content), "chat", raw
        thinking = raw.get("message", {}).get("thinking", "")
        chat_error = f"blank content after {len(thinking)} thinking characters"
    except Exception as error:
        chat_error = str(error)

    raw = _request(host, "/api/generate", {
        "model": model_name,
        "prompt": prompt,
        "format": RESPONSE_SCHEMA,
        "think": False,
        "stream": False,
        "options": options,
    })
    content = raw.get("response", "")
    if not content:
        raise RuntimeError(f"empty chat and generate response; chat={chat_error}")
    return json.loads(content), "generate", raw


def _map_unique(rows):
    result = {}
    duplicates = []
    for row in rows if isinstance(rows, list) else []:
        identifier = row.get("id") if isinstance(row, dict) else None
        if not identifier or identifier in result:
            duplicates.append(identifier)
        else:
            result[identifier] = row
    return result, duplicates


def _score(tasks, answer):
    checks = []

    def add(category, identifier, passed, expected, actual):
        checks.append({
            "category": category, "id": identifier, "passed": bool(passed),
            "expected": expected, "actual": actual,
        })

    output_maps = {}
    duplicates = []
    for category in (
        "proposal_decodes", "proposal_encodes", "vote_decodes",
        "vote_encodes", "malformed",
    ):
        output_maps[category], found = _map_unique(answer.get(category, []))
        duplicates.extend((category, item) for item in found)

    for item in tasks["proposal_decodes"]:
        actual = output_maps["proposal_decodes"].get(item["id"])
        expected = {"id": item["id"], **item["expected"]}
        add("proposal_decode", item["id"], actual == expected, expected, actual)

    for item in tasks["proposal_encodes"]:
        actual = output_maps["proposal_encodes"].get(item["id"])
        expected = {"id": item["id"], "packet": item["expected"]}
        add("proposal_encode", item["id"], actual == expected, expected, actual)

    for item in tasks["vote_decodes"]:
        actual = output_maps["vote_decodes"].get(item["id"])
        expected = {"id": item["id"], **item["expected"]}
        add("vote_decode", item["id"], actual == expected, expected, actual)

    for item in tasks["vote_encodes"]:
        actual = output_maps["vote_encodes"].get(item["id"])
        expected = {"id": item["id"], "packet": item["expected"]}
        add("vote_encode", item["id"], actual == expected, expected, actual)

    for item in tasks["malformed"]:
        actual = output_maps["malformed"].get(item["id"])
        expected = {"id": item["id"], "error": item["expected"]}
        add("malformed", item["id"], actual == expected, expected, actual)

    expected_ids = {
        category: {item["id"] for item in tasks[category]}
        for category in output_maps
    }
    extras = {
        category: sorted(set(rows) - expected_ids[category])
        for category, rows in output_maps.items()
        if set(rows) - expected_ids[category]
    }
    passed = sum(check["passed"] for check in checks)
    return {
        "passed": passed == len(checks) and not duplicates and not extras,
        "correct": passed,
        "total": len(checks),
        "checks": checks,
        "duplicates": duplicates,
        "extras": extras,
    }


def _run_literacy_impl(model_name, seeds):
    spec = spec_for_model(model_name)
    proc, env = _start_ollama(spec.get("tier", "medium"))
    try:
        _pull_model(model_name, env, spec.get("tier", "medium"))
        seed_results = []
        for seed in seeds:
            tasks = _build_tasks(seed)
            started = time.time()
            try:
                answer, transport, raw = _run_test_turn(model_name, tasks)
                score = _score(tasks, answer)
                score.update({
                    "seed": seed,
                    "transport": transport,
                    "elapsed_seconds": round(time.time() - started, 1),
                    "prompt_tokens": raw.get("prompt_eval_count"),
                    "output_tokens": raw.get("eval_count"),
                })
            except Exception as error:
                score = {
                    "seed": seed, "passed": False, "correct": 0,
                    "total": TASKS_PER_SEED, "error": str(error),
                    "elapsed_seconds": round(time.time() - started, 1),
                }
            seed_results.append(score)
        return {
            "model": model_name,
            "family": spec.get("family"),
            "tier": spec.get("tier"),
            "passed": all(result["passed"] for result in seed_results),
            "correct": sum(result["correct"] for result in seed_results),
            "total": sum(result["total"] for result in seed_results),
            "seeds": seed_results,
        }
    finally:
        proc.terminate()


@app.function(image=image, volumes={"/cache": model_cache}, gpu="L4", cpu=4, memory=24576, timeout=2400)
def test_literacy(model_name: str, seeds: list) -> dict:
    return _run_literacy_impl(model_name, seeds)


@app.function(image=image, volumes={"/cache": model_cache}, gpu="A100", cpu=8, memory=32768, timeout=3600)
def test_literacy_large(model_name: str, seeds: list) -> dict:
    return _run_literacy_impl(model_name, seeds)


@app.function(image=image, volumes={"/cache": model_cache}, gpu="H100", cpu=8, memory=32768, timeout=3600)
def test_literacy_xlarge(model_name: str, seeds: list) -> dict:
    return _run_literacy_impl(model_name, seeds)


@app.local_entrypoint()
def main(models: str = ",".join(DEFAULT_MODELS)):
    selected = [name.strip() for name in models.split(",") if name.strip()]
    print("━" * 72)
    print("  ⠿ Neural Mirror — Strict Computer-Braille Literacy Gate")
    print(f"  Admission: 100% exact on {len(SEEDS)} seeds × {TASKS_PER_SEED} tasks")
    print("━" * 72)

    runners = {"small": test_literacy, "medium": test_literacy, "large": test_literacy_large, "xlarge": test_literacy_xlarge}
    futures = {
        model: runners.get(spec_for_model(model).get("tier", "medium"), test_literacy).spawn(model, SEEDS)
        for model in selected
    }
    results = []
    for model, future in futures.items():
        try:
            result = future.get()
        except Exception as error:
            result = {
                "model": model, "family": spec_for_model(model).get("family"),
                "passed": False, "correct": 0,
                "total": len(SEEDS) * TASKS_PER_SEED, "error": str(error),
            }
        results.append(result)
        icon = "✅" if result["passed"] else "❌"
        print(f"  {icon} {model:<24s} {result['correct']:>2}/{result['total']:<2} {'ADMIT' if result['passed'] else 'EXCLUDE'}")
        for seed_result in result.get("seeds", []):
            if not seed_result["passed"]:
                failures = [
                    f"{item['category']}/{item['id']}"
                    for item in seed_result.get("checks", []) if not item["passed"]
                ]
                detail = seed_result.get("error") or ", ".join(failures[:8])
                print(f"       seed {seed_result['seed']}: {detail}")

    admitted = [result["model"] for result in results if result["passed"]]
    output = {
        "protocol": "neural-mirror-computer-braille-v1",
        "criterion": "100% exact encode/decode/malformed detection on every held-out seed",
        "seeds": SEEDS,
        "tasks_per_seed": TASKS_PER_SEED,
        "admitted": admitted,
        "excluded": [result["model"] for result in results if not result["passed"]],
        "results": results,
    }
    path = os.path.expanduser("~/neural-mirror/braille_literacy_results.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, path)
    print(f"\n  Admitted: {len(admitted)}/{len(selected)}")
    print(f"  Results: {path}")
