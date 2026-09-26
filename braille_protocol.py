#!/usr/bin/env python3
"""
Neural Mirror — Braille Protocol ⠿

8-dot computer-braille Unicode cells as a structured byte encoding for
model introspection and co-LoRA proposals. Fixed codebooks reject unknown
module names and malformed numeric fields; they do not guarantee truth.

Each braille cell = 1 byte = 256 patterns. We define:

  HEADER CELL: identifies message type
    ⠁ (0x01) = architecture summary
    ⠃ (0x02) = layer stats
    ⠇ (0x04) = LoRA proposal  
    ⠏ (0x08) = training result
    ⠟ (0x10) = evaluation/vote
    ⠿ (0x3F) = cross-model comparison

  MODULE ENCODING: fixed 1-cell codes for LoRA target modules
    ⠁ = q_proj     ⠃ = k_proj     ⠇ = v_proj     ⠏ = o_proj
    ⠟ = gate_proj  ⠿ = up_proj    ⡿ = down_proj
    
  NUMBER ENCODING: 2-cell fixed-point for weights/stats
    Cell 1: integer part (signed, -128..127)
    Cell 2: fractional part (0..255 maps to 0.000..0.996)

  RANK ENCODING: 1-cell (0-255 maps directly)
    ⠁ = r=1, ⠐ = r=16, ⡀ = r=64

This is NOT for human reading. It is a custom machine-to-machine byte
protocol rendered with Unicode braille cells. Every valid field has a fixed
meaning, but a model can still select a valid yet unsupported value. The host
must validate evidence references and semantics separately. Unicode cells are
not guaranteed to be single tokenizer tokens; measure each model explicitly.

Usage:
    from braille_protocol import encode_proposal, decode_proposal
    from braille_protocol import encode_architecture, decode_architecture
"""

import struct

# ── Braille Cell Primitives ──────────────────────────────────

def cell(n: int) -> str:
    """Encode a byte as a Unicode braille character."""
    return chr(0x2800 + (n & 0xFF))

def uncell(c: str) -> int:
    """Decode a braille character to a byte."""
    return ord(c) - 0x2800

def cells(data: bytes) -> str:
    """Encode bytes as braille string."""
    return ''.join(cell(b) for b in data)

def uncells(s: str) -> bytes:
    """Decode braille string to bytes."""
    return bytes(uncell(c) for c in s if 0x2800 <= ord(c) <= 0x28FF)


# ── Message Types ────────────────────────────────────────────

MSG_ARCH    = 0x01
MSG_LAYER   = 0x02
MSG_LORA    = 0x04
MSG_TRAIN   = 0x08
MSG_EVAL    = 0x10
MSG_BRAID   = 0x20
MSG_CROSS   = 0x3F

BRAID_VERSION = 1
BRAID_FRAME_LENGTH = 14
BRAID_OPERATIONS = {
    'observe': 0x01,
    'support': 0x02,
    'challenge': 0x03,
    'propose_test': 0x04,
    'synthesize': 0x05,
}
CODE_TO_BRAID_OPERATION = {value: key for key, value in BRAID_OPERATIONS.items()}
BRAID_RELATIONS = {
    'increase': 0x01,
    'decrease': 0x02,
    'similar': 0x03,
    'different': 0x04,
    'supports': 0x05,
    'contradicts': 0x06,
    'needs_test': 0x07,
}
CODE_TO_BRAID_RELATION = {value: key for key, value in BRAID_RELATIONS.items()}
BRAID_CAVEAT_MASK = 0x0F

# ── Module Codes ─────────────────────────────────────────────

MODULE_CODES = {
    'q_proj':    0x01,
    'k_proj':    0x02,
    'v_proj':    0x04,
    'o_proj':    0x08,
    'gate_proj': 0x10,
    'up_proj':   0x20,
    'down_proj': 0x40,
}

CODE_TO_MODULE = {v: k for k, v in MODULE_CODES.items()}
KNOWN_MODULE_MASK = sum(MODULE_CODES.values())

# ── Architecture Codes ───────────────────────────────────────

ARCH_CODES = {
    'qwen3': 0x01, 'qwen2': 0x02, 'qwen35': 0x03,
    'llama': 0x10, 'gemma3': 0x20, 'gemma4': 0x21,
    'phi3':  0x30, 'mistral': 0x40,
}

CODE_TO_ARCH = {v: k for k, v in ARCH_CODES.items()}


# ── Number Encoding ──────────────────────────────────────────

def encode_float16(f: float) -> bytes:
    """Encode float as 2 bytes (IEEE 754 half-precision)."""
    return struct.pack('<e', max(-65504, min(65504, f)))

def decode_float16(b: bytes) -> float:
    """Decode 2 bytes to float."""
    return struct.unpack('<e', b)[0]

def encode_uint16(n: int) -> bytes:
    """Encode unsigned int as 2 bytes."""
    return struct.pack('<H', min(65535, max(0, n)))

def decode_uint16(b: bytes) -> int:
    return struct.unpack('<H', b)[0]


# ── Architecture Summary ─────────────────────────────────────

def encode_architecture(arch: str, n_layers: int, embed_dim: int,
                       n_heads: int, kv_heads: int, params_b: float,
                       norm_growth: float) -> str:
    """
    Encode architecture summary as braille string.
    Format: [MSG_ARCH] [arch_code] [layers] [embed_dim:2] [heads] [kv] [params_fp16:2] [norm_fp16:2]
    Total: 11 braille cells (token count is tokenizer-dependent)
    """
    data = bytes([
        MSG_ARCH,
        ARCH_CODES.get(arch, 0xFF),
        min(255, n_layers),
    ])
    data += encode_uint16(embed_dim)
    data += bytes([min(255, n_heads), min(255, kv_heads)])
    data += encode_float16(params_b)
    data += encode_float16(norm_growth)
    return cells(data)


def decode_architecture(braille: str) -> dict:
    """Decode braille architecture summary."""
    b = uncells(braille)
    if len(b) < 11 or b[0] != MSG_ARCH:
        return {'error': 'invalid architecture message'}
    return {
        'arch': CODE_TO_ARCH.get(b[1], f'unknown(0x{b[1]:02x})'),
        'n_layers': b[2],
        'embed_dim': decode_uint16(b[3:5]),
        'n_heads': b[5],
        'kv_heads': b[6],
        'params_b': round(decode_float16(b[7:9]), 2),
        'norm_growth': round(decode_float16(b[9:11]), 2),
    }


# ── LoRA Proposal ────────────────────────────────────────────

def encode_proposal(target_modules: list, rank: int, alpha: int,
                   dropout_pct: int = 5, model_idx: int = 0) -> str:
    """
    Encode LoRA proposal as braille string.
    Format: [MSG_LORA] [model_idx] [modules_bitmask] [rank] [alpha] [dropout]
    Total: 6 braille cells (token count is tokenizer-dependent)
    
    vs JSON: {"target_modules":["q_proj","v_proj"],"r":16,"lora_alpha":32,"lora_dropout":0.05}
    This uses 6 cells; actual token savings are tokenizer-dependent.
    """
    # Encode modules as bitmask
    mask = 0
    for m in target_modules:
        mask |= MODULE_CODES.get(m, 0)
    
    data = bytes([
        MSG_LORA,
        min(255, model_idx),
        mask,
        min(255, rank),
        min(255, alpha),
        min(255, dropout_pct),
    ])
    return cells(data)


def decode_proposal(braille: str) -> dict:
    """Decode a LoRA proposal permissively for backward compatibility."""
    b = uncells(braille)
    if len(b) < 6 or b[0] != MSG_LORA:
        return {'error': 'invalid proposal message'}
    
    # Decode module bitmask
    mask = b[2]
    modules = [name for code, name in sorted(CODE_TO_MODULE.items()) if mask & code]
    
    return {
        'model_idx': b[1],
        'target_modules': modules,
        'r': b[3],
        'lora_alpha': b[4],
        'lora_dropout': b[5] / 100,
    }


def _strict_bytes(braille: str) -> tuple[bytes | None, str | None]:
    """Decode only if every input character is exactly one braille cell."""
    if not isinstance(braille, str) or not braille:
        return None, 'EMPTY'
    if any(not (0x2800 <= ord(char) <= 0x28FF) for char in braille):
        return None, 'NON_BRAILLE'
    return bytes(ord(char) - 0x2800 for char in braille), None


def encode_proposal_strict(target_modules: list, rank: int, alpha: int,
                           dropout_pct: int = 5, model_idx: int = 0) -> str:
    """Encode a proposal after validating every bounded field."""
    unknown = sorted(set(target_modules) - set(MODULE_CODES))
    if unknown:
        raise ValueError(f"unknown modules: {unknown}")
    if not target_modules:
        raise ValueError("at least one target module is required")
    if not 0 <= model_idx <= 255:
        raise ValueError("model_idx must be 0..255")
    if not 1 <= rank <= 128:
        raise ValueError("rank must be 1..128")
    if not 1 <= alpha <= 255:
        raise ValueError("alpha must be 1..255")
    if not 0 <= dropout_pct <= 100:
        raise ValueError("dropout_pct must be 0..100")
    return encode_proposal(target_modules, rank, alpha, dropout_pct, model_idx)


def decode_proposal_strict(braille: str) -> dict:
    """Validate an exact six-cell proposal frame before decoding it."""
    b, error = _strict_bytes(braille)
    if error:
        return {'error': error}
    if len(b) != 6:
        return {'error': 'LENGTH'}
    if b[0] != MSG_LORA:
        return {'error': 'TYPE'}
    if b[2] == 0 or b[2] & ~KNOWN_MODULE_MASK:
        return {'error': 'MODULE_MASK'}
    if not 1 <= b[3] <= 128:
        return {'error': 'RANK'}
    if b[4] == 0:
        return {'error': 'ALPHA'}
    if b[5] > 100:
        return {'error': 'DROPOUT'}
    return decode_proposal(braille)


# ── Layer Stats ──────────────────────────────────────────────

def encode_layer_stats(layer_num: int, norm_mean: float, 
                       attn_std: float, ffn_std: float,
                       sparsity_pct: int = 0) -> str:
    """
    Encode layer statistics.
    Format: [MSG_LAYER] [layer_num] [norm_fp16:2] [attn_std_fp16:2] [ffn_std_fp16:2] [sparsity]
    Total: 10 braille cells
    """
    data = bytes([MSG_LAYER, min(255, layer_num)])
    data += encode_float16(norm_mean)
    data += encode_float16(attn_std)
    data += encode_float16(ffn_std)
    data += bytes([min(255, sparsity_pct)])
    return cells(data)


def decode_layer_stats(braille: str) -> dict:
    b = uncells(braille)
    if len(b) < 9 or b[0] != MSG_LAYER:
        return {'error': f'invalid layer message (len={len(b)}, type=0x{b[0]:02x} expected 0x{MSG_LAYER:02x})'}
    return {
        'layer': b[1],
        'norm_mean': round(decode_float16(b[2:4]), 4),
        'attn_std': round(decode_float16(b[4:6]), 4),
        'ffn_std': round(decode_float16(b[6:8]), 4),
        'sparsity_pct': b[8],
    }


# ── Training Result ──────────────────────────────────────────

def encode_training_result(model_idx: int, loss: float,
                          trainable_m: float, time_secs: int) -> str:
    """
    Encode training result.
    Format: [MSG_TRAIN] [model_idx] [loss_fp16:2] [trainable_fp16:2] [time:2]
    Total: 8 cells
    """
    data = bytes([MSG_TRAIN, min(255, model_idx)])
    data += encode_float16(loss)
    data += encode_float16(trainable_m)
    data += encode_uint16(min(65535, time_secs))
    return cells(data)


def decode_training_result(braille: str) -> dict:
    b = uncells(braille)
    if len(b) < 8 or b[0] != MSG_TRAIN:
        return {'error': 'invalid training message'}
    return {
        'model_idx': b[1],
        'loss': round(decode_float16(b[2:4]), 4),
        'trainable_m': round(decode_float16(b[4:6]), 1),
        'time_secs': decode_uint16(b[6:8]),
    }


# ── Vote / Evaluation ────────────────────────────────────────

def encode_vote(voter_idx: int, rankings: list, 
                agree_with_self: list) -> str:
    """
    Encode evaluation vote.
    Format: [MSG_EVAL] [voter_idx] [n_models] [rank1,rank2,...] [agree1,agree2,...]
    """
    n = len(rankings)
    data = bytes([MSG_EVAL, voter_idx, n])
    data += bytes(min(255, r) for r in rankings)
    data += bytes(1 if a else 0 for a in agree_with_self)
    return cells(data)


def decode_vote(braille: str) -> dict:
    b = uncells(braille)
    if len(b) < 3 or b[0] != MSG_EVAL:
        return {'error': 'invalid vote message'}
    n = b[2]
    if len(b) < 3 + 2 * n:
        return {'error': 'vote too short'}
    rankings = list(b[3:3+n])
    agrees = [bool(x) for x in b[3+n:3+2*n]]
    return {
        'voter_idx': b[1],
        'rankings': rankings,
        'agree_with_self': agrees,
    }


def encode_vote_strict(voter_idx: int, rankings: list,
                       agree_with_self: list) -> str:
    """Encode a vote only when ranking and agreement vectors are exact."""
    n = len(rankings)
    if not 1 <= n <= 32:
        raise ValueError("vote must rank 1..32 models")
    if not 0 <= voter_idx <= 255:
        raise ValueError("voter_idx must be 0..255")
    if sorted(rankings) != list(range(n)):
        raise ValueError("rankings must be a permutation of 0..n-1")
    if len(agree_with_self) != n or any(type(value) is not bool for value in agree_with_self):
        raise ValueError("agreement vector must contain n booleans")
    return encode_vote(voter_idx, rankings, agree_with_self)


def decode_vote_strict(braille: str) -> dict:
    """Validate an exact vote frame before decoding it."""
    b, error = _strict_bytes(braille)
    if error:
        return {'error': error}
    if len(b) < 3:
        return {'error': 'LENGTH'}
    if b[0] != MSG_EVAL:
        return {'error': 'TYPE'}
    n = b[2]
    if not 1 <= n <= 32 or len(b) != 3 + 2 * n:
        return {'error': 'LENGTH'}
    rankings = list(b[3:3+n])
    agreements = list(b[3+n:3+2*n])
    if sorted(rankings) != list(range(n)):
        return {'error': 'RANKINGS'}
    if any(value not in (0, 1) for value in agreements):
        return {'error': 'AGREEMENT'}
    return decode_vote(braille)


# ── Strict Braiding Frame ────────────────────────────────────

def crc8(data: bytes) -> int:
    """CRC-8/ATM (polynomial 0x07, initial value 0)."""
    checksum = 0
    for byte in data:
        checksum ^= byte
        for _ in range(8):
            checksum = ((checksum << 1) ^ 0x07) & 0xFF if checksum & 0x80 else (checksum << 1) & 0xFF
    return checksum


def encode_braid_strict(sender_idx: int, round_idx: int, operation: str,
                        subject_idx: int, evidence_id: int, relation: str,
                        confidence_pct: int, caveat_flags: int,
                        value: float) -> str:
    """Encode one versioned, checksummed, exact-length braid message."""
    if not 0 <= sender_idx <= 255:
        raise ValueError("sender_idx must be 0..255")
    if not 0 <= round_idx <= 255:
        raise ValueError("round_idx must be 0..255")
    if operation not in BRAID_OPERATIONS:
        raise ValueError(f"unknown braid operation: {operation}")
    if not 0 <= subject_idx <= 255:
        raise ValueError("subject_idx must be 0..255")
    if not 1 <= evidence_id <= 65535:
        raise ValueError("evidence_id must be 1..65535")
    if relation not in BRAID_RELATIONS:
        raise ValueError(f"unknown braid relation: {relation}")
    if not 0 <= confidence_pct <= 100:
        raise ValueError("confidence_pct must be 0..100")
    if not 0 <= caveat_flags <= BRAID_CAVEAT_MASK:
        raise ValueError(f"caveat_flags must use only mask 0x{BRAID_CAVEAT_MASK:02x}")
    body = bytes([
        MSG_BRAID, BRAID_VERSION, sender_idx, round_idx,
        BRAID_OPERATIONS[operation], subject_idx,
    ])
    body += encode_uint16(evidence_id)
    body += bytes([BRAID_RELATIONS[relation], confidence_pct, caveat_flags])
    body += encode_float16(value)
    return cells(body + bytes([crc8(body)]))


def decode_braid_strict(braille: str, allowed_evidence_ids=None) -> dict:
    """Reject malformed braid frames and optionally enforce an evidence catalog."""
    data, error = _strict_bytes(braille)
    if error:
        return {'error': error}
    if len(data) != BRAID_FRAME_LENGTH:
        return {'error': 'LENGTH'}
    if data[0] != MSG_BRAID:
        return {'error': 'TYPE'}
    if data[1] != BRAID_VERSION:
        return {'error': 'VERSION'}
    if data[4] not in CODE_TO_BRAID_OPERATION:
        return {'error': 'OPERATION'}
    if data[8] not in CODE_TO_BRAID_RELATION:
        return {'error': 'RELATION'}
    if data[9] > 100:
        return {'error': 'CONFIDENCE'}
    if data[10] & ~BRAID_CAVEAT_MASK:
        return {'error': 'CAVEAT'}
    if crc8(data[:-1]) != data[-1]:
        return {'error': 'CHECKSUM'}
    evidence_id = decode_uint16(data[6:8])
    if evidence_id == 0:
        return {'error': 'EVIDENCE'}
    if allowed_evidence_ids is not None and evidence_id not in set(allowed_evidence_ids):
        return {'error': 'EVIDENCE'}
    return {
        'version': data[1],
        'sender_idx': data[2],
        'round_idx': data[3],
        'operation': CODE_TO_BRAID_OPERATION[data[4]],
        'subject_idx': data[5],
        'evidence_id': evidence_id,
        'relation': CODE_TO_BRAID_RELATION[data[8]],
        'confidence_pct': data[9],
        'caveat_flags': data[10],
        'value': round(decode_float16(data[11:13]), 6),
        'checksum': data[13],
    }


# ── Full Cross-Model Comparison ──────────────────────────────

def encode_cross_comparison(models: list) -> str:
    """
    Encode full cross-model comparison as compact braille.
    Each model entry: [arch_summary] + [layer0_stats] + [layerN_stats]
    
    A 3-model comparison that would be ~2000 tokens in JSON
    becomes ~100 braille cells.
    """
    result = cell(MSG_CROSS) + cell(len(models))
    
    for m in models:
        result += encode_architecture(
            m.get('arch', '?'),
            m.get('n_layers', 0),
            m.get('embed_dim', 0),
            m.get('n_heads', 0),
            m.get('kv_heads', 0),
            m.get('params_b', 0),
            m.get('norm_growth', 0),
        )
    
    return result


# ── Demo / Test ──────────────────────────────────────────────

def demo():
    """Show the encoding in action."""
    print("🪞⠿ Neural Mirror — Braille Protocol Demo\n")
    
    # Architecture
    arch_braille = encode_architecture('qwen3', 28, 2048, 16, 8, 2.03, 4.86)
    arch_decoded = decode_architecture(arch_braille)
    print(f"Architecture: {arch_braille}")
    print(f"  Decoded: {arch_decoded}")
    print(f"  Braille cells: {len(arch_braille)} (benchmark tokens per tokenizer)\n")
    
    # LoRA proposal
    proposal_braille = encode_proposal(['q_proj', 'v_proj', 'o_proj'], rank=16, alpha=32)
    proposal_decoded = decode_proposal(proposal_braille)
    print(f"LoRA proposal: {proposal_braille}")
    print(f"  Decoded: {proposal_decoded}")
    print(f"  Braille cells: {len(proposal_braille)} (benchmark tokens per tokenizer)\n")
    
    # Layer stats
    l0 = encode_layer_stats(0, 0.088, 0.0006, 0.0008, 92)
    l27 = encode_layer_stats(27, 17.67, 0.0009, 0.0001, 96)
    print(f"Layer 0 stats:  {l0}")
    print(f"  Decoded: {decode_layer_stats(l0)}")
    print(f"Layer 27 stats: {l27}")
    print(f"  Decoded: {decode_layer_stats(l27)}\n")
    
    # Training result
    train = encode_training_result(0, 0.856, 5.5, 40)
    print(f"Training result: {train}")
    print(f"  Decoded: {decode_training_result(train)}\n")
    
    # Vote
    vote = encode_vote(0, [1, 3, 2], [True, False, True])
    print(f"Vote: {vote}")
    print(f"  Decoded: {decode_vote(vote)}\n")
    
    # Representation-size comparison (not a tokenizer benchmark)
    json_example = '{"target_modules":["q_proj","v_proj","o_proj"],"r":16,"lora_alpha":32,"lora_dropout":0.05,"reasoning":"targeting attention projections for mathematical reasoning improvement"}'
    print(f"Token comparison:")
    print(f"  JSON proposal:    ~{len(json_example.split())} words, ~{len(json_example)//4} tokens")
    print(f"  Braille proposal: {len(proposal_braille)} cells; token count depends on tokenizer")
    print("  Syntax safety:    fixed module codebook; unknown names cannot enter the frame")
    print("  Truth guarantee:  NONE — evidence and semantics still require validation")


if __name__ == '__main__':
    demo()
