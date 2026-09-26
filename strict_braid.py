#!/usr/bin/env python3
"""Strict, auditable braille-braiding state machine.

This module proves the wire protocol and validation path independently of model
quality. A model may propose bounded fields, but only the host can assign
canonical evidence IDs and commit a validated frame to the braid.
"""

import hashlib
import json
import os
from dataclasses import dataclass, field

from braille_protocol import decode_braid_strict, decode_float16, encode_braid_strict, encode_float16


class BraidRejected(ValueError):
    pass


@dataclass
class EvidenceRecord:
    evidence_id: int
    owner_idx: int
    label: str
    payload: dict
    canonical_value: float | None = None


@dataclass
class StrictBraid:
    participants: list[str]
    evidence: dict[int, EvidenceRecord]
    frames: list[dict] = field(default_factory=list)
    seen_hashes: set[str] = field(default_factory=set)

    def submit(self, frame: str) -> dict:
        decoded = decode_braid_strict(frame, set(self.evidence))
        if "error" in decoded:
            raise BraidRejected(decoded["error"])
        sender = decoded["sender_idx"]
        subject = decoded["subject_idx"]
        if sender >= len(self.participants) or subject >= len(self.participants):
            raise BraidRejected("PARTICIPANT")
        if decoded["round_idx"] == 0:
            raise BraidRejected("ROUND")
        if self.frames and decoded["round_idx"] < self.frames[-1]["decoded"]["round_idx"]:
            raise BraidRejected("ROUND_ORDER")
        record = self.evidence[decoded["evidence_id"]]
        if decoded["operation"] == "observe" and record.owner_idx != sender:
            raise BraidRejected("EVIDENCE_OWNER")
        if record.canonical_value is not None:
            expected_value = round(decode_float16(encode_float16(record.canonical_value)), 6)
            if decoded["value"] != expected_value:
                raise BraidRejected("EVIDENCE_VALUE")
        digest = hashlib.sha256(frame.encode("utf-8")).hexdigest()
        if digest in self.seen_hashes:
            raise BraidRejected("DUPLICATE")
        committed = {
            "sequence": len(self.frames) + 1,
            "wire": frame,
            "wire_sha256": digest,
            "sender": self.participants[sender],
            "subject": self.participants[subject],
            "evidence_label": record.label,
            "decoded": decoded,
        }
        self.frames.append(committed)
        self.seen_hashes.add(digest)
        return committed

    def transcript(self) -> dict:
        canonical = {
            "protocol": "neural-mirror-strict-braid-v1",
            "participants": self.participants,
            "evidence": {
                str(key): {
                    "owner_idx": value.owner_idx,
                    "label": value.label,
                    "payload": value.payload,
                    "canonical_value": value.canonical_value,
                }
                for key, value in sorted(self.evidence.items())
            },
            "frames": self.frames,
        }
        encoded = json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode("utf-8")
        canonical["transcript_sha256"] = hashlib.sha256(encoded).hexdigest()
        return canonical


def prototype() -> dict:
    participants = [
        "qwen3:4b+qwen3-4b-braille-literacy-v3",
        "llama3.2:3b+llama3.2-3b-braille-literacy-v1",
    ]
    evidence = {
        1: EvidenceRecord(
            1, 0, "QWEN.ADAPTER.TOP_UPDATE",
            {"layer": 30, "module": "gate_proj", "delta_rms_upper_bound": 0.0010899367},
            0.0010899367,
        ),
        2: EvidenceRecord(
            2, 1, "LLAMA.ADMISSION",
            {"score": "28/28", "claim": "exact protocol exam passed"},
            1.0,
        ),
    }
    braid = StrictBraid(participants, evidence)
    frames = [
        encode_braid_strict(0, 1, "observe", 0, 1, "increase", 90, 0x01, 0.0010899367),
        encode_braid_strict(1, 1, "observe", 1, 2, "supports", 100, 0x00, 1.0),
        encode_braid_strict(1, 2, "challenge", 0, 1, "needs_test", 95, 0x05, 0.0010899367),
        encode_braid_strict(0, 2, "propose_test", 1, 2, "needs_test", 98, 0x04, 1.0),
    ]
    for frame in frames:
        braid.submit(frame)

    # Negative controls: corruption and unsupported evidence must fail closed.
    corrupted = frames[0][:-1] + chr(0x2800 + ((ord(frames[0][-1]) - 0x2800 + 1) & 0xFF))
    controls = {}
    for label, candidate in {
        "checksum_corruption": corrupted,
        "unsupported_evidence": encode_braid_strict(0, 3, "observe", 0, 999, "supports", 50, 0, 0.0),
        "mutated_evidence_value": encode_braid_strict(0, 3, "observe", 0, 1, "supports", 50, 0, 0.5),
    }.items():
        try:
            braid.submit(candidate)
            controls[label] = "FAILED_OPEN"
        except BraidRejected as error:
            controls[label] = f"REJECTED:{error}"
    result = braid.transcript()
    result["negative_controls"] = controls
    return result


def main():
    result = prototype()
    output = os.path.expanduser("~/neural-mirror/strict_braid_prototype.json")
    temporary = output + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, output)
    print(f"Committed {len(result['frames'])} strict frames")
    print(json.dumps(result["negative_controls"], indent=2))
    print(f"Transcript: {output}")


if __name__ == "__main__":
    main()
