#!/usr/bin/env python3
"""Check whether a HF safetensors model dir still carries MTP weights.

Reads config.json, hf_quant_config.json, the index and (authoritatively) every
safetensors header, then reports any MTP-related signal. Usage:
  python3 check-mtp.py <model-dir>
"""
import json
import struct
import sys
from pathlib import Path


def main() -> int:
    root = Path(sys.argv[1])
    print(f"model dir: {root}")

    cfg = json.loads((root / "config.json").read_text())
    print("\n--- config.json ---")
    for key in (
        "model_type",
        "architectures",
        "num_hidden_layers",
        "max_position_embeddings",
        "mtp_num_hidden_layers",
        "num_nextn_predict_layers",
        "speculative_config",
    ):
        print(f"  {key} = {cfg.get(key, '<absent>')}")
    mtp_cfg_keys = [k for k in cfg if "mtp" in k.lower() or "nextn" in k.lower()]
    print(f"  keys matching mtp/nextn: {mtp_cfg_keys}")

    quant_path = root / "hf_quant_config.json"
    if quant_path.exists():
        quant = json.loads(quant_path.read_text())
        print("\n--- hf_quant_config.json ---")
        print(f"  top-level keys: {list(quant)}")
        blob = json.dumps(quant)
        print(f"  contains 'mtp': {'mtp' in blob}")

    print("\n--- model.safetensors.index.json ---")
    index = json.loads((root / "model.safetensors.index.json").read_text())
    index_keys = list(index["weight_map"])
    index_mtp = [k for k in index_keys if k.startswith("mtp")]
    print(f"  total keys: {len(index_keys)}")
    print(f"  mtp keys:   {len(index_mtp)} {index_mtp[:5]}")

    print("\n--- safetensors headers (authoritative) ---")
    header_keys = []
    for shard in sorted(root.glob("*.safetensors")):
        with shard.open("rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(n))
        names = [k for k in header if k != "__metadata__"]
        header_keys.extend(names)
        mtp = [k for k in names if k.startswith("mtp")]
        print(
            f"  {shard.name}: {len(names)} tensors, mtp tensors: {len(mtp)} {mtp[:3]}"
        )
        meta = header.get("__metadata__")
        if meta:
            print(f"    __metadata__: {meta}")

    header_mtp = [k for k in header_keys if k.startswith("mtp")]
    print(f"\n  total header tensors: {len(header_keys)}")
    print(f"  header mtp tensors:   {len(header_mtp)} {header_mtp[:5]}")

    idx_set, hdr_set = set(index_keys), set(header_keys)
    print(f"\n  index vs header: index-only={len(idx_set - hdr_set)} header-only={len(hdr_set - idx_set)}")
    for name in sorted(hdr_set - idx_set)[:10]:
        print(f"    header-only: {name}")

    verdict = "NO MTP WEIGHTS PRESENT" if not header_mtp else "MTP WEIGHTS PRESENT"
    print(f"\nverdict: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())