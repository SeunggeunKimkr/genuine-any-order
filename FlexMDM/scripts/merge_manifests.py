"""Combine per-dataset metadata.json files into a unified manifest.json.

Each parallel pretokenize job writes <root>/<dataset>/metadata.json *and* a
single-dataset manifest.json at <root>/manifest.json. The four jobs race
on that top-level file, so its final contents are not deterministic. This
script reads every per-dataset metadata.json and rebuilds a unified
manifest in the same shape pretokenize_datasets() produces, so the
training dataloader (_load_pretokenized_manifest) can consume it.

Usage
-----
    python merge_pretokenized_manifests.py --root /path/to/pretokenized
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from flexmdm.data import (  # noqa: E402
    DEFAULT_MANIFEST_FILENAME,
    DEFAULT_TOKENIZER,
    _filters_snapshot,
    load_config,
)


def _read_json(path: str):
    with open(path, "r") as handle:
        return json.load(handle)


def _write_json(path: str, payload) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Pretokenized output root")
    p.add_argument(
        "--config",
        default=None,
        help="Optional YAML config to pull dataset_names + repeat factors from",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Override dataset names (else: all subdirs with metadata.json)",
    )
    args = p.parse_args()

    root = os.path.abspath(os.path.expanduser(args.root))
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Root does not exist: {root}")

    if args.datasets is not None:
        names = list(args.datasets)
    else:
        names = sorted(
            entry
            for entry in os.listdir(root)
            if os.path.isfile(os.path.join(root, entry, "metadata.json"))
        )
    if not names:
        raise RuntimeError(f"No per-dataset metadata.json found under {root}")

    datasets_payload = {}
    seen_config_fields = {
        "tokenizer": DEFAULT_TOKENIZER,
        "trust_remote_code": True,
        "max_length": None,
        "pad_token_id": None,
        "bos_token_id": None,
        "separator_kind": None,
        "separator_token_id": None,
        "prepend_bos": None,
    }
    for name in names:
        meta_path = os.path.join(root, name, "metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(meta_path)
        meta = _read_json(meta_path)
        datasets_payload[name] = meta
        for key in ("max_length", "pad_token_id", "bos_token_id",
                    "separator_kind", "separator_token_id", "prepend_bos"):
            v = meta.get(key)
            if v is None:
                continue
            if seen_config_fields[key] is None:
                seen_config_fields[key] = v
            elif seen_config_fields[key] != v:
                print(
                    f"WARNING: {key} differs across datasets: "
                    f"{seen_config_fields[key]!r} vs {v!r}",
                    file=sys.stderr,
                )

    repeat_factors = {}
    config_dataset_names = list(names)
    if args.config:
        cfg = load_config(args.config)
        data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
        configured_repeats = data_cfg.get("dataloader_repeat_factors", {}) or {}
        for n in names:
            repeat_factors[n] = int(configured_repeats.get(n, 1))
        cfg_names = data_cfg.get("dataset_names")
        if cfg_names:
            config_dataset_names = [str(x) for x in cfg_names]

    manifest = {
        "created_at_unix": time.time(),
        "finished_at_unix": time.time(),
        "elapsed_sec": 0.0,
        "output_root": root,
        "config": {
            "tokenizer": seen_config_fields["tokenizer"],
            "trust_remote_code": seen_config_fields["trust_remote_code"],
            "max_length": seen_config_fields["max_length"],
            "pad_token_id": seen_config_fields["pad_token_id"],
            "bos_token_id": seen_config_fields["bos_token_id"],
            "separator_kind": seen_config_fields["separator_kind"],
            "separator_token_id": seen_config_fields["separator_token_id"],
            "prepend_bos": seen_config_fields["prepend_bos"],
            "dataset_names": config_dataset_names,
            "limit": None,
            "filters": _filters_snapshot(names),
            "dataloader_repeat_factor_hint": repeat_factors,
        },
        "total_written": sum(int(d.get("kept", 0)) for d in datasets_payload.values()),
        "datasets": datasets_payload,
    }
    out = os.path.join(root, DEFAULT_MANIFEST_FILENAME)
    manifest["manifest_path"] = out
    _write_json(out, manifest)
    total = manifest["total_written"]
    print(f"Wrote merged manifest with {len(names)} datasets, total_written={total:,} → {out}")
    for n in names:
        kept = datasets_payload[n].get("kept", "?")
        seen = datasets_payload[n].get("seen", "?")
        print(f"  {n:38s} seen={seen:>10}  kept={kept:>10}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
