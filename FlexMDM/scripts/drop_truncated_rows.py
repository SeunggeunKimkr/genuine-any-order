"""Drop rows whose answer was truncated to fit max_length.

Identification: a row was truncated iff its stored seq_lens equals max_length.
That's because _encode_tokenization_batch sets:
    answer_capacity   = max_length - prompt_boundary_length
    kept_answer_ids   = answer_ids[:answer_capacity]
    sequence_length   = prompt_boundary_length + len(kept_answer_ids)
which equals max_length only when answer_length > answer_capacity (the
truncation case). All other rows have seq_lens < max_length.

The script rewrites each dataset's five .bin files in place, keeping only
the rows where seq_lens < max_length, and updates the per-dataset
metadata.json. Run merge_pretokenized_manifests.py afterward to refresh
the top-level manifest.json.

Usage:
    python drop_truncated_rows.py --root <pretokenized_root>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ARRAY_NAMES = ("input_ids", "prompt_mask", "attention_mask", "seq_lens", "prompt_lens")
DTYPES = {
    "input_ids": np.int32,
    "prompt_mask": np.uint8,
    "attention_mask": np.uint8,
    "seq_lens": np.uint16,
    "prompt_lens": np.uint16,
}


def filter_dataset(shard_dir: str) -> dict:
    meta_path = os.path.join(shard_dir, "metadata.json")
    with open(meta_path, "r") as h:
        meta = json.load(h)

    n = int(meta["kept"])
    L = int(meta["max_length"])
    files = meta["files"]

    seq_lens_path = os.path.join(shard_dir, files["seq_lens"])
    seq_lens = np.fromfile(seq_lens_path, dtype=DTYPES["seq_lens"])
    assert seq_lens.shape == (n,), (seq_lens.shape, n)

    keep_mask = seq_lens < L
    n_keep = int(keep_mask.sum())
    n_drop = n - n_keep
    print(f"  rows={n:,} keep={n_keep:,} drop={n_drop:,} ({100*n_drop/n:.2f}% truncated)")

    if n_drop == 0:
        print("  no truncated rows; skipping rewrite")
        return {"kept_before": n, "kept_after": n_keep, "dropped": 0}

    keep_idx = np.flatnonzero(keep_mask)

    for arr_name in ARRAY_NAMES:
        rel = files[arr_name]
        path = os.path.join(shard_dir, rel)
        dtype = DTYPES[arr_name]
        # 2-D arrays have shape (n, L); 1-D arrays have shape (n,)
        flat = np.fromfile(path, dtype=dtype)
        if flat.size == n * L:
            arr = flat.reshape(n, L)
            kept = arr[keep_idx]
        elif flat.size == n:
            kept = flat[keep_idx]
        else:
            raise RuntimeError(
                f"{path}: size {flat.size} not compatible with n={n}, L={L}"
            )
        tmp = path + ".tmp"
        kept.astype(dtype, copy=False).tofile(tmp)
        os.replace(tmp, path)
        print(f"    rewrote {rel} ({kept.shape})")

    # Update metadata
    meta["kept"] = n_keep
    meta["dropped_total"] = int(meta.get("dropped_total", 0)) + n_drop
    meta.setdefault("drops", {})
    meta["drops"]["truncated"] = int(meta["drops"].get("truncated", 0)) + n_drop
    meta["answer_truncated"] = 0  # no truncated rows remain
    meta["answer_truncation_rate"] = 0.0
    # shapes
    meta["shapes"] = {
        "input_ids": [n_keep, L],
        "prompt_mask": [n_keep, L],
        "attention_mask": [n_keep, L],
        "seq_lens": [n_keep],
        "prompt_lens": [n_keep],
    }
    # retention_rate is over `seen`, not affected by *additional* drops here,
    # but we recompute it conservatively:
    seen = int(meta.get("seen", n))
    if seen > 0:
        meta["retention_rate"] = n_keep / seen

    tmp_meta = meta_path + ".tmp"
    with open(tmp_meta, "w") as h:
        json.dump(meta, h, indent=2, sort_keys=True)
    os.replace(tmp_meta, meta_path)
    print(f"    updated metadata.json (kept {n_keep:,})")

    return {"kept_before": n, "kept_after": n_keep, "dropped": n_drop}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="If given, filter only these subdirs; else all subdirs containing metadata.json.",
    )
    args = p.parse_args()
    root = os.path.abspath(os.path.expanduser(args.root))
    if not os.path.isdir(root):
        raise FileNotFoundError(root)

    if args.datasets:
        names = list(args.datasets)
    else:
        names = sorted(
            entry
            for entry in os.listdir(root)
            if os.path.isfile(os.path.join(root, entry, "metadata.json"))
        )

    if not names:
        print(f"No datasets found under {root}", file=sys.stderr)
        return 1

    summary = {}
    for name in names:
        print(f"[{name}]")
        summary[name] = filter_dataset(os.path.join(root, name))

    print("\n=== summary ===")
    total_before = 0
    total_after = 0
    for name, s in summary.items():
        total_before += s["kept_before"]
        total_after += s["kept_after"]
        print(f"  {name:38s}  before={s['kept_before']:>10,}  after={s['kept_after']:>10,}  dropped={s['dropped']:>8,}")
    print(f"  {'total':38s}  before={total_before:>10,}  after={total_after:>10,}  dropped={total_before-total_after:>8,}")
    print("\nNow re-run merge_pretokenized_manifests.py to refresh the top-level manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
