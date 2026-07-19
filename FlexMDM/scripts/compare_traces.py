"""Bit-level comparison of generation trace records between two output roots.

Verifies reproducibility of the pass@k evaluation: regenerating any chunk
with the pinned environment (evals/requirements-repro.txt) must reproduce the
stored traces exactly (same seeds -> same tensors). Compares every .pt record
present under --root-b against the record at the same relative path under
--root-a.

Usage:
  python scripts/compare_traces.py \
      --root-a <original output_root>/humaneval/raw \
      --root-b <regenerated output_root>/humaneval/raw
"""

import argparse
import os
import sys

import torch


def compare_record(pa: str, pb: str) -> tuple[bool, str]:
    a = torch.load(pa, map_location="cpu", weights_only=False)
    b = torch.load(pb, map_location="cpu", weights_only=False)
    if a["prompt"] != b["prompt"]:
        return False, "prompt differs"
    if a["prompt_len"] != b["prompt_len"]:
        return False, "prompt_len differs"
    for key in ("sequences", "attention_masks", "insertion_masks"):
        ta, tb = a.get(key), b.get(key)
        if ta is None and tb is None:
            continue
        if ta is None or tb is None:
            return False, f"{key} present in only one record"
        if ta.shape != tb.shape:
            return False, f"{key} shape {tuple(ta.shape)} vs {tuple(tb.shape)}"
        if not torch.equal(ta, tb):
            # locate first diverging trajectory step for diagnosis
            step = next(
                (s for s in range(ta.shape[0]) if not torch.equal(ta[s], tb[s])),
                -1,
            )
            return False, f"{key} differs (first divergent step {step}/{ta.shape[0]})"
    return True, "identical"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root-a", required=True, help="reference raw/ dir")
    p.add_argument("--root-b", required=True, help="regenerated raw/ dir")
    args = p.parse_args()

    pairs = []
    for dirpath, _, files in os.walk(args.root_b):
        for f in sorted(files):
            if not f.endswith(".pt"):
                continue
            pb = os.path.join(dirpath, f)
            rel = os.path.relpath(pb, args.root_b)
            pa = os.path.join(args.root_a, rel)
            if os.path.isfile(pa):
                pairs.append((rel, pa, pb))

    if not pairs:
        print("no overlapping records found", flush=True)
        return 2

    n_ok = 0
    for rel, pa, pb in pairs:
        ok, why = compare_record(pa, pb)
        n_ok += ok
        print(f"{'MATCH ' if ok else 'DIFFER'} {rel}: {why}", flush=True)

    print(f"\n{n_ok}/{len(pairs)} records bit-identical", flush=True)
    return 0 if n_ok == len(pairs) else 1


if __name__ == "__main__":
    sys.exit(main())
