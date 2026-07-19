#!/usr/bin/env python
"""Assemble an inference-only, self-contained Hugging Face checkpoint from a
FlexMDM training checkpoint.

A training checkpoint (``global_step_N/``) contains ~48 GB: sharded model
weights, ``flexmdm_extras.pt`` (the FlexMDM insertion head + AdaLN time
conditioning), and training-only state (optimizer, RNG, step counter). For
release we keep only what inference needs (~17 GB) and drop the rest.

This script is READ-ONLY on ``--src``; it only writes to ``--dst``.

What it does:
  1. Copy every file from ``--src`` EXCEPT the training-only blocklist
     (``optimizer_state.pt``, ``training_state.pt``, ``rng_state/``). This keeps
     the sharded ``model-*.safetensors`` weights, ``flexmdm_extras.pt``, the
     tokenizer files, ``config.json``, and ``generation_config.json``.
  2. Optionally vendor the Dream backbone's remote-code modeling files
     (``configuration_dream.py``, ``modeling_dream.py``, ``tokenization_dream.py``,
     ``generation_utils.py``) from ``--modeling-from`` and rewrite the ``auto_map``
     entries in ``config.json`` / ``tokenizer_config.json`` to point at the local
     copies, so the released checkpoint loads offline without the base repo.

The released checkpoint is loaded with this repo's loader, NOT a bare
``AutoModel`` (which would return only the Dream backbone):

    from flexmdm.utils import load_model_and_tokenizer
    model, tok = load_model_and_tokenizer(checkpoint_dir=<dir>, max_length=768)

Usage:
    python scripts/prepare_release_checkpoint.py \\
        --src  /path/to/checkpoints/global_step_49500 \\
        --dst  /path/to/release/flexmdm \\
        --modeling-from /path/to/Dream-Coder-v0-Base-7B   # optional but recommended
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

# Training-only artifacts that inference does not need.
BLOCKLIST = {"optimizer_state.pt", "training_state.pt", "rng_state"}

# Dream backbone remote-code files to vendor for a self-contained checkpoint.
MODELING_FILES = (
    "configuration_dream.py",
    "modeling_dream.py",
    "tokenization_dream.py",
    "generation_utils.py",
)


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024.0


def _dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total


def _strip_repo_prefix(value):
    """'Repo--module.Class' -> 'module.Class'; leave already-local values as-is."""
    if isinstance(value, str) and "--" in value:
        return value.split("--", 1)[1]
    if isinstance(value, list):
        return [_strip_repo_prefix(v) for v in value]
    return value


def _rewrite_auto_map(json_path: str) -> bool:
    if not os.path.isfile(json_path):
        return False
    with open(json_path) as fh:
        cfg = json.load(fh)
    am = cfg.get("auto_map")
    if not am:
        return False
    new_am = {k: _strip_repo_prefix(v) for k, v in am.items()}
    if new_am == am:
        return False
    cfg["auto_map"] = new_am
    with open(json_path, "w") as fh:
        json.dump(cfg, fh, indent=2)
        fh.write("\n")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="training checkpoint dir (global_step_N)")
    ap.add_argument("--dst", required=True, help="output dir for the release checkpoint")
    ap.add_argument("--modeling-from", default=None,
                    help="dir with configuration_dream.py/modeling_dream.py/etc. to vendor")
    ap.add_argument("--keep-optimizer", action="store_true",
                    help="also copy optimizer_state.pt (NOT recommended for release)")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow writing into a non-empty --dst")
    args = ap.parse_args()

    src, dst = os.path.abspath(args.src), os.path.abspath(args.dst)
    if not os.path.isdir(src):
        ap.error(f"--src not found: {src}")
    if not os.path.isfile(os.path.join(src, "flexmdm_extras.pt")):
        ap.error(f"--src has no flexmdm_extras.pt (is this a FlexMDM checkpoint?): {src}")
    if os.path.abspath(src) == os.path.abspath(dst):
        ap.error("--src and --dst must differ")
    if os.path.isdir(dst) and os.listdir(dst) and not args.overwrite:
        ap.error(f"--dst exists and is non-empty (use --overwrite): {dst}")

    blocklist = set(BLOCKLIST)
    if args.keep_optimizer:
        blocklist.discard("optimizer_state.pt")

    os.makedirs(dst, exist_ok=True)
    copied, skipped = [], []
    for name in sorted(os.listdir(src)):
        sp = os.path.join(src, name)
        if name in blocklist:
            skipped.append(name)
            continue
        dp = os.path.join(dst, name)
        if os.path.isdir(sp):
            shutil.copytree(sp, dp, dirs_exist_ok=True)
        else:
            shutil.copy2(sp, dp)
        copied.append(name)

    vendored = []
    if args.modeling_from:
        mf = os.path.abspath(args.modeling_from)
        for name in MODELING_FILES:
            sp = os.path.join(mf, name)
            if os.path.isfile(sp):
                shutil.copy2(sp, os.path.join(dst, name))
                vendored.append(name)
        # Point auto_map at the local files instead of the base repo.
        for jname in ("config.json", "tokenizer_config.json"):
            if _rewrite_auto_map(os.path.join(dst, jname)):
                print(f"[prep] rewrote auto_map -> local in {jname}")

    print("\n=== release checkpoint prepared ===")
    print(f"src: {src}")
    print(f"dst: {dst}")
    print(f"copied ({len(copied)}): {', '.join(copied)}")
    print(f"skipped ({len(skipped)}): {', '.join(skipped) or '(none)'}")
    if vendored:
        print(f"vendored modeling files: {', '.join(vendored)}")
    elif args.modeling_from:
        print("WARNING: --modeling-from set but no modeling files found there.")
    else:
        print("NOTE: no --modeling-from; config.json auto_map still references the "
              "base repo (loads via trust_remote_code from the Hub).")
    print(f"total size: {_human(_dir_size(dst))}")
    # Basic sanity: are the safetensors shards all present?
    idx = os.path.join(dst, "model.safetensors.index.json")
    if os.path.isfile(idx):
        with open(idx) as fh:
            shards = set(json.load(fh).get("weight_map", {}).values())
        missing = [s for s in shards if not os.path.isfile(os.path.join(dst, s))]
        print(f"safetensors shards: {len(shards)} referenced, "
              f"{'all present' if not missing else 'MISSING: ' + ', '.join(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
