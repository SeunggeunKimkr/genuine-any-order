"""Extraction-robust pass@k: a sample is correct iff ANY of the 4 extraction
modes (prompt_tail_sanitize, prompt_cleaned_sanitize, prompt_tail_raw,
cleaned_raw) yields a passing program. Reads
`<output_root>/passk/per_sample.jsonl` produced by
`evals.humaneval_compare.passk` and emits an equivalent summary.

Why: extracting the program from raw model output is a preprocessing choice,
and the sanitize/raw conventions fail on disjoint parsing quirks (some
completions survive raw extraction but not sanitize, others the reverse).
Grading over the union of the four modes makes the score invariant to that
choice while remaining honest about *executable correctness* — the extracted
program still has to pass the full test suite. Applied identically to every
model.

Results are grouped per ``gen_dataset|model|alg`` (per_sample.jsonl may hold
several datasets and model/alg combinations in one output root — pooling them
would blend HumanEval with MBPP or mix models). Writes
``<output_root>/passk/robust_summary.json`` by default.

Usage:
    python -m evals.humaneval_compare.robust_passk \
        --output-root /path/to/run \
        [--variants base plus] [--ks 1 2 4 8 16] [--n-samples 16]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable


DEFAULT_KS = (1, 2, 4, 8, 16)


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _row_gen_dataset(r: dict) -> str:
    """gen_dataset of a per-sample row; falls back to the task_id prefix for
    per_sample.jsonl files written before the field existed."""
    gd = r.get("gen_dataset")
    if gd:
        return str(gd)
    prefix = str(r.get("task_id", "")).split("/", 1)[0].lower()
    return {"humaneval": "humaneval", "mbpp": "mbpp"}.get(prefix, prefix or "unknown")


def compute(
    per_sample_path: Path,
    *,
    variants: Iterable[str] = ("base", "plus"),
    ks: Iterable[int] = DEFAULT_KS,
    n_samples: int = 16,
) -> dict:
    """Aggregate per (gen_dataset|model|alg, variant) — never pool datasets,
    models, or algs together."""
    # rows[(gd, model, alg, variant)] -> per-task pass sets / per-mode counts
    any_pass: dict[tuple, dict] = defaultdict(lambda: defaultdict(set))
    per_mode_correct: dict[tuple, dict] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    tasks: dict[tuple, set] = defaultdict(set)
    modes: dict[tuple, set] = defaultdict(set)
    samples_seen: dict[tuple, set] = defaultdict(set)
    wanted = set(variants)
    with per_sample_path.open() as f:
        for line in f:
            r = json.loads(line)
            if r["variant"] not in wanted:
                continue
            key = (_row_gen_dataset(r), r["model"], r["alg"], r["variant"])
            tasks[key].add(r["task_id"])
            modes[key].add(r["mode"])
            samples_seen[key].add(r["sample_k"])
            if r["passed"]:
                any_pass[key][r["task_id"]].add(r["sample_k"])
                per_mode_correct[key][r["mode"]][r["task_id"]] += 1

    out: dict = {}
    for key in sorted(tasks):
        gen_dataset, model, alg, variant = key
        n_tasks = len(tasks[key])
        n_seen = len(samples_seen[key])
        if n_seen < n_samples:
            print(f"[warn] {gen_dataset}|{model}|{alg}|{variant}: only "
                  f"{n_seen} distinct sample_k on disk but "
                  f"--n-samples={n_samples}; missing samples count as failures.")
        # any-of-4 pass@k (iterate tasks in sorted order so the float sum is
        # deterministic across runs / hash seeds)
        robust = {k: 0.0 for k in ks}
        for tid in sorted(tasks[key]):
            c = len(any_pass[key].get(tid, ()))
            for k in ks:
                robust[k] += pass_at_k(n_samples, c, k)
        robust = {k: v / n_tasks for k, v in robust.items()}
        # per-mode pass@k
        per_mode_passk = {}
        for mode in sorted(modes[key]):
            mp = {k: 0.0 for k in ks}
            for tid in sorted(tasks[key]):
                c = per_mode_correct[key][mode].get(tid, 0)
                for k in ks:
                    mp[k] += pass_at_k(n_samples, c, k)
            per_mode_passk[mode] = {k: v / n_tasks for k, v in mp.items()}
        out.setdefault(f"{gen_dataset}|{model}|{alg}", {})[variant] = {
            "n_tasks": n_tasks,
            "n_samples": n_samples,
            "n_samples_seen": n_seen,
            "robust": robust,
            "per_mode": per_mode_passk,
        }
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--variants", nargs="+", default=["base", "plus"])
    p.add_argument("--ks", nargs="+", type=int, default=list(DEFAULT_KS))
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--per-sample", type=Path, default=None,
                   help="per_sample.jsonl to aggregate. Default: "
                        "<output-root>/passk/per_sample.jsonl. Point at "
                        "another passk.py run's per_sample.jsonl (e.g. one "
                        "written via --out) to aggregate that run instead.")
    p.add_argument("--out-json", type=Path, default=None,
                   help="Where to write the summary JSON. Default: "
                        "robust_summary.json next to the per-sample file.")
    args = p.parse_args(argv)

    per_sample = args.per_sample or (args.output_root / "passk" / "per_sample.jsonl")
    if not per_sample.exists():
        raise SystemExit(f"missing {per_sample}")

    summary = compute(
        per_sample,
        variants=args.variants,
        ks=args.ks,
        n_samples=args.n_samples,
    )
    if not summary:
        raise SystemExit(f"no rows matched variants={args.variants} "
                         f"in {per_sample}")

    print(f"# extraction-robust pass@k for {args.output_root}")
    for combo, per_variant in summary.items():
        for variant, data in per_variant.items():
            print(f"\n## {combo}  variant={variant}  "
                  f"(n_tasks={data['n_tasks']}, n_samples={data['n_samples']})")
            print(f"{'mode':30s} | " +
                  " ".join(f"pass@{k:<2d}" for k in args.ks))
            for mode, vals in data["per_mode"].items():
                cells = " ".join(f"{vals[k]*100:6.2f}" for k in args.ks)
                print(f"{mode:30s} | {cells}")
            cells = " ".join(f"{data['robust'][k]*100:6.2f}" for k in args.ks)
            print(f"{'ROBUST (any-of-4)':30s} | {cells}")

    out_json = args.out_json or (per_sample.parent / "robust_summary.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\n[done] wrote {out_json}")


if __name__ == "__main__":
    main()
