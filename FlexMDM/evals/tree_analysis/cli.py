"""CLI for any-order metric computation over .pt traces.

Reads ``.pt`` records produced by ``evals.humaneval_compare.generate``,
computes the CBC / RUB / OBW metrics defined in ``evals/metrics.md`` per
sample, and writes:

  - ``<output_root>/anyorder/per_sample/<gen>__<task>__<model>__<alg>__sample_kk.json``
  - ``<output_root>/anyorder/summary.json``  (keyed by ``gen|model|alg``)

Usage:

    python -m evals.tree_analysis.cli \\
        --output-root /path/to/eval_run \\
        --datasets humaneval humaneval_plus mbpp mbpp_plus \\
        --models flexmdm dreamcoder \\
        --algs entropy origin \\
        --tokenizer Dream-org/Dream-Coder-v0-Base-7B \\
        --workers 8
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, List, Optional, Sequence, Tuple

from evals.humaneval_compare.common import (
    ALL_DATASETS,
    Task,
    check_records_exist,
    gen_dataset_for,
    load_saved_tasks,
    model_alg_pairs,
    record_path,
    tasks_path,
)
from evals.tree_analysis.compute_metrics import (
    aggregate_per_sample_jsons,
    analyze_sample_to_disk,
    summary_json_path,
)


def _load_tokenizer(tokenizer_path: str) -> Any:
    from transformers import AutoTokenizer  # local import — heavy
    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def _resolve_variants(flag_value: bool) -> Sequence[bool]:
    """The CLI flag is a single bool; we always compute *both* variants for
    completeness so the summary JSON has them, but the bool toggles which
    variant flows into the printed-table aggregate.
    """
    # Always compute both variants on disk; use ``flag_value`` only for
    # the summary picker downstream.
    return (True, False)


def _datasets_to_units(args: argparse.Namespace) -> List[Tuple[str, Task]]:
    """Build a deduplicated list of (gen_dataset, task). HE and HE+ collapse
    to a single set of generation traces, so we only iterate once per
    underlying gen dataset.
    """
    seen_gen: dict[str, list[Task]] = {}
    for dataset in args.datasets:
        gd = gen_dataset_for(dataset)
        if gd in seen_gen:
            continue
        tasks_file = tasks_path(args.output_root, gd)
        if not os.path.isfile(tasks_file):
            raise FileNotFoundError(
                f"Missing tasks.json at {tasks_file}; did generate.py "
                f"run for {gd}?"
            )
        seen_gen[gd] = load_saved_tasks(tasks_file)

    units: list[tuple[str, Task]] = []
    for gd, tasks in seen_gen.items():
        if args.limit is not None:
            tasks = tasks[: args.limit]
        for task in tasks:
            units.append((gd, task))
    return units


def _worker_init(tokenizer_path: str) -> None:
    """ProcessPool worker initialiser: load the tokenizer once per worker."""
    global _WORKER_TOKENIZER  # noqa: PLW0603 — module-level cache is intentional
    _WORKER_TOKENIZER = _load_tokenizer(tokenizer_path)


_WORKER_TOKENIZER: Any = None


def _worker_analyze(
    output_root: str,
    gen_dataset: str,
    task_dict: dict,
    model: str,
    alg: str,
    sample_k: int,
    include_variants: Tuple[bool, ...],
) -> Optional[dict]:
    from evals.humaneval_compare.common import _task_from_dict  # noqa: PLC0415
    task = _task_from_dict(task_dict)
    return analyze_sample_to_disk(
        output_root=output_root,
        gen_dataset=gen_dataset,
        task=task,
        model=model,
        alg=alg,
        sample_k=sample_k,
        tokenizer=_WORKER_TOKENIZER,
        include_reference_nodes_variants=include_variants,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-root", required=True,
        help="Generation output root (passed to generate.py).",
    )
    p.add_argument(
        "--datasets", nargs="+",
        default=["humaneval", "humaneval_plus", "mbpp", "mbpp_plus"],
        choices=list(ALL_DATASETS),
    )
    p.add_argument(
        "--models", nargs="+",
        default=["flexmdm", "dreamcoder"],
    )
    p.add_argument(
        "--algs", nargs="+",
        default=["entropy", "origin"],
        help="Algs to analyse, crossed with every --models entry. The "
             "released runs pair per model instead — prefer --model-algs.",
    )
    p.add_argument(
        "--model-algs", nargs="+", default=None,
        metavar="MODEL=ALG[,ALG...]",
        help="Per-model alg pairing, e.g. 'flexmdm=top_k dreamcoder=entropy' "
             "(the released configuration). Overrides --models/--algs.",
    )
    p.add_argument(
        "--allow-missing", action="store_true", default=False,
        help="Proceed even if a requested (model, alg) has no records on "
             "disk. Without this flag such a mismatch is a hard error.",
    )
    p.add_argument(
        "--n-samples", type=int, default=32,
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="If set, only analyse the first N tasks per gen dataset.",
    )
    p.add_argument(
        "--workers", type=int, default=8,
    )
    p.add_argument(
        "--tokenizer", required=True,
        help="Tokenizer path/HF id used to decode .pt records. "
             "Must match the tokenizer that produced the traces.",
    )
    p.add_argument(
        "--include-reference-nodes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the reference-aware variant for the printed summary "
             "(both variants are always written to summary.json).",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    # Pre-fetch task lists for each gen dataset (also validates their existence).
    units = _datasets_to_units(args)
    pairs = model_alg_pairs(args.models, args.algs, args.model_algs)
    print(
        f"[info] {len(units)} (gen, task) units; model/alg pairs={pairs} "
        f"n_samples={args.n_samples} workers={args.workers}",
        flush=True,
    )

    # Fail loudly when a requested (model, alg) was never generated.
    gen_datasets = sorted({gen_dataset_for(d) for d in args.datasets})
    problems = check_records_exist(args.output_root, gen_datasets, pairs)
    if problems:
        for msg in problems:
            print(f"[{'warn' if args.allow_missing else 'error'}] {msg}",
                  file=sys.stderr, flush=True)
        if not args.allow_missing:
            raise SystemExit(
                "requested (model, alg) combinations have no records on disk "
                "(see above). Fix --model-algs / --models / --algs, or pass "
                "--allow-missing."
            )

    include_variants = _resolve_variants(bool(args.include_reference_nodes))

    # Build the full work list ahead of time so we can show progress.
    from evals.humaneval_compare.common import _task_to_dict  # noqa: PLC0415
    work: List[Tuple[str, dict, str, str, int]] = []
    for gd, task in units:
        td = _task_to_dict(task)
        for model, alg in pairs:
            for k in range(args.n_samples):
                work.append((gd, td, model, alg, k))

    print(f"[info] {len(work)} samples to analyse", flush=True)

    per_sample_results: List[dict] = []
    if args.workers <= 1:
        # Inline path — useful for debugging and for tests.
        tokenizer = _load_tokenizer(args.tokenizer)
        for i, (gd, td, model, alg, k) in enumerate(work, 1):
            from evals.humaneval_compare.common import _task_from_dict  # noqa
            row = analyze_sample_to_disk(
                output_root=args.output_root,
                gen_dataset=gd,
                task=_task_from_dict(td),
                model=model,
                alg=alg,
                sample_k=k,
                tokenizer=tokenizer,
                include_reference_nodes_variants=include_variants,
            )
            if row is not None:
                per_sample_results.append(row)
            if i % 50 == 0:
                print(f"[{i}/{len(work)}]", flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=mp.get_context("spawn"),
            initializer=_worker_init,
            initargs=(args.tokenizer,),
        ) as pool:
            futures = {
                pool.submit(
                    _worker_analyze,
                    args.output_root,
                    gd,
                    td,
                    model,
                    alg,
                    k,
                    tuple(include_variants),
                ): (gd, td["task_id"], model, alg, k)
                for (gd, td, model, alg, k) in work
            }
            for i, fut in enumerate(as_completed(futures), 1):
                gd, tid, model, alg, k = futures[fut]
                try:
                    row = fut.result()
                except Exception as exc:
                    print(
                        f"[err] {gd}/{tid}/{model}/{alg}/sample_{k}: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    continue
                if row is not None:
                    per_sample_results.append(row)
                if i % 50 == 0:
                    print(f"[{i}/{len(futures)}]", flush=True)

    summary = aggregate_per_sample_jsons(per_sample_results)
    out_path = summary_json_path(args.output_root)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n=========== any-order summary ===========")
    print(f"  (all 3 variants below; summary.json holds the full data)")
    headers = [
        "gen|model|alg|variant",
        "n_samples", "n_parsed",
        "ovrCBC", "ovrRUB", "ovrOBW",
        "splCBC", "splRUB", "splOBW",
    ]
    print("  ".join(f"{h:<32}" if i == 0 else f"{h:<10}"
                    for i, h in enumerate(headers)))
    for key, row in sorted(summary.items()):
        for variant_key in ("reference_aware", "code_only", "entry_only"):
            var = row.get(variant_key, {})
            cells = [
                f"{key}|{variant_key}",
                str(row["n_samples"]),
                str(row["n_parsed"]),
                _fmt(var.get("overall_cbc")),
                _fmt(var.get("overall_rub")),
                _fmt(var.get("overall_obw")),
                _fmt(var.get("split_only_cbc")),
                _fmt(var.get("split_only_rub")),
                _fmt(var.get("split_only_obw")),
            ]
            print("  ".join(f"{c:<32}" if i == 0 else f"{c:<10}"
                            for i, c in enumerate(cells)))
    print(f"\n[done] wrote {out_path}")
    return 0


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{v:.3f}"


if __name__ == "__main__":
    sys.exit(main())
