"""Compute pass@k for HumanEval / HumanEval+ / MBPP / MBPP+ from .pt traces.

For each (gen_dataset, task, model, alg, sample) tuple:
    1. Load the .pt record produced by ``generate.py``.
    2. Decode the post-prompt portion of the final sequence with the model's
       tokenizer to get the raw completion.
    3. Build candidate code via one of four sanitize modes:
         - prompt_tail_sanitize    (default; strict — AST sanitize on
                                    full prompt + tail)
         - prompt_cleaned_sanitize (strict on prompt + closing-fence-trimmed)
         - prompt_tail_raw         (generous — split on closing fence,
                                    no AST sanitize)
         - cleaned_raw             (most generous — just the closing-fence-
                                    trimmed text)
    4. Wrap as a runnable program for the dataset's test convention:
         - HE / HE+: test code defines `check(candidate)`, we call check(<entry>)
         - MBPP base: test_list assertions, plus test_imports prefix
         - MBPP+: test code uses `candidate` as a free var; we bind
                  `candidate = <entry_point>` before running.
    5. Run in a sandboxed subprocess with a wallclock timeout; capture pass/fail.
    6. Compute the unbiased pass@k estimator (Codex 2021) for k ∈ {1..16}
       per (task, model, alg, mode, variant); aggregate over tasks.

Usage:
    python -m evals.humaneval_compare.passk \\
        --output-root /path/to/eval_run \\
        --datasets humaneval humaneval_plus mbpp mbpp_plus \\
        --models flexmdm dreamcoder \\
        --algs entropy \\
        --modes prompt_tail_sanitize prompt_cleaned_sanitize \
                prompt_tail_raw cleaned_raw \\
        --tokenizer Dream-org/Dream-Coder-v0-Base-7B \\
        --workers 16 --timeout 30.0
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch

from evals.humaneval_compare.common import (
    GEN_HUMANEVAL,
    GEN_MBPP,
    Task,
    VARIANT_BASE,
    VARIANT_PLUS,
    check_records_exist,
    gen_dataset_for,
    load_saved_tasks,
    model_alg_pairs,
    record_path,
    tasks_path,
    variant_for,
)


# Load the local lm-eval sanitize helper directly from file to bypass the
# package-level __init__.py (which transitively imports `evaluate`/`packaging`,
# both of which can be broken in some shared envs).
import importlib.util as _ilu  # noqa: E402

_SANITIZE_PATH = os.path.join(
    os.path.dirname(__file__), "_sanitize_utils.py",
)
_spec = _ilu.spec_from_file_location("humaneval_sanitize_utils", _SANITIZE_PATH)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
sanitize = _mod.sanitize


FENCE = "`" * 3

ALL_MODES = (
    "prompt_tail_sanitize",
    "prompt_cleaned_sanitize",
    "prompt_tail_raw",
    "cleaned_raw",
)
DEFAULT_KS = (1, 2, 4, 8, 16)


# ---------------------------------------------------------------------------
# Candidate construction (4 modes, dataset-aware prefix)
# ---------------------------------------------------------------------------


def candidate_prefix_for(task: Task) -> str:
    """Code prefix joined with the completion before sanitize.

    For HumanEval the prompt is the function skeleton (runnable Python), so
    the official Dream-Coder eval feeds ``prompt + completion`` to sanitize.
    For MBPP the prompt is natural-language plus markdown opener, NOT
    runnable Python, so we feed the completion alone.
    """
    if task.dataset == GEN_HUMANEVAL:
        return task.prompt
    if task.dataset == GEN_MBPP:
        return ""
    raise AssertionError(f"Unhandled dataset {task.dataset!r}.")


def split_tail_cleaned(completion: str) -> tuple[str, str]:
    """Split a raw model completion into ``(tail, cleaned)`` where:

    - ``tail`` is the completion as-is (raw model output after the prompt).
    - ``cleaned`` is ``tail.split("\\`\\`\\`", 1)[0]`` stripped of leading /
      trailing whitespace.

    The closing-fence cut keeps post-explanation natural-language text out of
    the candidate; the strip avoids leading blank lines that would push the
    function definition past sanitize's `def` search.
    """
    cleaned = completion.split(FENCE, 1)[0].strip()
    return completion, cleaned


def _safe_sanitize(source: str, entry_point: str) -> tuple[str, Optional[str]]:
    try:
        return sanitize(source, entry_point), None
    except Exception:
        return "", traceback.format_exc(limit=1).strip()


def build_candidate_code(
    *,
    task: Task,
    completion: str,
    mode: str,
) -> tuple[str, Optional[str]]:
    """Run the 4 sanitize modes for the given (task, completion).

    Returns ``(candidate_code, sanitize_error_or_None)``. Empty candidate +
    error message ≡ sanitize raised; the candidate cannot pass tests.
    """
    if mode not in ALL_MODES:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {ALL_MODES}.")
    prefix = candidate_prefix_for(task)
    tail, cleaned = split_tail_cleaned(completion)
    if mode == "prompt_tail_sanitize":
        source = (prefix + "\n" + tail) if prefix else tail
        return _safe_sanitize(source, task.entry_point)
    if mode == "prompt_cleaned_sanitize":
        source = (prefix + "\n" + cleaned) if prefix else cleaned
        return _safe_sanitize(source, task.entry_point)
    if mode == "prompt_tail_raw":
        # Generous mode: prefix + tail.split(FENCE, 1)[0] (no .strip()).
        cut = tail.split(FENCE, 1)[0]
        return ((prefix + "\n" + cut) if prefix else cut), None
    if mode == "cleaned_raw":
        return cleaned, None
    raise AssertionError(f"Unhandled mode {mode!r}.")  # pragma: no cover


# ---------------------------------------------------------------------------
# Program assembly (per dataset / variant)
# ---------------------------------------------------------------------------


def build_program(*, task: Task, candidate_code: str, variant: str) -> str:
    """Assemble a runnable program: candidate + test code + invocation."""
    test_code = task.test_for(variant)
    if task.dataset == GEN_HUMANEVAL:
        # HE / HE+: the test field defines `def check(candidate): ...`.
        return (
            f"{candidate_code}\n\n"
            f"{test_code}\n\n"
            f"check({task.entry_point})\n"
        )
    if task.dataset == GEN_MBPP:
        if variant == VARIANT_BASE:
            # MBPP base: test_list assertions, plus task-specific imports.
            imports = "\n".join(task.test_imports)
            return (
                (imports + "\n\n" if imports else "")
                + f"{candidate_code}\n\n"
                + f"{test_code}\n"
            )
        # MBPP+: top-level test references `candidate` as a free variable.
        return (
            f"{candidate_code}\n\n"
            f"candidate = {task.entry_point}\n"
            f"{test_code}\n"
        )
    raise AssertionError(f"Unhandled dataset {task.dataset!r}.")


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------


_RUNNER_PROLOGUE = (
    "import math\n"
    "import re\n"
    "import sys\n"
    "import time\n"
    "import itertools\n"
    "import functools\n"
    "import collections\n"
    "from typing import *\n"
    "\n"
)


def run_one(
    program: str,
    *,
    timeout_s: float,
    python_bin: Optional[str] = None,
) -> tuple[bool, str, str]:
    """Execute ``program`` in a sandboxed subprocess.

    The program is piped via stdin (``python -``) rather than passed as
    ``-c <program>`` because HE+ test code can exceed the OS argv limit
    (~128 KB on Linux); piping has no such limit.

    Returns ``(passed, error_type, error_msg)``. ``error_type`` is one of
    ``"passed"``, ``"timeout"``, ``"syntax_error"``, ``"assertion_error"``,
    ``"name_error"``, ``"import_error"``, ``"runtime_error"``,
    ``"subprocess_error"``.
    """
    binary = python_bin or sys.executable
    full = _RUNNER_PROLOGUE + program
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    try:
        proc = subprocess.run(
            [binary, "-I", "-"],
            input=full,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout", f"Timed out after {timeout_s}s"
    except Exception as exc:
        return False, "subprocess_error", repr(exc)

    if proc.returncode == 0:
        return True, "passed", ""
    err = (proc.stderr or proc.stdout or "").strip()
    if "SyntaxError" in err:
        kind = "syntax_error"
    elif "AssertionError" in err:
        kind = "assertion_error"
    elif "NameError" in err:
        kind = "name_error"
    elif "ImportError" in err or "ModuleNotFoundError" in err:
        kind = "import_error"
    else:
        kind = "runtime_error"
    return False, kind, err[-1200:]


# ---------------------------------------------------------------------------
# pass@k estimator (Chen et al., Codex 2021)
# ---------------------------------------------------------------------------


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from ``n`` total samples and ``c`` correct."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if n < 1:
        raise ValueError(f"n must be >=1, got {n}")
    if c < 0 or c > n:
        raise ValueError(f"c must be in [0, n]; got n={n}, c={c}")
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


# ---------------------------------------------------------------------------
# Decoding completions from .pt records
# ---------------------------------------------------------------------------


def decode_completion(record: dict, tokenizer: Any) -> str:
    """Decode the post-prompt portion of the FINAL sequence.

    The record stores per-step state; pass@k only needs the final state at
    ``sequences[-1]``. The "answer" is the slice
    ``sequences[-1, prompt_len : real_len]``, where ``real_len`` is read from
    the final attention mask.
    """
    seqs = record["sequences"]
    am = record["attention_masks"]
    pl = int(record["prompt_len"])
    final_seq = seqs[-1]
    final_am = am[-1]
    real_len = int(final_am.sum().item())
    if real_len <= pl:
        return ""
    body_ids = final_seq[pl:real_len].tolist()
    return tokenizer.decode(body_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Per-unit driver
# ---------------------------------------------------------------------------


@dataclass
class SampleResult:
    task_id: str
    gen_dataset: str
    model: str
    alg: str
    sample_k: int
    mode: str
    variant: str
    passed: bool
    error_type: str
    error: str


@dataclass
class UnitResult:
    """Pass@k for one (gen_dataset, task, model, alg, mode, variant) unit."""

    task_id: str
    gen_dataset: str
    model: str
    alg: str
    mode: str
    variant: str
    n_samples: int
    n_correct: int
    pass_at: dict[int, float]
    sample_pass: list[bool]
    error_counts: dict[str, int]


def evaluate_unit(
    *,
    output_root: str,
    gen_dataset: str,
    task: Task,
    model: str,
    alg: str,
    n_samples: int,
    modes: Sequence[str],
    variants: Sequence[str],
    timeout_s: float,
    tokenizer_path: str,
    ks: Sequence[int] = DEFAULT_KS,
) -> list[UnitResult]:
    """Run all ``n_samples`` samples and return one UnitResult per
    (mode, variant) combination.

    Each sample is decoded once and then evaluated under every mode. This
    avoids re-loading .pt files and re-instantiating the tokenizer per mode.
    """
    from transformers import AutoTokenizer  # local import — not a worker dep
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True
    )

    sample_results: dict[tuple[str, str], list[SampleResult]] = defaultdict(list)
    for k in range(n_samples):
        rp = record_path(output_root, gen_dataset, task.task_id, model, alg, k)
        if not os.path.isfile(rp):
            for mode in modes:
                for variant in variants:
                    sample_results[(mode, variant)].append(SampleResult(
                        task_id=task.task_id, gen_dataset=gen_dataset, model=model, alg=alg,
                        sample_k=k, mode=mode, variant=variant,
                        passed=False, error_type="missing",
                        error=f"missing record {rp}",
                    ))
            continue

        rec = torch.load(rp, map_location="cpu", weights_only=False)
        completion = decode_completion(rec, tokenizer)

        for mode in modes:
            candidate, sanitize_err = build_candidate_code(
                task=task, completion=completion, mode=mode,
            )
            if sanitize_err:
                for variant in variants:
                    sample_results[(mode, variant)].append(SampleResult(
                        task_id=task.task_id, gen_dataset=gen_dataset, model=model, alg=alg,
                        sample_k=k, mode=mode, variant=variant,
                        passed=False, error_type="sanitize_error",
                        error=sanitize_err,
                    ))
                continue
            for variant in variants:
                program = build_program(
                    task=task, candidate_code=candidate, variant=variant,
                )
                passed, err_type, err = run_one(program, timeout_s=timeout_s)
                sample_results[(mode, variant)].append(SampleResult(
                    task_id=task.task_id, gen_dataset=gen_dataset, model=model, alg=alg,
                    sample_k=k, mode=mode, variant=variant,
                    passed=passed, error_type=err_type, error=err,
                ))

    out: list[UnitResult] = []
    for (mode, variant), rows in sample_results.items():
        passes = [r.passed for r in rows]
        ec: dict[str, int] = defaultdict(int)
        for r in rows:
            if not r.passed:
                ec[r.error_type] += 1
        n = len(passes)
        c = sum(passes)
        pa = {k: pass_at_k(n, c, k) for k in ks if k <= n}
        out.append(UnitResult(
            task_id=task.task_id, gen_dataset=gen_dataset, model=model, alg=alg,
            mode=mode, variant=variant,
            n_samples=n, n_correct=c,
            pass_at=pa, sample_pass=passes, error_counts=dict(ec),
        ))
    return out


# ---------------------------------------------------------------------------
# CLI orchestrator
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", required=True,
                   help="Generation output root (passed to generate.py).")
    p.add_argument("--datasets", nargs="+",
                   default=["humaneval", "humaneval_plus", "mbpp", "mbpp_plus"])
    p.add_argument("--models", nargs="+", default=["flexmdm", "dreamcoder"])
    p.add_argument("--algs", nargs="+", default=["entropy"],
                   help="Algs to score, crossed with every --models entry. "
                        "The released runs pair per model instead — prefer "
                        "--model-algs.")
    p.add_argument("--model-algs", nargs="+", default=None,
                   metavar="MODEL=ALG[,ALG...]",
                   help="Per-model alg pairing, e.g. "
                        "'flexmdm=top_k dreamcoder=entropy' (the released "
                        "configuration). Overrides --models/--algs.")
    p.add_argument("--allow-missing", action="store_true", default=False,
                   help="Score even if a requested (model, alg) has no records "
                        "on disk (missing samples count as failures). Without "
                        "this flag such a mismatch is a hard error.")
    p.add_argument("--modes", nargs="+", default=list(ALL_MODES),
                   choices=list(ALL_MODES))
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--ks", nargs="+", type=int, default=list(DEFAULT_KS))
    p.add_argument("--timeout", type=float, default=30.0,
                   help="Per-sample wall timeout in seconds. 30.0 is the "
                        "published setting (the paper's tables; machine-"
                        "robust). Shorter timeouts leave base rows unchanged "
                        "but undercount the plus variants and make them "
                        "CPU-speed-sensitive (see docs/REPRODUCE.md).")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--tokenizer", required=True,
                   help="Tokenizer path/HF id used to decode .pt records. "
                        "Should match the tokenizer that produced the traces "
                        "(typically the model's tokenizer).")
    p.add_argument("--limit", type=int, default=None,
                   help="If set, only evaluate the first N tasks per dataset "
                        "(sanity-check shortcut).")
    p.add_argument("--out", default=None,
                   help="Where to write pass@k results. Default: "
                        "<output-root>/passk/")
    return p.parse_args(argv)


def _datasets_to_units(args: argparse.Namespace) -> list[tuple[str, str, Task, str, str]]:
    """Build the (gen_dataset, variant, task, model, alg) work list.

    A pair (humaneval, humaneval_plus) generates only one set of .pt traces
    under output_root/humaneval/, but we evaluate it twice — once with
    test_base (variant=base), once with test_plus (variant=plus). Same for
    MBPP / MBPP+. So we group by (gen_dataset, model, alg) and emit two
    variant rows per requested CLI dataset pair.
    """
    units: list[tuple[str, str, Task, str, str]] = []
    seen_gen: dict[str, list[Task]] = {}
    pairs = model_alg_pairs(args.models, args.algs, args.model_algs)
    for dataset in args.datasets:
        gd = gen_dataset_for(dataset)
        variant = variant_for(dataset)
        if gd not in seen_gen:
            tasks_file = tasks_path(args.output_root, gd)
            if not os.path.isfile(tasks_file):
                raise FileNotFoundError(
                    f"Missing tasks.json at {tasks_file}; did generate.py "
                    f"run for {gd}?"
                )
            seen_gen[gd] = load_saved_tasks(tasks_file)
        tasks = seen_gen[gd]
        if args.limit is not None:
            tasks = tasks[: args.limit]
        for task in tasks:
            for model, alg in pairs:
                units.append((gd, variant, task, model, alg))
    return units


def _aggregate(
    rows: Sequence[UnitResult], *, ks: Sequence[int],
) -> dict[str, Any]:
    """Aggregate per-(gen_dataset, model, alg, mode, variant) means across
    tasks. The gen_dataset key component keeps HumanEval and MBPP separate
    when one output root holds both (as run_full_eval.sh produces)."""
    agg: dict[tuple[str, str, str, str, str], list[UnitResult]] = defaultdict(list)
    for r in rows:
        agg[(r.gen_dataset, r.model, r.alg, r.mode, r.variant)].append(r)
    summary: dict[str, Any] = {}
    for (gen_dataset, model, alg, mode, variant), unit_rs in agg.items():
        row: dict[str, Any] = {
            "n_units": len(unit_rs),
            "n_samples_total": sum(r.n_samples for r in unit_rs),
            "n_correct_total": sum(r.n_correct for r in unit_rs),
        }
        for k in ks:
            vals = [r.pass_at[k] for r in unit_rs if k in r.pass_at]
            row[f"pass@{k}"] = (
                float(sum(vals) / len(vals)) if vals else None
            )
        # Aggregate error types.
        err_total: dict[str, int] = defaultdict(int)
        for r in unit_rs:
            for kind, count in r.error_counts.items():
                err_total[kind] += count
        row["error_counts"] = dict(err_total)
        key = f"{gen_dataset}|{model}|{alg}|mode={mode}|variant={variant}"
        summary[key] = row
    return summary


def main() -> int:
    args = parse_args()
    units = _datasets_to_units(args)
    print(f"[info] {len(units)} (gen, variant, task, model, alg) units; "
          f"{args.workers} workers", flush=True)

    # Fail loudly when a requested (model, alg) was never generated — scoring
    # it would silently mark every sample missing and report pass@k = 0.
    pairs = model_alg_pairs(args.models, args.algs, args.model_algs)
    gen_datasets = sorted({gen_dataset_for(d) for d in args.datasets})
    problems = check_records_exist(args.output_root, gen_datasets, pairs)
    if problems:
        for msg in problems:
            print(f"[{'warn' if args.allow_missing else 'error'}] {msg}",
                  file=sys.stderr, flush=True)
        if not args.allow_missing:
            raise SystemExit(
                "requested (model, alg) combinations have no records on disk "
                "(see above). Fix --model-algs / --models / --algs to match "
                "what generate.py produced, or pass --allow-missing."
            )

    out_dir = args.out or os.path.join(args.output_root, "passk")
    os.makedirs(out_dir, exist_ok=True)

    all_results: list[UnitResult] = []
    # Group units by (gen_dataset, task, model, alg) so we evaluate each
    # sample once and reuse it across variants.
    grouped: dict[tuple[str, Task, str, str], list[str]] = defaultdict(list)
    for gd, variant, task, model, alg in units:
        grouped[(gd, task, model, alg)].append(variant)
    work = list(grouped.items())

    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=mp.get_context("spawn"),
    ) as pool:
        futures = {
            pool.submit(
                evaluate_unit,
                output_root=args.output_root,
                gen_dataset=gd,
                task=task,
                model=model,
                alg=alg,
                n_samples=args.n_samples,
                modes=args.modes,
                variants=sorted(set(variants)),
                timeout_s=args.timeout,
                tokenizer_path=args.tokenizer,
                ks=tuple(args.ks),
            ): (gd, task.task_id, model, alg)
            for (gd, task, model, alg), variants in work
        }
        for i, fut in enumerate(as_completed(futures), 1):
            gd, tid, model, alg = futures[fut]
            try:
                unit_results = fut.result()
                all_results.extend(unit_results)
                # Brief progress: print pass-rate per mode for this task.
                lines = [f"  {r.mode}/{r.variant}: "
                         f"{r.n_correct}/{r.n_samples}"
                         for r in unit_results]
                print(f"[{i}/{len(futures)}] {gd}/{tid}/{model}/{alg}\n"
                      + "\n".join(lines), flush=True)
            except Exception as exc:
                print(f"[err] {gd}/{tid}/{model}/{alg}: "
                      f"{type(exc).__name__}: {exc}", flush=True)

    # ---- per-sample jsonl ----
    per_sample_path = os.path.join(out_dir, "per_sample.jsonl")
    with open(per_sample_path, "w") as fh:
        for r in all_results:
            for k, ok in enumerate(r.sample_pass):
                fh.write(json.dumps({
                    "task_id": r.task_id, "gen_dataset": r.gen_dataset,
                    "model": r.model, "alg": r.alg,
                    "mode": r.mode, "variant": r.variant,
                    "sample_k": k, "passed": bool(ok),
                }) + "\n")

    # ---- per-unit json ----
    per_unit = {
        f"{r.task_id}|{r.model}|{r.alg}|mode={r.mode}|variant={r.variant}": {
            "task_id": r.task_id, "gen_dataset": r.gen_dataset,
            "model": r.model, "alg": r.alg,
            "mode": r.mode, "variant": r.variant,
            "n_samples": r.n_samples, "n_correct": r.n_correct,
            "pass_at": {str(k): v for k, v in r.pass_at.items()},
            "error_counts": r.error_counts,
        }
        for r in all_results
    }
    with open(os.path.join(out_dir, "per_unit.json"), "w") as fh:
        json.dump(per_unit, fh, indent=2)

    # ---- aggregate summary ----
    summary = _aggregate(all_results, ks=args.ks)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # ---- print table ----
    print("\n=========== pass@k summary ===========")
    headers = ["dataset|model|alg|mode|variant"] + [f"pass@{k}" for k in args.ks]
    print("  ".join(f"{h:<40}" if i == 0 else f"{h:<10}"
                    for i, h in enumerate(headers)))
    for key, row in sorted(summary.items()):
        cells = [key] + [
            f"{row[f'pass@{k}']*100:.2f}" if row.get(f"pass@{k}") is not None
            else "—"
            for k in args.ks
        ]
        print("  ".join(f"{c:<40}" if i == 0 else f"{c:<10}"
                        for i, c in enumerate(cells)))
    print(f"\n[done] wrote {per_sample_path} + per_unit.json + summary.json "
          f"under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
