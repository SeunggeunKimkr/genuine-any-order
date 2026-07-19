"""Shared helpers for code-eval generation across HumanEval/MBPP family."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, List, Sequence

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

# "Generation" datasets. HE and HE+ share the same prompt/skeleton, so we
# generate once under HUMANEVAL and reuse the traces for HE+ at pass@k time.
# Same for MBPP / MBPP+.
GEN_HUMANEVAL = "humaneval"
GEN_MBPP = "mbpp"
ALL_GEN_DATASETS = (GEN_HUMANEVAL, GEN_MBPP)

# pass@k variants — choose which test code to grade against.
VARIANT_BASE = "base"
VARIANT_PLUS = "plus"
ALL_VARIANTS = (VARIANT_BASE, VARIANT_PLUS)

# Convenience flag-style dataset names (what users type on the CLI / yaml).
DATASET_HUMANEVAL = "humaneval"           # HE base
DATASET_HUMANEVAL_PLUS = "humaneval_plus" # HE+
DATASET_MBPP = "mbpp"                     # MBPP base
DATASET_MBPP_PLUS = "mbpp_plus"           # MBPP+
ALL_DATASETS = (
    DATASET_HUMANEVAL, DATASET_HUMANEVAL_PLUS, DATASET_MBPP, DATASET_MBPP_PLUS,
)


def gen_dataset_for(dataset: str) -> str:
    """Map a CLI dataset name onto its underlying generation dataset."""
    if dataset in (DATASET_HUMANEVAL, DATASET_HUMANEVAL_PLUS):
        return GEN_HUMANEVAL
    if dataset in (DATASET_MBPP, DATASET_MBPP_PLUS):
        return GEN_MBPP
    raise ValueError(f"Unknown dataset {dataset!r}; expected one of {ALL_DATASETS}.")


def variant_for(dataset: str) -> str:
    if dataset in (DATASET_HUMANEVAL, DATASET_MBPP):
        return VARIANT_BASE
    if dataset in (DATASET_HUMANEVAL_PLUS, DATASET_MBPP_PLUS):
        return VARIANT_PLUS
    raise ValueError(f"Unknown dataset {dataset!r}.")


# ---------------------------------------------------------------------------
# Task data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Task:
    """A single code-eval problem in a dataset-agnostic shape.

    ``prompt`` is the text actually fed to the encoder (skeleton for HE;
    fully-rendered Dream-Coder MBPP-style prompt for MBPP — see
    ``render_mbpp_prompt`` for the format).

    ``test_base`` / ``test_plus`` hold the dataset-native test code:
      - HumanEval: row["test"] from openai/openai_humaneval
      - HumanEval+: row["test"] from evalplus/humanevalplus
      - MBPP: assembled from row["test_list"] (3 baseline asserts)
      - MBPP+: row["test"] from evalplus/mbppplus

    ``test_list`` and ``test_imports`` are MBPP-specific and empty for HE.
    """

    dataset: str  # GEN_HUMANEVAL | GEN_MBPP
    index: int
    task_id: str
    prompt: str
    entry_point: str
    canonical_solution: str
    test_base: str = ""
    test_plus: str = ""
    test_list: tuple[str, ...] = field(default_factory=tuple)
    test_imports: tuple[str, ...] = field(default_factory=tuple)

    def test_for(self, variant: str) -> str:
        if variant == VARIANT_BASE:
            return self.test_base
        if variant == VARIANT_PLUS:
            return self.test_plus
        raise ValueError(f"Unknown variant {variant!r}.")


# Back-compat alias for callers that still import the old name.
HumanEvalTask = Task


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


# Matches Dream-Coder's official MBPP eval prompt
# (~/Dream-Coder/base/lm_eval/tasks/mbpp/mbpp.yaml).
MBPP_PROMPT_TEMPLATE = (
    "{problem}\n"
    "The code should pass these tests:\n\n"
    "{test_a}\n"
    "{test_b}\n"
    "{test_c}\n\n"
    "Here is the completed function:\n```python\n"
)


def render_mbpp_prompt(*, problem: str, test_list: Sequence[str]) -> str:
    """Render the Dream-Coder MBPP prompt for a single problem.

    Requires exactly 3 entries in ``test_list`` (the official MBPP convention;
    every problem in evalplus/mbppplus has ≥ 3).
    """
    if len(test_list) < 3:
        raise ValueError(
            f"MBPP problem must have ≥3 baseline tests; got {len(test_list)}."
        )
    return MBPP_PROMPT_TEMPLATE.format(
        problem=problem,
        test_a=test_list[0],
        test_b=test_list[1],
        test_c=test_list[2],
    )


def _first_def_name(code: str) -> str:
    """Naive: split on the first ``def `` and grab the name. Matches the
    historical Dream-Coder convention but breaks when the canonical solution
    defines a helper function before the actual entry point."""
    try:
        return code.split("def ", 1)[1].split("(", 1)[0].strip()
    except IndexError:
        raise ValueError(f"Could not extract entry point from MBPP code: {code!r}")


def extract_mbpp_entry_point(code: str, test_list: Sequence[str] = ()) -> str:
    """Extract the function name that the tests actually call.

    The Dream-Coder convention (just take the first ``def`` in ``code``) is
    wrong on the ~8 MBPP tasks where the canonical solution defines a helper
    function *before* the entry point — e.g., Mbpp/6 declares
    ``is_Power_Of_Two`` first but tests call ``differ_At_One_Bit_Pos``.

    Strategy:
      1. AST-parse ``code`` and collect every top-level (or nested-Module-body)
         function name that's *defined*.
      2. AST-parse the joined ``test_list`` (or the canonical-solution call
         sites) and look at every function-call name.
      3. Return the first defined function that the tests actually call.
      4. If no test_list is supplied or no overlap exists, fall back to the
         first ``def`` (preserves prior behavior for tasks where it was right).
    """
    import ast as _ast
    defined: list[str] = []
    try:
        tree = _ast.parse(code)
        for node in tree.body:
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                defined.append(node.name)
    except SyntaxError:
        pass

    if not defined:
        return _first_def_name(code)

    if test_list:
        called: set[str] = set()
        for line in test_list:
            try:
                t = _ast.parse(str(line))
            except SyntaxError:
                continue
            for node in _ast.walk(t):
                if isinstance(node, _ast.Call):
                    fn = node.func
                    if isinstance(fn, _ast.Name):
                        called.add(fn.id)
                    elif isinstance(fn, _ast.Attribute):
                        # `math.isclose(volume_sphere(10), ...)` — the inner
                        # call is what we care about. The walk visits both.
                        pass
        for name in defined:
            if name in called:
                return name

    # Single-def case → only one option.
    if len(defined) == 1:
        return defined[0]
    return _first_def_name(code)


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


def _hf_load(path: str, *, split: str = "test"):
    from datasets import load_dataset
    return load_dataset(path, split=split)


def load_humaneval_tasks() -> list[Task]:
    """Load all 164 HumanEval problems with both base and plus tests.

    The base dataset (openai/openai_humaneval) and plus dataset
    (evalplus/humanevalplus) share task_ids; we join on ``task_id`` so each
    Task has both ``test_base`` and ``test_plus`` populated.
    """
    he = _hf_load("openai/openai_humaneval")
    hep = _hf_load("evalplus/humanevalplus")
    plus_by_id = {str(row["task_id"]): row for row in hep}
    tasks: list[Task] = []
    for idx, row in enumerate(he):
        task_id = str(row["task_id"])
        plus_row = plus_by_id.get(task_id, {})
        tasks.append(
            Task(
                dataset=GEN_HUMANEVAL,
                index=idx,
                task_id=task_id,
                prompt=str(row["prompt"]),
                entry_point=str(row["entry_point"]),
                canonical_solution=str(row.get("canonical_solution", "")),
                test_base=str(row["test"]),
                test_plus=str(plus_row.get("test", "")),
            )
        )
    return tasks


def load_mbpp_tasks() -> list[Task]:
    """Load all evalplus/mbppplus problems.

    For MBPP we serve both base (test_list[0..2] joined as `test_base`) and
    plus (row["test"] as `test_plus`) tests. The model prompt is rendered via
    ``render_mbpp_prompt`` using the 3 baseline tests, matching Dream-Coder.
    """
    ds = _hf_load("evalplus/mbppplus")
    tasks: list[Task] = []
    for idx, row in enumerate(ds):
        problem = str(row.get("prompt") or row.get("text") or "")
        test_list = tuple(str(t) for t in row["test_list"])
        if len(test_list) < 3:
            # Skip pathological rows; evalplus/mbppplus shouldn't have any.
            continue
        rendered = render_mbpp_prompt(problem=problem, test_list=test_list)
        entry_point = extract_mbpp_entry_point(str(row["code"]), test_list)
        # The official MBPP base eval uses test_list as the entire test code.
        test_base = "\n".join(test_list[:3])
        tasks.append(
            Task(
                dataset=GEN_MBPP,
                index=idx,
                task_id=f"Mbpp/{int(row['task_id'])}",
                prompt=rendered,
                entry_point=entry_point,
                canonical_solution=str(row.get("code", "")),
                test_base=test_base,
                test_plus=str(row.get("test", "")),
                test_list=test_list,
                test_imports=tuple(str(t) for t in row.get("test_imports", []) or []),
            )
        )
    return tasks


def load_tasks(dataset: str) -> list[Task]:
    """Top-level loader keyed on the CLI dataset name. HE and HE+ produce the
    same Task list (just an alias since the generation prompts are identical);
    same for MBPP and MBPP+. The pass@k step picks ``test_base`` vs
    ``test_plus`` based on the variant."""
    gd = gen_dataset_for(dataset)
    if gd == GEN_HUMANEVAL:
        return load_humaneval_tasks()
    if gd == GEN_MBPP:
        return load_mbpp_tasks()
    raise AssertionError(f"Unhandled gen dataset {gd!r}.")


# ---------------------------------------------------------------------------
# (De)serializing the chosen task list to JSON for re-use across shards
# ---------------------------------------------------------------------------


def _task_to_dict(t: Task) -> dict:
    return {
        "dataset": t.dataset,
        "index": t.index,
        "task_id": t.task_id,
        "prompt": t.prompt,
        "entry_point": t.entry_point,
        "canonical_solution": t.canonical_solution,
        "test_base": t.test_base,
        "test_plus": t.test_plus,
        "test_list": list(t.test_list),
        "test_imports": list(t.test_imports),
    }


def _task_from_dict(d: dict) -> Task:
    return Task(
        dataset=str(d["dataset"]),
        index=int(d["index"]),
        task_id=str(d["task_id"]),
        prompt=str(d["prompt"]),
        entry_point=str(d["entry_point"]),
        canonical_solution=str(d.get("canonical_solution", "")),
        test_base=str(d.get("test_base", "")),
        test_plus=str(d.get("test_plus", "")),
        test_list=tuple(d.get("test_list", []) or []),
        test_imports=tuple(d.get("test_imports", []) or []),
    )


def save_tasks(path: str, tasks: Sequence[Task]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as h:
        json.dump([_task_to_dict(t) for t in tasks], h, indent=2)


def load_saved_tasks(path: str) -> list[Task]:
    with open(path, "r") as h:
        raw = json.load(h)
    return [_task_from_dict(row) for row in raw]


# Backwards-compat aliases used by old code.
def save_chosen_tasks(path: str, tasks: Sequence[Task]) -> None:
    save_tasks(path, tasks)


def load_chosen_tasks(path: str) -> list[Task]:
    return load_saved_tasks(path)


# ---------------------------------------------------------------------------
# Stratified subset (used for sanity-check runs)
# ---------------------------------------------------------------------------


def stratified_indices(n_total: int, n: int) -> list[int]:
    """Evenly spaced integer indices across [0, n_total-1]."""
    if n > n_total:
        raise ValueError(
            f"requested {n} stratified indices but only {n_total} available"
        )
    picks = np.linspace(0, n_total - 1, n).round().astype(int)
    dedup: list[int] = []
    seen: set[int] = set()
    for p in picks:
        p = int(p)
        while p in seen:
            p += 1
        seen.add(p)
        dedup.append(p)
    return dedup


def subset_tasks(tasks: Sequence[Task], *, limit: int | None,
                 stratified: bool = False) -> list[Task]:
    """Return a subset of tasks. ``limit=None`` returns all tasks unchanged."""
    if limit is None or limit >= len(tasks):
        return list(tasks)
    if stratified:
        picks = stratified_indices(len(tasks), limit)
        return [tasks[i] for i in picks]
    return list(tasks[:limit])


# Legacy helper, retained so old scripts don't break — uses HE only.
def load_humaneval_stratified(n: int = 25) -> list[Task]:
    return subset_tasks(load_humaneval_tasks(), limit=n, stratified=True)


# ---------------------------------------------------------------------------
# Prompt encoders (FlexMDM native, Dream-Coder)
# ---------------------------------------------------------------------------


def encode_flexmdm_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    max_length: int,
    pad_id: int,
    prepend_bos: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """FlexMDM prompt encoding — mirrors training.

    Layout: ``[bos] + prompt_ids + [pad_id] separator``, then padded with
    ``pad_id`` to ``max_length``. ``attention_mask == prompt_mask`` over the
    prefix. This matches data.py:744-794 and
    flexmdm_inference_test_.py:encode_prompts. Returns single-batch (1, L)
    tensors plus the prefix boundary length.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    prefix: list[int] = []
    if prepend_bos:
        if tokenizer.bos_token_id is None:
            raise ValueError("prepend_bos=True but tokenizer has no bos_token_id.")
        prefix.append(int(tokenizer.bos_token_id))
    prefix.extend(int(t) for t in prompt_ids)
    prefix.append(int(pad_id))  # separator
    boundary = len(prefix)
    if boundary >= max_length:
        raise ValueError(
            f"Prompt boundary {boundary} >= max_length {max_length}; "
            "shorten the prompt."
        )
    input_ids = torch.full((1, max_length), pad_id, dtype=torch.long)
    input_ids[0, :boundary] = torch.tensor(prefix, dtype=torch.long)
    attention_mask = torch.zeros((1, max_length), dtype=torch.bool)
    attention_mask[0, :boundary] = True
    prompt_mask = attention_mask.clone()
    return input_ids, attention_mask, prompt_mask, boundary


# DC instruct wrap (HE only) — matches Dream-Coder's evalplus instruct format.
_DC_INSTRUCT_USER_PREFIX = (
    "Please provide a self-contained Python script that solves the following "
    "problem in a markdown code block:"
)
_DC_INSTRUCT_RESPONSE_PREFIX = (
    "Below is a Python script with a self-contained function that solves the "
    "problem and passes corresponding tests:"
)


def encode_dreamcoder_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    add_bos: bool = True,
    prompt_style: str = "raw",
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Dream-Coder base prompt encoding.

    ``prompt_style="raw"``:
        ``[bos] + prompt_ids``. Used when ``prompt`` is already fully rendered
        (e.g., Dream-Coder MBPP format, which already contains the
        ``Here is the completed function:\\n\\`\\`\\`python\\n`` opener).
    ``prompt_style="instruct"``:
        ``[bos] + DC instruct user turn wrapping prompt + DC response opener``.
        Used for HumanEval, where ``prompt`` is the bare function skeleton.

    Returns (1, L) input_ids, (1, L) attention_mask, and prompt_len.
    """
    if prompt_style == "raw":
        text = tokenizer.bos_token + prompt if add_bos else prompt
    elif prompt_style == "instruct":
        user = (
            f"{_DC_INSTRUCT_USER_PREFIX}\n```python\n{prompt.strip()}\n```\n"
        )
        resp = f"{_DC_INSTRUCT_RESPONSE_PREFIX}\n```python\n"
        text = (tokenizer.bos_token if add_bos else "") + user + resp
    else:
        raise ValueError(
            f"prompt_style must be 'raw' or 'instruct', got {prompt_style!r}"
        )
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc.input_ids
    if tokenizer.pad_token_id is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = input_ids.ne(tokenizer.pad_token_id)
    return input_ids, attention_mask.bool(), int(input_ids.shape[1])


def dreamcoder_prompt_style_for(gen_dataset: str) -> str:
    """Per MBPP-A: HumanEval uses DC instruct wrap; MBPP feeds the rendered
    Dream-Coder MBPP prompt raw."""
    if gen_dataset == GEN_HUMANEVAL:
        return "instruct"
    if gen_dataset == GEN_MBPP:
        return "raw"
    raise ValueError(f"Unknown gen_dataset {gen_dataset!r}.")


# ---------------------------------------------------------------------------
# Trace stacking & .pt I/O
# ---------------------------------------------------------------------------


def stack_trajectory(snapshots: List[torch.Tensor]) -> torch.Tensor:
    """Stack a list of (B, L) per-step tensors into (S, B, L), cast to int32."""
    stacked = torch.stack([s.cpu() for s in snapshots], dim=0)
    if stacked.dtype != torch.int32:
        stacked = stacked.to(torch.int32)
    return stacked


def stack_attention(snapshots: List[torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack([s.cpu() for s in snapshots], dim=0)
    return stacked.to(torch.bool)


def save_record(
    path: str,
    *,
    prompt: str,
    prompt_len: int,
    sequences: torch.Tensor,        # (S, L) int32
    attention_masks: torch.Tensor,  # (S, L) bool
    mask_id: int,
    pad_id: int,
    meta: dict,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "prompt": prompt,
            "prompt_len": int(prompt_len),
            "sequences": sequences.contiguous(),
            "attention_masks": attention_masks.contiguous(),
            "mask_id": int(mask_id),
            "pad_id": int(pad_id),
            "meta": dict(meta),
        },
        path,
    )


def record_path(output_root: str, gen_dataset: str, task_id: str,
                model: str, alg: str, sample_k: int) -> str:
    safe_task = task_id.replace("/", "_")
    folder = os.path.join(output_root, gen_dataset, "raw",
                          f"{safe_task}__{model}__{alg}")
    return os.path.join(folder, f"sample_{sample_k:02d}.pt")


def html_path(output_root: str, gen_dataset: str, task_id: str,
              model: str, alg: str, sample_k: int) -> str:
    safe_task = task_id.replace("/", "_")
    folder = os.path.join(output_root, gen_dataset, "html",
                          f"{safe_task}__{model}__{alg}")
    return os.path.join(folder, f"sample_{sample_k:02d}.html")


def tasks_path(output_root: str, gen_dataset: str) -> str:
    return os.path.join(output_root, gen_dataset, "tasks.json")


# ---------------------------------------------------------------------------
# Sharding & work-unit construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkItem:
    task: Task
    model: str          # "flexmdm" | "dreamcoder"
    alg: str            # "top_k" (FlexMDM) | "entropy" | "origin"
    sample_k: int


def build_work_list(
    tasks: Sequence[Task],
    *,
    models: Sequence[str],
    algs: Sequence[str],
    n_samples: int,
) -> list[WorkItem]:
    """Deterministic ordering: task-major, then model, then alg, then sample."""
    work: list[WorkItem] = []
    for task in tasks:
        for model in models:
            for alg in algs:
                for k in range(n_samples):
                    work.append(WorkItem(task=task, model=model, alg=alg, sample_k=k))
    return work


def shard(items: Sequence[Any], shard_index: int, num_shards: int) -> list[Any]:
    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    if not (0 <= shard_index < num_shards):
        raise ValueError(f"shard_index {shard_index} out of range [0, {num_shards})")
    return [x for i, x in enumerate(items) if i % num_shards == shard_index]


# ---------------------------------------------------------------------------
# Per-model (model, alg) pairing for the scoring CLIs
# ---------------------------------------------------------------------------


def parse_model_algs(specs: Sequence[str]) -> dict[str, list[str]]:
    """Parse ``model=alg[,alg...]`` pairings, e.g. ``flexmdm=top_k``."""
    pairs: dict[str, list[str]] = {}
    for spec in specs:
        model, sep, algs = spec.partition("=")
        model = model.strip()
        alg_list = [a.strip() for a in algs.split(",") if a.strip()]
        if not sep or not model or not alg_list:
            raise ValueError(
                f"--model-algs entries must look like model=alg[,alg...], got {spec!r}"
            )
        pairs.setdefault(model, []).extend(alg_list)
    return pairs


def model_alg_pairs(
    models: Sequence[str],
    algs: Sequence[str],
    model_algs: Sequence[str] | None = None,
) -> list[tuple[str, str]]:
    """(model, alg) combinations to score.

    When ``model_algs`` is given (e.g. ``["flexmdm=top_k", "dreamcoder=entropy"]``)
    each model is paired only with its own algs — the released runs use
    different decoding algs per model, so the plain ``models x algs`` cartesian
    would request combinations that were never generated. Otherwise falls back
    to the cartesian product of ``models`` and ``algs``.
    """
    if model_algs:
        parsed = parse_model_algs(model_algs)
        return [(m, a) for m, alg_list in parsed.items() for a in alg_list]
    return [(m, a) for m in models for a in algs]


def available_record_algs(output_root: str, gen_dataset: str) -> dict[str, set[str]]:
    """Scan ``<output_root>/<gen_dataset>/raw`` and return the {model: {alg,...}}
    combinations that actually have record directories on disk."""
    raw = os.path.join(output_root, gen_dataset, "raw")
    found: dict[str, set[str]] = {}
    if not os.path.isdir(raw):
        return found
    for name in os.listdir(raw):
        parts = name.split("__")
        if len(parts) >= 3:
            found.setdefault(parts[-2], set()).add(parts[-1])
    return found


def check_records_exist(
    output_root: str,
    gen_datasets: Sequence[str],
    pairs: Sequence[tuple[str, str]],
) -> list[str]:
    """Return one error string per (gen_dataset, model, alg) with no records
    on disk. Callers should fail loudly on a non-empty result: scoring a
    (model, alg) that was never generated would otherwise silently count
    every sample as missing (pass@k = 0)."""
    errors: list[str] = []
    for gd in gen_datasets:
        found = available_record_algs(output_root, gd)
        for model, alg in pairs:
            if alg not in found.get(model, set()):
                have = ", ".join(
                    f"{m}: {sorted(a)}" for m, a in sorted(found.items())
                ) or "no record dirs at all"
                errors.append(
                    f"{gd}: no records for model={model!r} alg={alg!r} under "
                    f"{os.path.join(output_root, gd, 'raw')} (on disk -> {have})"
                )
    return errors
