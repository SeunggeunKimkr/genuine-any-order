"""Streaming dataset stats, filtering, and tokenization pipeline for FlexMDM."""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "config",
    "fsdp_train.yaml",
)
DEFAULT_TOKENIZER = os.environ.get("BASE_MODEL", "Dream-org/Dream-Coder-v0-Base-7B")
# Fallback root when a config does not specify `data.scratch_root`; normally set
# via the SCRATCH_ROOT env var (see config/paths.env.template).
DEFAULT_SCRATCH_ROOT = os.environ.get("SCRATCH_ROOT", "./flexmdm_tokenized")
DEFAULT_MAX_LENGTH = 768
DEFAULT_SEP = "\n"
DEFAULT_PREPEND_BOS = True
DEFAULT_STATS_FILENAME = "dataset_stats.json"
DEFAULT_PRETOKENIZED_DIRNAME = "pretokenized"
DEFAULT_MANIFEST_FILENAME = "manifest.json"
DEFAULT_SEPARATOR_KIND = "internal_pad"


class FlexMDMDataStatsError(RuntimeError):
    """Raised when FlexMDM dataset stats cannot be collected."""


@dataclass(frozen=True)
class DatasetSpec:
    dataset_path: str
    split: str = "train"
    config: Optional[str] = None
    prompt_field: Optional[str] = None
    answer_field: Optional[str] = None
    streaming: bool = True
    kind: str = "prompt_answer"
    require_is_passed: bool = False
    require_is_passed_if_present: bool = False
    require_score_at_least: Optional[float] = None
    require_score_exact: Optional[float] = None
    require_python_code: bool = False
    strip_trailing_demo: bool = False
    require_llm_judgement_min_score: Optional[int] = None
    llm_judgement_field: Optional[str] = None
    require_subset_in: Optional[Tuple[str, ...]] = None
    require_style_in: Optional[Tuple[str, ...]] = None
    require_gpt_pass_percentage_above: Optional[float] = None
    prompt_uses_starter_code: bool = False


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "rstarcoder-seed-sft": DatasetSpec(
        dataset_path="microsoft/rStar-Coder",
        config="seed_sft",
        kind="rstar",
        require_is_passed=True,
        prompt_uses_starter_code=True,
    ),
    "rstarcoder-synthetic-sft": DatasetSpec(
        dataset_path="microsoft/rStar-Coder",
        config="synthetic_sft",
        kind="rstar",
        require_is_passed_if_present=True,
    ),
    "opc-sft-stage2-educational": DatasetSpec(
        dataset_path="OpenCoder-LLM/opc-sft-stage2",
        config="educational_instruct",
        prompt_field="instruction",
        answer_field="code",
    ),
    "KodCode-V1-SFT-4o": DatasetSpec(
        dataset_path="KodCode/KodCode-V1-SFT-4o",
        kind="kodcode",
    ),
    "KodCode-V1": DatasetSpec(
        dataset_path="KodCode/KodCode-V1",
        kind="kodcode_v1",
        require_subset_in=(
            "Prefill",
            "Filter",
            "Leetcode",
            "Algorithm",
            "Data_Structure",
            "Apps",
        ),
        require_style_in=("instruct", "complete"),
        require_gpt_pass_percentage_above=0.1,
    ),
    "OpenCodeInstruct-score1-py-all": DatasetSpec(
        dataset_path="nvidia/OpenCodeInstruct",
        prompt_field="input",
        answer_field="output",
        require_score_at_least=0.9,
        require_python_code=True,
        strip_trailing_demo=True,
    ),
    "OpenCodeInstruct-hq-py": DatasetSpec(
        dataset_path="nvidia/OpenCodeInstruct",
        prompt_field="input",
        answer_field="output",
        require_score_exact=1.0,
        require_python_code=True,
        strip_trailing_demo=True,
        require_llm_judgement_min_score=5,
        llm_judgement_field="llm_judgement",
    ),
}
DEFAULT_DATASET_NAMES = tuple(DATASET_SPECS.keys())

DROP_REASONS = (
    "missing_field",
    "empty_text",
    "quality_filter",
    "postprocess_filter",
    "prompt_too_long",
    "flask_main_demo",
    "judgement_filter",
)

_FENCE_RE = re.compile(
    r"```[ \t]*([a-zA-Z0-9_+\-]*)[ \t]*\n(.*?)```",
    flags=re.DOTALL,
)
PY_MARKERS = ("def ", "class ", "import ", "from ")
NONPY_MARKERS = (
    "#include",
    "public class",
    "using ",
    "func ",
    "package ",
    "namespace ",
    "std::",
    "System.out",
)


@dataclass
class LengthStats:
    values: List[int] = field(default_factory=list)

    def add(self, value: int) -> None:
        self.values.append(int(value))

    def summary(self) -> Dict[str, Optional[float]]:
        if not self.values:
            return {
                "count": 0,
                "min": None,
                "mean": None,
                "p50": None,
                "p90": None,
                "p95": None,
                "p99": None,
                "max": None,
            }

        values = sorted(self.values)
        return {
            "count": len(values),
            "min": float(values[0]),
            "mean": float(sum(values) / len(values)),
            "p50": float(_percentile(values, 0.50)),
            "p90": float(_percentile(values, 0.90)),
            "p95": float(_percentile(values, 0.95)),
            "p99": float(_percentile(values, 0.99)),
            "max": float(values[-1]),
        }


@dataclass
class CorpusStats:
    dataset_name: str
    seen: int = 0
    kept: int = 0
    dropped: Dict[str, int] = field(
        default_factory=lambda: {reason: 0 for reason in DROP_REASONS}
    )
    answer_truncated: int = 0
    zero_answer_capacity: int = 0
    prompt_boundary_lengths: LengthStats = field(default_factory=LengthStats)
    answer_lengths: LengthStats = field(default_factory=LengthStats)
    total_lengths: LengthStats = field(default_factory=LengthStats)

    def drop(self, reason: str) -> None:
        if reason not in self.dropped:
            self.dropped[reason] = 0
        self.dropped[reason] += 1

    def add_candidate_lengths(
        self,
        *,
        prompt_boundary_length: int,
        answer_length: int,
        total_length: int,
    ) -> None:
        self.prompt_boundary_lengths.add(prompt_boundary_length)
        self.answer_lengths.add(answer_length)
        self.total_lengths.add(total_length)

    def mark_kept(
        self,
        *,
        total_length: int,
        max_length: int,
    ) -> None:
        self.kept += 1
        if total_length > max_length:
            self.answer_truncated += 1

    def to_dict(self) -> Dict[str, Any]:
        dropped_total = sum(self.dropped.values())
        retention_rate = self.kept / self.seen if self.seen else 0.0
        truncation_rate = (
            self.answer_truncated / self.kept if self.kept else 0.0
        )
        return {
            "dataset_name": self.dataset_name,
            "seen": self.seen,
            "kept": self.kept,
            "dropped_total": dropped_total,
            "retention_rate": retention_rate,
            "drops": dict(self.dropped),
            "answer_truncated": self.answer_truncated,
            "answer_truncation_rate": truncation_rate,
            "zero_answer_capacity": self.zero_answer_capacity,
            "lengths": {
                "prompt_boundary": self.prompt_boundary_lengths.summary(),
                "answer": self.answer_lengths.summary(),
                "total": self.total_lengths.summary(),
            },
        }


def _percentile(sorted_values: Sequence[int], q: float) -> int:
    if not sorted_values:
        raise ValueError("Cannot compute percentile of empty values")
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = int(math.ceil(q * len(sorted_values))) - 1
    idx = max(0, min(idx, len(sorted_values) - 1))
    return sorted_values[idx]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "pass", "passed"}


def extract_code_only_prefer_python(text: Any) -> Optional[str]:
    if text is None:
        return None

    stripped = str(text).strip()
    if not stripped:
        return None

    blocks = list(_FENCE_RE.finditer(stripped))
    if blocks:
        python_blocks = [
            block
            for block in blocks
            if block.group(1).strip().lower() in {"py", "python"}
        ]
        picked = python_blocks[-1] if python_blocks else blocks[-1]
        code = picked.group(2).strip("\n\r\t ")
        return code if code else None

    looks_python = any(marker in stripped for marker in PY_MARKERS) and not any(
        marker in stripped for marker in NONPY_MARKERS
    )
    return stripped if looks_python else None


def is_python_like(code: str) -> bool:
    if not code:
        return False
    stripped = code.strip()
    return any(marker in stripped for marker in PY_MARKERS) and not any(
        marker in stripped for marker in NONPY_MARKERS
    )


_KEEP_KINDS = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Import,
    ast.ImportFrom,
)

_PROMPT_SCRIPT_RE = re.compile(
    r"\b(script|program|command[- ]line|cli|argparse|argv|sys\.argv|standard input)\b"
    r"|input\(\s*\)",
    re.IGNORECASE,
)

_TRAILING_DEMO_BANNER_RE = re.compile(
    r"^\s*#\s*(example(\s*usage)?|test(\s+the(\s+\w+)?)?|test\s+cases?"
    r"|unit\s+tests?|demo|usage|sample|driver|main)\b[:\s].*$",
    re.IGNORECASE,
)

_DEMO_DEF_NAME_RE = re.compile(
    r"^("
    r"test|tests|example|examples|demo|usage|sample|samples|driver|main|run"
    r"|(test|example|examples|demo|usage|sample|samples|main|run)_.+"
    r"|.+_(test|tests|example|examples|demo|usage)"
    r")$",
    re.IGNORECASE,
)


def _is_demo_like_def(
    node: ast.AST, prompt: Optional[str] = None
) -> bool:
    """A top-level FunctionDef whose name looks like a test/example/demo
    helper, and that is *not* explicitly named in the prompt (which would
    indicate it is the actual answer)."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    if not _DEMO_DEF_NAME_RE.match(node.name):
        return False
    if prompt and node.name in prompt:
        return False
    return True


def _is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if not isinstance(test, ast.Compare):
        return False
    if not (isinstance(test.left, ast.Name) and test.left.id == "__name__"):
        return False
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    if len(test.comparators) != 1:
        return False
    cmp = test.comparators[0]
    return isinstance(cmp, ast.Constant) and cmp.value == "__main__"


def _is_pure_demo_node(
    node: ast.AST, prompt: Optional[str] = None
) -> bool:
    """Whitelist of trailing top-level statements considered safe to strip."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return _is_demo_like_def(node, prompt)
    if isinstance(node, ast.If):
        if _is_main_guard(node):
            return (
                all(_is_pure_demo_node(child, prompt) for child in node.body)
                and not node.orelse
            )
        return False
    if isinstance(node, ast.Expr):
        return isinstance(node.value, ast.Call)
    if isinstance(node, (ast.Assign, ast.AugAssign)):
        return True
    if isinstance(node, (ast.For, ast.While)):
        return (
            all(_is_pure_demo_node(child, prompt) for child in node.body)
            and not node.orelse
        )
    return False


def _count_print_calls(node: ast.AST) -> int:
    count = 0
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            if isinstance(func, ast.Name) and func.id == "print":
                count += 1
    return count


def _has_argv_or_input(tree: ast.Module) -> bool:
    for n in ast.walk(tree):
        if (
            isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name)
            and n.value.id == "sys"
            and n.attr == "argv"
        ):
            return True
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "input"
        ):
            return True
    return False


def _has_flask_signals(tree: ast.Module) -> bool:
    """Detect Flask app patterns: @<x>.route decorator, Flask(...) call,
    or top-level <x>.run() call."""
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in n.decorator_list:
                f = d.func if isinstance(d, ast.Call) else d
                if isinstance(f, ast.Attribute) and f.attr == "route":
                    return True
        if isinstance(n, ast.Assign):
            v = n.value
            if (
                isinstance(v, ast.Call)
                and isinstance(v.func, ast.Name)
                and v.func.id == "Flask"
            ):
                return True
    for n in ast.walk(tree):
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "run"
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "app"
        ):
            return True
    return False


def _trim_trailing_banner_lines(text: str) -> str:
    """Remove trailing blank lines and demo-banner-comment lines from text."""
    lines = text.splitlines(keepends=True)
    while lines:
        last = lines[-1]
        if last.strip() == "":
            lines.pop()
            continue
        if _TRAILING_DEMO_BANNER_RE.match(last):
            lines.pop()
            continue
        break
    return "".join(lines)


def strip_trailing_demo_block(
    code: str, prompt: Optional[str] = None
) -> Tuple[str, Optional[str], str]:
    """Conservatively strip a trailing demo block from the answer code.

    Returns (cleaned_code, drop_reason, action), where:
      - action ∈ {"unchanged", "stripped", "drop"}
      - drop_reason is set only when action == "drop"

    Rules:
      - If the answer cannot be parsed, no top-level def/class/import is
        present, or there is no trailing/interleaved non-keep code → unchanged.
      - If the answer has a `__main__` guard AND uses Flask → drop.
      - If non-keep statements appear interleaved between def/class/imports
        (and it isn't a pure Flask app handled above) → unchanged.
      - If function bodies use `sys.argv` / `input()` or the prompt mentions
        script / program / argparse / CLI → unchanged.
      - Otherwise: strip the trailing block when it (a) consists only of
        whitelisted demo statements, OR (b) contains more than two `print(...)`
        calls (counted recursively, including inside `__main__` / `try` / loop
        bodies). Everything else → unchanged.
    """
    if not code or not code.strip():
        return code, None, "unchanged"
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, None, "unchanged"

    real_keep_indices = [
        i
        for i, node in enumerate(tree.body)
        if isinstance(node, _KEEP_KINDS) and not _is_demo_like_def(node, prompt)
    ]
    if not real_keep_indices:
        return code, None, "unchanged"
    last_real_keep_idx = real_keep_indices[-1]
    last_keep_end = tree.body[last_real_keep_idx].end_lineno

    interleaved = [
        node
        for i, node in enumerate(tree.body)
        if i < last_real_keep_idx and not isinstance(node, _KEEP_KINDS)
    ]
    tail = list(tree.body[last_real_keep_idx + 1 :])

    has_main_in_tail = any(_is_main_guard(node) for node in tail)
    has_flask = _has_flask_signals(tree)

    if has_flask and has_main_in_tail:
        return code, "flask_main_demo", "drop"

    if not tail and not interleaved:
        return code, None, "unchanged"

    if interleaved or has_flask:
        return code, None, "unchanged"

    if _has_argv_or_input(tree):
        return code, None, "unchanged"

    if prompt and _PROMPT_SCRIPT_RE.search(prompt):
        return code, None, "unchanged"

    print_count = sum(_count_print_calls(node) for node in tail)
    pure_demo = all(_is_pure_demo_node(node, prompt) for node in tail)
    if not (pure_demo or print_count > 2):
        return code, None, "unchanged"

    lines = code.splitlines(keepends=True)
    head_text = "".join(lines[:last_keep_end])
    head_text = _trim_trailing_banner_lines(head_text)
    if not head_text.strip():
        return code, None, "unchanged"
    try:
        ast.parse(head_text)
    except SyntaxError:
        return code, None, "unchanged"
    return head_text.rstrip() + "\n", None, "stripped"


def normalize_example(
    dataset_name: str,
    spec: DatasetSpec,
    example: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Normalize one source row to (prompt, answer, drop_reason)."""
    if spec.kind == "rstar":
        if spec.require_is_passed:
            if "is_passed" not in example:
                return None, None, "missing_field"
            if not truthy(example.get("is_passed")):
                return None, None, "quality_filter"
        elif spec.require_is_passed_if_present and "is_passed" in example:
            if not truthy(example.get("is_passed")):
                return None, None, "quality_filter"

        if "question" not in example or "code" not in example:
            return None, None, "missing_field"

        prompt = str(example.get("question") or "").strip()
        starter_code = str(example.get("starter_code") or "").strip()
        if spec.prompt_uses_starter_code and starter_code:
            prompt = f"{prompt}\n\nStarter code:\n{starter_code}".strip()
        answer = str(example.get("code") or "").strip()
        if not prompt or not answer:
            return None, None, "empty_text"
        return prompt, answer, None

    if spec.kind == "kodcode":
        if "question" not in example:
            return None, None, "missing_field"
        answer_key = (
            "4o_solution"
            if example.get("4o_solution") is not None
            else "solution"
        )
        if answer_key not in example:
            return None, None, "missing_field"
        prompt = str(example.get("question") or "").strip()
        answer = str(example.get(answer_key) or "").strip()
        if not prompt or not answer:
            return None, None, "empty_text"
        return prompt, answer, None

    if spec.kind == "kodcode_v1":
        if spec.require_style_in is not None:
            if str(example.get("style") or "") not in spec.require_style_in:
                return None, None, "quality_filter"
        if spec.require_subset_in is not None:
            subset = str(example.get("subset") or "")
            if subset not in spec.require_subset_in:
                return None, None, "quality_filter"
        if spec.require_gpt_pass_percentage_above is not None:
            gpp = safe_float(example.get("gpt_pass_percentage", 0.0), 0.0)
            if gpp <= float(spec.require_gpt_pass_percentage_above):
                return None, None, "quality_filter"

        if "question" not in example or "solution" not in example:
            return None, None, "missing_field"
        prompt = str(example.get("question") or "").strip()
        answer = str(example.get("solution") or "").strip()
        if not prompt or not answer:
            return None, None, "empty_text"
        return prompt, answer, None

    if spec.prompt_field is None or spec.answer_field is None:
        raise KeyError(f"Dataset {dataset_name} has no prompt/answer fields")
    if spec.prompt_field not in example or spec.answer_field not in example:
        return None, None, "missing_field"

    if spec.require_score_at_least is not None:
        score = safe_float(example.get("average_test_score", 0.0), 0.0)
        if score < float(spec.require_score_at_least):
            return None, None, "quality_filter"

    if spec.require_score_exact is not None:
        score = safe_float(example.get("average_test_score", 0.0), 0.0)
        if score != float(spec.require_score_exact):
            return None, None, "quality_filter"

    if spec.require_llm_judgement_min_score is not None:
        field_name = spec.llm_judgement_field or "llm_judgement"
        raw_judgement = example.get(field_name)
        if raw_judgement is None:
            return None, None, "judgement_filter"
        try:
            parsed = (
                json.loads(raw_judgement)
                if isinstance(raw_judgement, str)
                else raw_judgement
            )
        except (TypeError, ValueError):
            return None, None, "judgement_filter"
        if not isinstance(parsed, dict) or not parsed:
            return None, None, "judgement_filter"
        threshold = int(spec.require_llm_judgement_min_score)
        ok = True
        for value in parsed.values():
            if not isinstance(value, dict):
                ok = False
                break
            try:
                if int(value.get("score")) < threshold:
                    ok = False
                    break
            except (TypeError, ValueError):
                ok = False
                break
        if not ok:
            return None, None, "judgement_filter"

    prompt = str(example.get(spec.prompt_field) or "").strip()
    answer = str(example.get(spec.answer_field) or "").strip()
    if not prompt or not answer:
        return None, None, "empty_text"

    if spec.require_python_code:
        answer = extract_code_only_prefer_python(answer) or ""
        if not answer or not is_python_like(answer):
            return None, None, "postprocess_filter"

    if spec.strip_trailing_demo:
        cleaned, drop_reason, _action = strip_trailing_demo_block(answer, prompt)
        if drop_reason is not None:
            return None, None, drop_reason
        answer = cleaned
        if not answer or not is_python_like(answer):
            return None, None, "postprocess_filter"

    return prompt, answer, None


def load_config(config_path: str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise FlexMDMDataStatsError(
            "PyYAML is required to read FlexMDM config files."
        ) from exc

    with open(os.path.expanduser(config_path), "r") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise FlexMDMDataStatsError(f"Config {config_path} did not load as a dict")
    return loaded


def _config_get(config: Dict[str, Any], *path: str, default: Any = None) -> Any:
    value: Any = config
    for key in path:
        if isinstance(value, dict):
            if key not in value:
                return default
            value = value[key]
        elif hasattr(value, "get"):
            next_value = value.get(key, default)
            if next_value is default:
                return default
            value = next_value
        elif hasattr(value, key):
            value = getattr(value, key)
        else:
            return default
    return value


def _get_config_value(container: Any, key: str, default: Any = None) -> Any:
    if isinstance(container, dict):
        return container.get(key, default)
    if hasattr(container, "get"):
        value = container.get(key, default)
        if value is not None:
            return value
        return default
    return getattr(container, key, default)


def _resolve_dataset_names(
    config: Dict[str, Any],
    cli_dataset_names: Optional[Sequence[str]],
) -> List[str]:
    if cli_dataset_names:
        names = list(cli_dataset_names)
    else:
        names = _config_get(config, "data", "dataset_names", default=None)
        if not names:
            names = list(DEFAULT_DATASET_NAMES)
        elif isinstance(names, str):
            names = [names]
        else:
            names = [str(name) for name in names]

    unknown = [name for name in names if name not in DATASET_SPECS]
    if unknown:
        known = ", ".join(DATASET_SPECS)
        raise KeyError(f"Unknown FlexMDM dataset aliases {unknown}; known: {known}")
    return names


def _load_streaming_dataset(spec: DatasetSpec):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise FlexMDMDataStatsError(
            "The `datasets` package is required for FlexMDM streaming stats."
        ) from exc

    if spec.config is not None:
        return load_dataset(
            spec.dataset_path,
            spec.config,
            split=spec.split,
            streaming=spec.streaming,
        )
    return load_dataset(
        spec.dataset_path,
        split=spec.split,
        streaming=spec.streaming,
    )


def _load_tokenizer(
    tokenizer_name: str,
    trust_remote_code: bool,
    prepend_bos: bool,
):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise FlexMDMDataStatsError(
            "The `transformers` package is required for FlexMDM data stats."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=trust_remote_code,
    )
    if prepend_bos and tokenizer.bos_token_id is None:
        raise FlexMDMDataStatsError(
            f"Tokenizer {tokenizer_name} has no bos_token_id, but prepend_bos=True"
        )
    return tokenizer


def _single_input_ids(tokenizer: Any, text: str) -> List[int]:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def _token_lengths(
    tokenizer: Any,
    prompt: str,
    answer: str,
    separator_len: int,
    prepend_bos: bool,
) -> Tuple[int, int, int]:
    prompt_ids = _single_input_ids(tokenizer, prompt)
    answer_ids = _single_input_ids(tokenizer, answer)
    bos_len = 1 if prepend_bos else 0
    prompt_boundary_length = bos_len + len(prompt_ids) + int(separator_len)
    answer_length = len(answer_ids)
    total_length = prompt_boundary_length + answer_length
    return prompt_boundary_length, answer_length, total_length


def _process_stats_batch(
    *,
    tokenizer: Any,
    prompts: Sequence[str],
    answers: Sequence[str],
    separator_len: int,
    prepend_bos: bool,
    max_length: int,
    stats: CorpusStats,
) -> None:
    if not prompts:
        return

    prompt_ids_batch = tokenizer(
        list(prompts),
        add_special_tokens=False,
    ).input_ids
    answer_ids_batch = tokenizer(
        list(answers),
        add_special_tokens=False,
    ).input_ids
    bos_len = 1 if prepend_bos else 0

    for prompt_ids, answer_ids in zip(prompt_ids_batch, answer_ids_batch):
        prompt_boundary_length = bos_len + len(prompt_ids) + int(separator_len)
        answer_length = len(answer_ids)
        total_length = prompt_boundary_length + answer_length

        stats.add_candidate_lengths(
            prompt_boundary_length=prompt_boundary_length,
            answer_length=answer_length,
            total_length=total_length,
        )
        if prompt_boundary_length >= max_length:
            stats.zero_answer_capacity += 1
            stats.drop("prompt_too_long")
            continue

        stats.mark_kept(
            total_length=total_length,
            max_length=max_length,
        )


def collect_dataset_stats(
    config: Optional[Dict[str, Any]] = None,
    tokenizer: Any = None,
    *,
    config_path: Optional[str] = None,
    dataset_names: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    output_path: Optional[str] = None,
    batch_size: Optional[int] = None,
    print_table: bool = True,
) -> Dict[str, Any]:
    """Stream datasets and collect filtering/token length stats."""
    if config is None:
        config = load_config(config_path or DEFAULT_CONFIG_PATH)

    names = _resolve_dataset_names(config, dataset_names)
    tokenizer_name = str(
        _config_get(config, "model", "partial_pretrain", default=DEFAULT_TOKENIZER)
    )
    trust_remote_code = bool(
        _config_get(config, "model", "trust_remote_code", default=True)
    )
    data_config = _config_get(config, "data", default={}) or {}
    scratch_root = os.path.expanduser(
        str(data_config.get("scratch_root", DEFAULT_SCRATCH_ROOT))
    )
    max_length = int(data_config.get("max_length", DEFAULT_MAX_LENGTH))
    separator_kind = str(
        data_config.get("separator_kind", DEFAULT_SEPARATOR_KIND)
    )
    prepend_bos = bool(data_config.get("prepend_bos", DEFAULT_PREPEND_BOS))
    stats_filename = str(
        data_config.get("stats_output", DEFAULT_STATS_FILENAME)
    )
    if batch_size is None:
        batch_size = int(
            data_config.get(
                "stats_batch_size",
                data_config.get("pretokenize_batch_size", 2048),
            )
        )

    if max_length <= 0:
        raise ValueError(f"max_length must be positive, got {max_length}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    if tokenizer is None:
        tokenizer = _load_tokenizer(
            tokenizer_name=tokenizer_name,
            trust_remote_code=trust_remote_code,
            prepend_bos=prepend_bos,
        )
    pad_token_id = _resolve_pad_token_id(tokenizer, data_config)
    if separator_kind != DEFAULT_SEPARATOR_KIND:
        raise ValueError(
            "FlexMDM dataset stats currently supports only "
            f"separator_kind={DEFAULT_SEPARATOR_KIND!r}, got {separator_kind!r}."
        )
    separator_len = 1

    started_at = time.time()
    results: Dict[str, Any] = {}
    for name in names:
        stats = CorpusStats(dataset_name=name)
        spec = DATASET_SPECS[name]
        dataset = _load_streaming_dataset(spec)
        batch_prompts: List[str] = []
        batch_answers: List[str] = []

        for example in dataset:
            if limit is not None and stats.seen >= int(limit):
                break

            stats.seen += 1
            prompt, answer, drop_reason = normalize_example(name, spec, example)
            if drop_reason is not None:
                stats.drop(drop_reason)
                continue
            assert prompt is not None and answer is not None

            batch_prompts.append(prompt)
            batch_answers.append(answer)
            if len(batch_prompts) >= batch_size:
                _process_stats_batch(
                    tokenizer=tokenizer,
                    prompts=batch_prompts,
                    answers=batch_answers,
                    separator_len=separator_len,
                    prepend_bos=prepend_bos,
                    max_length=max_length,
                    stats=stats,
                )
                batch_prompts.clear()
                batch_answers.clear()

        if batch_prompts:
            _process_stats_batch(
                tokenizer=tokenizer,
                prompts=batch_prompts,
                answers=batch_answers,
                separator_len=separator_len,
                prepend_bos=prepend_bos,
                max_length=max_length,
                stats=stats,
            )

        results[name] = stats.to_dict()
        if print_table:
            _print_one_line(results[name])

    finished_at = time.time()
    payload = {
        "created_at_unix": started_at,
        "finished_at_unix": finished_at,
        "elapsed_sec": finished_at - started_at,
        "config": {
            "tokenizer": tokenizer_name,
            "trust_remote_code": trust_remote_code,
            "max_length": max_length,
            "separator_kind": separator_kind,
            "separator_token_id": pad_token_id,
            "separator_len": separator_len,
            "prepend_bos": prepend_bos,
            "scratch_root": scratch_root,
            "dataset_names": names,
            "limit": None if limit is None else int(limit),
            "batch_size": batch_size,
            "filters": _filters_snapshot(names),
        },
        "datasets": results,
    }

    if output_path is None:
        output_path = os.path.join(scratch_root, stats_filename)
    output_path = os.path.expanduser(output_path)
    payload["output_path"] = output_path
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    if print_table:
        print(f"\nWrote FlexMDM dataset stats to {output_path}")

    return payload


def _import_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise FlexMDMDataStatsError(
            "NumPy is required for FlexMDM pre-tokenization."
        ) from exc
    return np


def _resolve_pad_token_id(tokenizer: Any, data_config: Dict[str, Any]) -> int:
    configured = data_config.get("pad_token_id", None)
    if configured is not None:
        return int(configured)
    if getattr(tokenizer, "pad_token_id", None) is not None:
        return int(tokenizer.pad_token_id)
    raise FlexMDMDataStatsError(
        "Could not resolve pad_token_id. Set data.pad_token_id in config."
    )


def _open_shard_files(
    shard_dir: str,
    *,
    overwrite: bool,
) -> Dict[str, Any]:
    os.makedirs(shard_dir, exist_ok=True)
    paths = {
        "input_ids": os.path.join(shard_dir, "input_ids.bin"),
        "prompt_mask": os.path.join(shard_dir, "prompt_mask.bin"),
        "attention_mask": os.path.join(shard_dir, "attention_mask.bin"),
        "seq_lens": os.path.join(shard_dir, "seq_lens.bin"),
        "prompt_lens": os.path.join(shard_dir, "prompt_lens.bin"),
    }
    if not overwrite:
        existing = [path for path in paths.values() if os.path.exists(path)]
        if existing:
            joined = ", ".join(existing)
            raise FileExistsError(
                f"Tokenized shard files already exist for {shard_dir}: {joined}. "
                "Pass --overwrite to replace them."
            )
    return {name: open(path, "wb") for name, path in paths.items()}


def _close_shard_files(files: Dict[str, Any]) -> None:
    for handle in files.values():
        handle.close()


def _encode_tokenization_batch(
    *,
    tokenizer: Any,
    prompts: Sequence[str],
    answers: Sequence[str],
    prepend_bos: bool,
    bos_token_id: Optional[int],
    pad_token_id: int,
    max_length: int,
    stats: CorpusStats,
):
    np = _import_numpy()
    if not prompts:
        return None

    prompt_ids_batch = tokenizer(
        list(prompts),
        add_special_tokens=False,
    ).input_ids
    answer_ids_batch = tokenizer(
        list(answers),
        add_special_tokens=False,
    ).input_ids

    rows: List[Any] = []
    prompt_masks: List[Any] = []
    attention_masks: List[Any] = []
    seq_lens: List[int] = []
    prompt_lens: List[int] = []

    for prompt_ids, answer_ids in zip(prompt_ids_batch, answer_ids_batch):
        prefix_ids: List[int] = []
        if prepend_bos:
            if bos_token_id is None:
                raise FlexMDMDataStatsError(
                    "prepend_bos=True but tokenizer has no bos_token_id."
                )
            prefix_ids.append(int(bos_token_id))
        prefix_ids.extend(int(token_id) for token_id in prompt_ids)
        prefix_ids.append(int(pad_token_id))

        prompt_boundary_length = len(prefix_ids)
        answer_length = len(answer_ids)
        total_length = prompt_boundary_length + answer_length
        stats.add_candidate_lengths(
            prompt_boundary_length=prompt_boundary_length,
            answer_length=answer_length,
            total_length=total_length,
        )

        if answer_length == 0:
            stats.drop("empty_text")
            continue
        if prompt_boundary_length >= max_length:
            stats.zero_answer_capacity += 1
            stats.drop("prompt_too_long")
            continue

        answer_capacity = max_length - prompt_boundary_length
        kept_answer_ids = [
            int(token_id) for token_id in answer_ids[:answer_capacity]
        ]
        if not kept_answer_ids:
            stats.drop("empty_text")
            continue

        input_ids = np.full(max_length, pad_token_id, dtype=np.int32)
        prompt_mask = np.zeros(max_length, dtype=np.uint8)
        attention_mask = np.zeros(max_length, dtype=np.uint8)
        sequence_ids = prefix_ids + kept_answer_ids
        sequence_length = len(sequence_ids)

        input_ids[:sequence_length] = np.asarray(sequence_ids, dtype=np.int32)
        prompt_mask[:prompt_boundary_length] = 1
        attention_mask[:sequence_length] = 1
        rows.append(input_ids)
        prompt_masks.append(prompt_mask)
        attention_masks.append(attention_mask)
        seq_lens.append(sequence_length)
        prompt_lens.append(prompt_boundary_length)
        stats.mark_kept(total_length=total_length, max_length=max_length)

    if not rows:
        return None

    return {
        "input_ids": np.stack(rows, axis=0),
        "prompt_mask": np.stack(prompt_masks, axis=0),
        "attention_mask": np.stack(attention_masks, axis=0),
        "seq_lens": np.asarray(seq_lens, dtype=np.uint16),
        "prompt_lens": np.asarray(prompt_lens, dtype=np.uint16),
    }


def _write_tokenization_batch(
    *,
    files: Dict[str, Any],
    encoded: Optional[Dict[str, Any]],
) -> int:
    if encoded is None:
        return 0
    encoded["input_ids"].tofile(files["input_ids"])
    encoded["prompt_mask"].tofile(files["prompt_mask"])
    encoded["attention_mask"].tofile(files["attention_mask"])
    encoded["seq_lens"].tofile(files["seq_lens"])
    encoded["prompt_lens"].tofile(files["prompt_lens"])
    return int(encoded["input_ids"].shape[0])


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _build_shard_metadata(
    *,
    dataset_name: str,
    shard_dir: str,
    stats: CorpusStats,
    max_length: int,
    pad_token_id: int,
    bos_token_id: Optional[int],
    prepend_bos: bool,
    separator_kind: str,
    separator_token_id: int,
    files: Dict[str, str],
) -> Dict[str, Any]:
    payload = stats.to_dict()
    payload.update(
        {
            "shard_dir": shard_dir,
            "max_length": max_length,
            "pad_token_id": pad_token_id,
            "bos_token_id": bos_token_id,
            "prepend_bos": prepend_bos,
            "separator_kind": separator_kind,
            "separator_token_id": separator_token_id,
            "files": files,
            "dtypes": {
                "input_ids": "int32",
                "prompt_mask": "uint8",
                "attention_mask": "uint8",
                "seq_lens": "uint16",
                "prompt_lens": "uint16",
            },
            "shapes": {
                "input_ids": [stats.kept, max_length],
                "prompt_mask": [stats.kept, max_length],
                "attention_mask": [stats.kept, max_length],
                "seq_lens": [stats.kept],
                "prompt_lens": [stats.kept],
            },
        }
    )
    return payload


def _repeat_factor_hints(names: Sequence[str]) -> Dict[str, int]:
    return {
        name: 1 if name == "OpenCodeInstruct-score1-py-all" else 2
        for name in names
    }


def pretokenize_datasets(
    config: Optional[Dict[str, Any]] = None,
    tokenizer: Any = None,
    *,
    config_path: Optional[str] = None,
    dataset_names: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    output_root: Optional[str] = None,
    batch_size: Optional[int] = None,
    overwrite: bool = False,
    log_every: int = 10000,
    print_table: bool = True,
) -> Dict[str, Any]:
    """Stream, filter, tokenize, and write fixed-width binary shards."""
    if config is None:
        config = load_config(config_path or DEFAULT_CONFIG_PATH)

    names = _resolve_dataset_names(config, dataset_names)
    tokenizer_name = str(
        _config_get(config, "model", "partial_pretrain", default=DEFAULT_TOKENIZER)
    )
    trust_remote_code = bool(
        _config_get(config, "model", "trust_remote_code", default=True)
    )
    data_config = _config_get(config, "data", default={}) or {}
    scratch_root = os.path.expanduser(
        str(data_config.get("scratch_root", DEFAULT_SCRATCH_ROOT))
    )
    if output_root is None:
        output_root = str(
            data_config.get(
                "pretokenized_root",
                os.path.join(scratch_root, DEFAULT_PRETOKENIZED_DIRNAME),
            )
        )
    output_root = os.path.expanduser(output_root)
    max_length = int(data_config.get("max_length", DEFAULT_MAX_LENGTH))
    separator_kind = str(
        data_config.get("separator_kind", DEFAULT_SEPARATOR_KIND)
    )
    prepend_bos = bool(data_config.get("prepend_bos", DEFAULT_PREPEND_BOS))
    if batch_size is None:
        batch_size = int(
            data_config.get(
                "pretokenize_batch_size",
                data_config.get("stats_batch_size", 2048),
            )
        )

    if max_length <= 0:
        raise ValueError(f"max_length must be positive, got {max_length}")
    if max_length > 65535:
        raise ValueError(
            "The current pre-tokenized format stores lengths as uint16; "
            f"max_length={max_length} is too large."
        )
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if log_every < 0:
        raise ValueError(f"log_every must be nonnegative, got {log_every}")

    if tokenizer is None:
        tokenizer = _load_tokenizer(
            tokenizer_name=tokenizer_name,
            trust_remote_code=trust_remote_code,
            prepend_bos=prepend_bos,
        )
    pad_token_id = _resolve_pad_token_id(tokenizer, data_config)
    if separator_kind != DEFAULT_SEPARATOR_KIND:
        raise ValueError(
            "FlexMDM pre-tokenization currently supports only "
            f"separator_kind={DEFAULT_SEPARATOR_KIND!r}, got {separator_kind!r}."
        )
    bos_token_id = (
        int(tokenizer.bos_token_id)
        if getattr(tokenizer, "bos_token_id", None) is not None
        else None
    )
    separator_token_id = pad_token_id

    os.makedirs(output_root, exist_ok=True)
    started_at = time.time()
    results: Dict[str, Any] = {}

    if print_table:
        print(
            "dataset".ljust(34),
            "seen".rjust(10),
            "written".rjust(10),
            "retain".rjust(9),
            "trunc".rjust(9),
            "drop_quality".rjust(13),
            "drop_post".rjust(10),
            "drop_long".rjust(10),
        )
        print("-" * 111)

    for name in names:
        stats = CorpusStats(dataset_name=name)
        spec = DATASET_SPECS[name]
        dataset = _load_streaming_dataset(spec)
        shard_dir = os.path.join(output_root, name)
        shard_files = {
            "input_ids": "input_ids.bin",
            "prompt_mask": "prompt_mask.bin",
            "attention_mask": "attention_mask.bin",
            "seq_lens": "seq_lens.bin",
            "prompt_lens": "prompt_lens.bin",
        }
        files = _open_shard_files(shard_dir, overwrite=overwrite)
        batch_prompts: List[str] = []
        batch_answers: List[str] = []

        try:
            for example in dataset:
                if limit is not None and stats.seen >= int(limit):
                    break

                stats.seen += 1
                prompt, answer, drop_reason = normalize_example(name, spec, example)
                if drop_reason is not None:
                    stats.drop(drop_reason)
                    continue
                assert prompt is not None and answer is not None
                batch_prompts.append(prompt)
                batch_answers.append(answer)

                if len(batch_prompts) >= batch_size:
                    encoded = _encode_tokenization_batch(
                        tokenizer=tokenizer,
                        prompts=batch_prompts,
                        answers=batch_answers,
                        prepend_bos=prepend_bos,
                        bos_token_id=bos_token_id,
                        pad_token_id=pad_token_id,
                        max_length=max_length,
                        stats=stats,
                    )
                    _write_tokenization_batch(files=files, encoded=encoded)
                    batch_prompts.clear()
                    batch_answers.clear()

                if log_every and stats.seen % int(log_every) == 0:
                    print(
                        f"[{name}] seen={stats.seen:,} written={stats.kept:,} "
                        f"dropped={sum(stats.dropped.values()):,}",
                        flush=True,
                    )

            if batch_prompts:
                encoded = _encode_tokenization_batch(
                    tokenizer=tokenizer,
                    prompts=batch_prompts,
                    answers=batch_answers,
                    prepend_bos=prepend_bos,
                    bos_token_id=bos_token_id,
                    pad_token_id=pad_token_id,
                    max_length=max_length,
                    stats=stats,
                )
                _write_tokenization_batch(files=files, encoded=encoded)
        finally:
            _close_shard_files(files)

        metadata = _build_shard_metadata(
            dataset_name=name,
            shard_dir=shard_dir,
            stats=stats,
            max_length=max_length,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            prepend_bos=prepend_bos,
            separator_kind=separator_kind,
            separator_token_id=separator_token_id,
            files=shard_files,
        )
        metadata_path = os.path.join(shard_dir, "metadata.json")
        _write_json(metadata_path, metadata)
        results[name] = metadata
        if print_table:
            row = dict(metadata)
            row["kept"] = metadata["kept"]
            _print_one_line(row)

    finished_at = time.time()
    manifest = {
        "created_at_unix": started_at,
        "finished_at_unix": finished_at,
        "elapsed_sec": finished_at - started_at,
        "output_root": output_root,
        "config": {
            "tokenizer": tokenizer_name,
            "trust_remote_code": trust_remote_code,
            "max_length": max_length,
            "pad_token_id": pad_token_id,
            "bos_token_id": bos_token_id,
            "separator_kind": separator_kind,
            "separator_token_id": separator_token_id,
            "prepend_bos": prepend_bos,
            "dataset_names": names,
            "limit": None if limit is None else int(limit),
            "batch_size": batch_size,
            "filters": _filters_snapshot(names),
            "dataloader_repeat_factor_hint": {
                name: int(
                    (data_config.get("dataloader_repeat_factors", {}) or {}).get(
                        name,
                        _repeat_factor_hints(names)[name],
                    )
                )
                for name in names
            },
        },
        "total_written": sum(int(row["kept"]) for row in results.values()),
        "datasets": results,
    }
    manifest_path = os.path.join(output_root, DEFAULT_MANIFEST_FILENAME)
    manifest["manifest_path"] = manifest_path
    _write_json(manifest_path, manifest)
    if print_table:
        print(f"\nWrote FlexMDM pre-tokenized manifest to {manifest_path}")
    return manifest


def _filters_snapshot(names: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    snapshot = {}
    for name in names:
        spec = DATASET_SPECS.get(name)
        if spec is None:
            snapshot[name] = {"dataset_spec_missing": True}
            continue
        snapshot[name] = {
            "require_is_passed": spec.require_is_passed,
            "require_is_passed_if_present": spec.require_is_passed_if_present,
            "require_score_at_least": spec.require_score_at_least,
            "require_score_exact": spec.require_score_exact,
            "require_python_code": spec.require_python_code,
            "strip_trailing_demo": spec.strip_trailing_demo,
            "require_llm_judgement_min_score": spec.require_llm_judgement_min_score,
            "require_subset_in": list(spec.require_subset_in)
                if spec.require_subset_in is not None
                else None,
            "require_style_in": list(spec.require_style_in)
                if spec.require_style_in is not None
                else None,
            "require_gpt_pass_percentage_above": spec.require_gpt_pass_percentage_above,
            "prompt_uses_starter_code": spec.prompt_uses_starter_code,
        }
    return snapshot


def _print_table_header() -> None:
    print(
        "dataset".ljust(34),
        "seen".rjust(10),
        "kept".rjust(10),
        "retain".rjust(9),
        "trunc".rjust(9),
        "drop_quality".rjust(13),
        "drop_post".rjust(10),
        "drop_long".rjust(10),
    )
    print("-" * 111)


def _print_one_line(row: Dict[str, Any]) -> None:
    if row["seen"] == 0:
        retain = 0.0
    else:
        retain = 100.0 * row["retention_rate"]
    trunc = 100.0 * row["answer_truncation_rate"]
    drops = row["drops"]
    print(
        str(row["dataset_name"])[:34].ljust(34),
        f"{row['seen']:,}".rjust(10),
        f"{row['kept']:,}".rjust(10),
        f"{retain:8.2f}%".rjust(9),
        f"{trunc:8.2f}%".rjust(9),
        f"{drops.get('quality_filter', 0):,}".rjust(13),
        f"{drops.get('postprocess_filter', 0):,}".rjust(10),
        f"{drops.get('prompt_too_long', 0):,}".rjust(10),
    )


class FlexMDMTokenizedShardDataset:
    """Map-style dataset over one pre-tokenized binary shard."""

    ARRAY_NAMES = (
        "input_ids",
        "prompt_mask",
        "attention_mask",
        "seq_lens",
        "prompt_lens",
    )

    def __init__(self, shard_metadata: Dict[str, Any]):
        np = _import_numpy()
        self.metadata = dict(shard_metadata)
        self.dataset_name = str(self.metadata["dataset_name"])
        self.shard_dir = os.path.expanduser(str(self.metadata["shard_dir"]))
        self.files = dict(self.metadata["files"])
        self.dtypes = dict(self.metadata["dtypes"])
        self.shapes = dict(self.metadata["shapes"])
        self.length = int(self.metadata["kept"])
        self.arrays: Dict[str, Any] = {}

        for name in self.ARRAY_NAMES:
            if name not in self.files:
                raise KeyError(
                    f"Shard {self.dataset_name} metadata is missing file for {name}"
                )
            path = self.files[name]
            if not os.path.isabs(path):
                path = os.path.join(self.shard_dir, path)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Shard {self.dataset_name} is missing {name} file: {path}"
                )
            self.arrays[name] = np.memmap(
                path,
                dtype=np.dtype(self.dtypes[name]),
                mode="r",
                shape=tuple(int(dim) for dim in self.shapes[name]),
            )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Dict[str, Any]:
        np = _import_numpy()
        torch = _import_torch()
        if index < 0:
            index = self.length + index
        if index < 0 or index >= self.length:
            raise IndexError(index)

        return {
            "input_ids": torch.as_tensor(
                np.array(self.arrays["input_ids"][index], copy=True),
                dtype=torch.long,
            ),
            "prompt_mask": torch.as_tensor(
                np.array(self.arrays["prompt_mask"][index], copy=True),
                dtype=torch.bool,
            ),
            "attention_mask": torch.as_tensor(
                np.array(self.arrays["attention_mask"][index], copy=True),
                dtype=torch.bool,
            ),
            "seq_lens": torch.as_tensor(
                int(self.arrays["seq_lens"][index]),
                dtype=torch.long,
            ),
            "prompt_lens": torch.as_tensor(
                int(self.arrays["prompt_lens"][index]),
                dtype=torch.long,
            ),
            "dataset_name": self.dataset_name,
        }


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise FlexMDMDataStatsError(
            "PyTorch is required for FlexMDM dataloaders."
        ) from exc
    return torch


def _load_pretokenized_manifest(root: str) -> Dict[str, Any]:
    manifest_path = os.path.join(
        os.path.expanduser(root),
        DEFAULT_MANIFEST_FILENAME,
    )
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Could not find FlexMDM pre-tokenized manifest: {manifest_path}"
        )
    with open(manifest_path, "r") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise FlexMDMDataStatsError(
            f"Manifest {manifest_path} did not load as a dict"
        )
    return manifest


def _repeat_factor_for_dataset(data_config: Any, dataset_name: str) -> int:
    configured = _get_config_value(data_config, "dataloader_repeat_factors", {}) or {}
    if hasattr(configured, "get"):
        repeat = configured.get(dataset_name, None)
        if repeat is not None:
            return max(1, int(repeat))
    return 1 if dataset_name == "OpenCodeInstruct-score1-py-all" else 2


def _resolve_batch_size(
    data_config: Any,
    *,
    key: str,
    fallback_key: str,
    rank: int,
    world_size: int,
) -> int:
    value = _get_config_value(data_config, key, None)
    if value is None:
        value = _get_config_value(data_config, fallback_key, 1)
    batch_size = max(1, int(value))
    if bool(_get_config_value(data_config, "train_batch_size_is_global", True)):
        per_rank = max(1, batch_size // max(1, int(world_size)))
        if rank == 0 and batch_size % max(1, int(world_size)) != 0:
            print(
                f"FlexMDM dataloader: {key}={batch_size} is not divisible by "
                f"world_size={world_size}; using per-rank batch size {per_rank}.",
                flush=True,
            )
        return per_rank
    return batch_size


def _build_pretokenized_dataset(
    *,
    root: str,
    data_config: Any,
    dataset_names: Optional[Sequence[str]] = None,
    use_repeat_factors: bool,
):
    torch = _import_torch()
    from torch.utils.data import ConcatDataset

    manifest = _load_pretokenized_manifest(root)
    manifest_datasets = manifest.get("datasets", {})
    if not isinstance(manifest_datasets, dict):
        raise FlexMDMDataStatsError("Manifest `datasets` field must be a dict")

    if dataset_names is None:
        configured_names = _get_config_value(data_config, "dataset_names", None)
        if configured_names is None:
            names = list(manifest_datasets.keys())
        elif isinstance(configured_names, str):
            names = [configured_names]
        else:
            names = [str(name) for name in configured_names]
    else:
        names = [str(name) for name in dataset_names]

    datasets = []
    for name in names:
        if name not in manifest_datasets:
            raise KeyError(
                f"Dataset {name!r} is not present in manifest {root}. "
                f"Available: {sorted(manifest_datasets)}"
            )
        shard = FlexMDMTokenizedShardDataset(manifest_datasets[name])
        if len(shard) == 0:
            continue
        repeat = _repeat_factor_for_dataset(data_config, name) if use_repeat_factors else 1
        datasets.extend([shard] * repeat)

    if not datasets:
        raise FlexMDMDataStatsError(
            f"No non-empty pre-tokenized datasets found under {root}"
        )
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def _build_dataloader(
    *,
    dataset: Any,
    data_config: Any,
    batch_size: int,
    rank: int,
    world_size: int,
    shuffle: bool,
):
    torch = _import_torch()
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    sampler = None
    if int(world_size) > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=int(world_size),
            rank=int(rank),
            shuffle=shuffle,
            drop_last=bool(_get_config_value(data_config, "drop_last", True)),
        )

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=int(_get_config_value(data_config, "num_workers", 0)),
        pin_memory=bool(_get_config_value(data_config, "pin_memory", True)),
        drop_last=bool(_get_config_value(data_config, "drop_last", True))
        and sampler is None,
        persistent_workers=bool(
            _get_config_value(data_config, "persistent_workers", False)
        )
        and int(_get_config_value(data_config, "num_workers", 0)) > 0,
    )


def prepare_flexmdm_data(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Collect streaming stats for the current FlexMDM data contract."""
    return collect_dataset_stats(*args, **kwargs)


def _resolve_train_pretokenized_root(data_config: Any) -> str:
    return os.path.expanduser(
        str(
            _get_config_value(
                data_config,
                "pretokenized_root",
                os.path.join(
                    str(_get_config_value(data_config, "scratch_root", DEFAULT_SCRATCH_ROOT)),
                    DEFAULT_PRETOKENIZED_DIRNAME,
                ),
            )
        )
    )


def _holdout_split(dataset: Any, *, validation_fraction: float, seed: int):
    """Return (train_subset, val_subset) via a deterministic permutation.

    The permutation is seeded identically on every rank so train/val
    membership is consistent across processes.
    """
    torch = _import_torch()
    from torch.utils.data import Subset

    total = len(dataset)
    if total <= 0:
        return dataset, None
    val_count = int(round(total * float(validation_fraction)))
    val_count = max(1, min(val_count, total - 1))

    gen = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(total, generator=gen).tolist()
    val_indices = perm[:val_count]
    train_indices = perm[val_count:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def build_train_dataloader(
    *,
    config: Any,
    tokenizer: Any = None,
    rank: int = 0,
    world_size: int = 1,
):
    """Build the FlexMDM train dataloader from pre-tokenized binary shards."""
    del tokenizer
    data_config = _config_get(config, "data", default={}) or {}
    root = _resolve_train_pretokenized_root(data_config)
    dataset = _build_pretokenized_dataset(
        root=root,
        data_config=data_config,
        use_repeat_factors=True,
    )

    # If a separate validation root is configured we don't split the training
    # set. Otherwise hold out ``validation_fraction`` deterministically.
    if _get_config_value(data_config, "validation_pretokenized_root", None) is None:
        validation_fraction = float(
            _get_config_value(data_config, "validation_fraction", 0.0) or 0.0
        )
        if validation_fraction > 0.0:
            split_seed_value = _get_config_value(
                data_config,
                "validation_seed",
                None,
            )
            if split_seed_value is None:
                split_seed_value = _config_get(
                    config,
                    "trainer",
                    "seed",
                    default=0,
                )
            split_seed = int(split_seed_value or 0)
            dataset, _val = _holdout_split(
                dataset,
                validation_fraction=validation_fraction,
                seed=split_seed,
            )

    batch_size = _resolve_batch_size(
        data_config,
        key="train_batch_size",
        fallback_key="micro_batch_size_per_gpu",
        rank=rank,
        world_size=world_size,
    )
    return _build_dataloader(
        dataset=dataset,
        data_config=data_config,
        batch_size=batch_size,
        rank=rank,
        world_size=world_size,
        shuffle=True,
    )


def build_validation_dataloader(
    *,
    config: Any,
    tokenizer: Any = None,
    rank: int = 0,
    world_size: int = 1,
):
    """Build the FlexMDM validation dataloader.

    Priority:
      1. If ``data.validation_pretokenized_root`` is set, build from that root.
      2. Otherwise, if ``data.validation_fraction > 0``, hold out that
         fraction of the training concat dataset with a deterministic
         permutation (same seed across ranks).
      3. Otherwise return ``None``.
    """
    del tokenizer
    data_config = _config_get(config, "data", default={}) or {}
    root = _get_config_value(data_config, "validation_pretokenized_root", None)
    if root is not None:
        dataset_names = _get_config_value(data_config, "validation_dataset_names", None)
        dataset = _build_pretokenized_dataset(
            root=os.path.expanduser(str(root)),
            data_config=data_config,
            dataset_names=dataset_names,
            use_repeat_factors=False,
        )
    else:
        validation_fraction = float(
            _get_config_value(data_config, "validation_fraction", 0.0) or 0.0
        )
        if validation_fraction <= 0.0:
            return None
        train_root = _resolve_train_pretokenized_root(data_config)
        full_dataset = _build_pretokenized_dataset(
            root=train_root,
            data_config=data_config,
            use_repeat_factors=True,
        )
        split_seed_value = _get_config_value(
            data_config,
            "validation_seed",
            None,
        )
        if split_seed_value is None:
            split_seed_value = _config_get(
                config,
                "trainer",
                "seed",
                default=0,
            )
        split_seed = int(split_seed_value or 0)
        _train, dataset = _holdout_split(
            full_dataset,
            validation_fraction=validation_fraction,
            seed=split_seed,
        )
        if dataset is None:
            return None

    batch_size = _resolve_batch_size(
        data_config,
        key="validation_batch_size",
        fallback_key="train_batch_size",
        rank=rank,
        world_size=world_size,
    )
    return _build_dataloader(
        dataset=dataset,
        data_config=data_config,
        batch_size=batch_size,
        rank=rank,
        world_size=world_size,
        shuffle=False,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FlexMDM data utilities")
    subparsers = parser.add_subparsers(dest="command")

    stats = subparsers.add_parser(
        "stats",
        help="Stream datasets and collect filtering/token length stats.",
    )
    stats.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    stats.add_argument("--limit", type=int, default=None)
    stats.add_argument("--output", default=None)
    stats.add_argument("--datasets", nargs="+", default=None)
    stats.add_argument("--batch-size", type=int, default=None)
    stats.add_argument("--quiet", action="store_true")

    tokenize = subparsers.add_parser(
        "tokenize",
        help="Stream datasets and write fixed-width tokenized binary shards.",
    )
    tokenize.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    tokenize.add_argument("--limit", type=int, default=None)
    tokenize.add_argument("--output-root", default=None)
    tokenize.add_argument("--datasets", nargs="+", default=None)
    tokenize.add_argument("--batch-size", type=int, default=None)
    tokenize.add_argument("--overwrite", action="store_true")
    tokenize.add_argument("--log-every", type=int, default=10000)
    tokenize.add_argument("--quiet", action="store_true")

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        raise SystemExit(0)
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.command == "stats":
        if not args.quiet:
            _print_table_header()
        collect_dataset_stats(
            config_path=args.config,
            dataset_names=args.datasets,
            limit=args.limit,
            output_path=args.output,
            batch_size=args.batch_size,
            print_table=not args.quiet,
        )
        return

    if args.command == "tokenize":
        pretokenize_datasets(
            config_path=args.config,
            dataset_names=args.datasets,
            limit=args.limit,
            output_root=args.output_root,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
            log_every=args.log_every,
            print_table=not args.quiet,
        )
        return

    raise SystemExit(f"Unknown FlexMDM data command {args.command}")


if __name__ == "__main__":
    main()


__all__ = [
    "DATASET_SPECS",
    "DEFAULT_DATASET_NAMES",
    "FlexMDMDataStatsError",
    "FlexMDMTokenizedShardDataset",
    "build_train_dataloader",
    "build_validation_dataloader",
    "collect_dataset_stats",
    "extract_code_only_prefer_python",
    "is_python_like",
    "normalize_example",
    "strip_trailing_demo_block",
    "prepare_flexmdm_data",
    "pretokenize_datasets",
    "safe_float",
    "truthy",
]
