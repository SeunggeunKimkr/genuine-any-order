"""AST tree, token mapping, and any-order metric computation.

This module keeps only the analysis routines from the original visualization
script. The HTML rendering helpers (``write_sample_html``,
``write_unparseable_html``, code/tree/SVG renderers, the standalone CLI
``run`` / ``parse_args`` / ``main`` entry points) are intentionally dropped:
``flexmdm/utils.py:write_trace_html`` already handles HTML viewing of our
``.pt`` records, and we only need the analysis pieces here.

The core analysis pipeline kept here is:

  1. ``parse_with_fallbacks(source)`` parses the model's final source
     (with a fenced-code-block fallback) into an ``ast.Module``.
  2. ``ast_to_record_tree`` flattens the AST to a list of record dicts and
     a parent map.
  3. ``build_visualization_tree`` produces the simplified "viz tree" we
     score over (collapses docstrings into their owners, splits off a
     synthetic ``Generated Tests`` node when the model emitted top-level
     example asserts after the function definition).
  4. ``tokenize_source`` tokenizes the parsed source via the std-lib
     ``tokenize`` module; ``map_tokens_to_nodes`` aligns each token to
     the deepest viz node whose source span contains it.
  5. ``compute_tree_order_metrics_variant`` consumes a per-step
     ``generation_progress`` (the structure produced by
     ``compute_metrics.py`` from our ``.pt`` records) and produces the
     CBC / RUB / OBW per non-leaf viz node.
  6. ``aggregate_tree_order_metrics`` averages the per-node scores over
     all non-leaf nodes (``overall``) and over split nodes only
     (``split_only``), and emits a per-depth breakdown.

Both metric variants ("reference-aware" — keep ``Generated Tests`` /
docstring nodes — and "code-only" — drop them) are supported via the
``include_reference_nodes`` flag on
``compute_tree_order_metrics_variant``.
"""

from __future__ import annotations

import ast
import difflib
import io
import re
import tokenize
import token as token_mod
from typing import Any, Dict, List, Optional, Sequence, Tuple


VISUAL_STATEMENT_TYPES = {
    "FunctionDef",
    "AsyncFunctionDef",
    "ClassDef",
    "For",
    "AsyncFor",
    "While",
    "If",
    "Try",
    "With",
    "AsyncWith",
    "Assign",
    "AnnAssign",
    "AugAssign",
    "Return",
    "Expr",
    "Import",
    "ImportFrom",
    "Raise",
    "Assert",
    "Break",
    "Continue",
    "Pass",
    "Delete",
    "Global",
    "Nonlocal",
    "Match",
}

CONTROL_FLOW_TYPES = {
    "For", "AsyncFor", "While", "If", "Try", "With", "AsyncWith", "Match",
}


REFERENCE_NODE_TYPES = {
    "Generated Tests",
    "Reference/Test Cases",
    "Generated Tests / Examples",
    "Examples",
}


# ---------------------------------------------------------------------------
# Source / answer normalization
# ---------------------------------------------------------------------------


def clean_generated_answer(text: str) -> str:
    """Remove trace/model markers while preserving generated Python layout."""

    if text is None:
        return ""
    cleaned = text.replace("<|beginoftext|>", "")
    cleaned = cleaned.replace("<|endoftext|>", "")
    cleaned = cleaned.replace("[M]", "")
    return cleaned.strip()


def clean_snapshot_answer_for_alignment(text: str) -> str:
    """Remove boundary markers while preserving mask markers for snapshot metadata."""

    if text is None:
        return ""
    cleaned = text.replace("<|beginoftext|>", "")
    cleaned = cleaned.replace("<|endoftext|>", "")
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Source position helpers
# ---------------------------------------------------------------------------


def line_start_offsets(source: str) -> List[int]:
    offsets = [0]
    total = 0
    for line in source.splitlines(True):
        total += len(line)
        offsets.append(total)
    if not source.endswith(("\n", "\r")):
        offsets.append(len(source))
    return offsets


def line_texts(source: str) -> List[str]:
    lines = source.splitlines(True)
    if not lines:
        return [""]
    return lines


def byte_col_to_char_col(line_text: str, byte_col: int) -> int:
    """AST columns are UTF-8 byte offsets; convert them to Python char columns."""

    if byte_col <= 0:
        return 0
    raw = line_text.encode("utf-8")
    prefix = raw[: min(byte_col, len(raw))]
    return len(prefix.decode("utf-8", errors="ignore"))


def position_to_offset(
    starts: Sequence[int],
    lines: Sequence[str],
    line: Optional[int],
    col: Optional[int],
    *,
    ast_col: bool = False,
) -> Optional[int]:
    if line is None or col is None:
        return None
    if line <= 0:
        return 0
    if line - 1 >= len(starts):
        return starts[-1]
    char_col = col
    if ast_col and line - 1 < len(lines):
        char_col = byte_col_to_char_col(lines[line - 1], col)
    return min(starts[line - 1] + char_col, starts[-1])


def span_text(source: str, start: Optional[int], end: Optional[int]) -> str:
    if start is None or end is None:
        return ""
    return source[max(0, start) : min(len(source), end)]


def compact_snippet(text: str, limit: int = 120) -> str:
    snippet = " ".join(text.strip().split())
    if len(snippet) > limit:
        return snippet[: limit - 1].rstrip() + "..."
    return snippet


def unparse_short(node: ast.AST, limit: int = 64) -> str:
    try:
        text = ast.unparse(node)
    except Exception:
        return ""
    return compact_snippet(text, limit)


# ---------------------------------------------------------------------------
# AST -> record tree
# ---------------------------------------------------------------------------


def ast_label(node: ast.AST) -> str:
    typ = type(node).__name__
    lineno = getattr(node, "lineno", None)
    suffix = f"@L{lineno}" if lineno is not None else ""

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return f"{typ}: {node.name}"
    if isinstance(node, ast.ClassDef):
        return f"ClassDef: {node.name}"
    if isinstance(node, ast.Call):
        return f"Call: {unparse_short(node.func, 48) or suffix}"
    if isinstance(node, ast.Assign):
        target = unparse_short(node.targets[0], 48) if node.targets else ""
        return f"Assign: {target}" if target else f"Assign{suffix}"
    if isinstance(node, ast.AnnAssign):
        target = unparse_short(node.target, 48)
        return f"AnnAssign: {target}" if target else f"AnnAssign{suffix}"
    if isinstance(node, ast.AugAssign):
        target = unparse_short(node.target, 48)
        op_name = type(node.op).__name__
        return f"AugAssign: {target} {op_name}" if target else f"AugAssign{suffix}"
    if isinstance(node, ast.Return):
        return f"Return{suffix}"
    if isinstance(node, ast.If):
        test = unparse_short(node.test, 48)
        return f"If: {test}" if test else f"If{suffix}"
    if isinstance(node, (ast.For, ast.AsyncFor)):
        target = unparse_short(node.target, 36)
        iter_text = unparse_short(node.iter, 36)
        return (
            f"{typ}: {target} in {iter_text}"
            if target or iter_text
            else f"{typ}{suffix}"
        )
    if isinstance(node, ast.While):
        test = unparse_short(node.test, 48)
        return f"While: {test}" if test else f"While{suffix}"
    if isinstance(node, (ast.With, ast.AsyncWith)):
        return f"{typ}{suffix}"
    if isinstance(node, ast.Try):
        return f"Try{suffix}"
    if isinstance(node, ast.Expr):
        value = unparse_short(node.value, 56)
        return f"Expr: {value}" if value else f"Expr{suffix}"
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return unparse_short(node, 72) or f"{typ}{suffix}"
    if isinstance(node, ast.Assert):
        return f"Assert{suffix}"
    if isinstance(node, ast.Raise):
        return f"Raise{suffix}"
    if isinstance(node, ast.Name):
        return f"Name: {node.id}"
    if isinstance(node, ast.Constant):
        value = repr(node.value)
        if len(value) > 48:
            value = value[:47] + "..."
        return f"Constant: {value}"
    if isinstance(node, ast.Module):
        return "Module"
    return f"{typ}{suffix}"


def ast_to_record_tree(
    tree: ast.AST, source: str
) -> Tuple[List[Dict[str, Any]], Dict[int, int], Dict[int, ast.AST], Dict[int, Optional[int]]]:
    starts = line_start_offsets(source)
    lines = line_texts(source)
    records: List[Dict[str, Any]] = []
    ast_obj_to_id: Dict[int, int] = {}
    id_to_ast_obj: Dict[int, ast.AST] = {}
    ast_parent: Dict[int, Optional[int]] = {}

    def visit(node: ast.AST, parent_id: Optional[int], depth: int) -> int:
        node_id = len(records)
        ast_obj_to_id[id(node)] = node_id
        id_to_ast_obj[node_id] = node
        ast_parent[node_id] = parent_id

        lineno = getattr(node, "lineno", None)
        col = getattr(node, "col_offset", None)
        end_lineno = getattr(node, "end_lineno", None)
        end_col = getattr(node, "end_col_offset", None)

        if isinstance(node, ast.Module):
            lineno = 1 if source else None
            col = 0 if source else None
            end_lineno = (
                source.count("\n") + (0 if source.endswith("\n") else 1)
                if source
                else None
            )
            end_col = len(lines[-1].rstrip("\r\n")) if source else None
            char_start = 0
            char_end = len(source)
        else:
            char_start = position_to_offset(starts, lines, lineno, col, ast_col=True)
            char_end = position_to_offset(
                starts, lines, end_lineno, end_col, ast_col=True
            )

        record: Dict[str, Any] = {
            "node_id": node_id,
            "type": type(node).__name__,
            "label": ast_label(node),
            "parent_id": parent_id,
            "children": [],
            "depth": depth,
            "lineno": lineno,
            "col_offset": col,
            "end_lineno": end_lineno,
            "end_col_offset": end_col,
            "char_start": char_start,
            "char_end": char_end,
            "source_snippet": compact_snippet(span_text(source, char_start, char_end)),
            "token_indices": [],
            "direct_token_indices": [],
            "subtree_token_indices": [],
        }
        records.append(record)

        for child in ast.iter_child_nodes(node):
            child_id = visit(child, node_id, depth + 1)
            record["children"].append(child_id)
        return node_id

    visit(tree, None, 0)
    return records, ast_obj_to_id, id_to_ast_obj, ast_parent


def nested_tree_from_records(
    records: Sequence[Dict[str, Any]], root_id: int = 0
) -> Dict[str, Any]:
    rec = records[root_id]
    return {
        "node_id": rec["node_id"],
        "type": rec["type"],
        "label": rec["label"],
        "lineno": rec["lineno"],
        "end_lineno": rec["end_lineno"],
        "children": [
            nested_tree_from_records(records, child_id) for child_id in rec["children"]
        ],
    }


# ---------------------------------------------------------------------------
# Visualization tree (collapses docstrings, splits off generated-test nodes)
# ---------------------------------------------------------------------------


def is_docstring_stmt(stmt: ast.AST) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def all_descendant_ast_ids(
    ast_records: Sequence[Dict[str, Any]], ast_id: int
) -> List[int]:
    out = [ast_id]
    stack = list(ast_records[ast_id]["children"])
    while stack:
        node_id = stack.pop()
        out.append(node_id)
        stack.extend(ast_records[node_id]["children"])
    return out


def has_top_level_test_marker(stmt: ast.AST) -> bool:
    if isinstance(stmt, ast.Assert):
        return True
    if isinstance(stmt, ast.If):
        test_text = unparse_short(stmt.test, 120)
        if "__name__" in test_text and "__main__" in test_text:
            return True
        return any(
            has_top_level_test_marker(child) for child in stmt.body + stmt.orelse
        )
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        return True
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
        return False
    return False


def detect_generated_test_start(
    body: Sequence[ast.stmt],
) -> Tuple[Optional[int], List[str]]:
    seen_function_or_class = False
    reasons: List[str] = []
    for idx, stmt in enumerate(body):
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            seen_function_or_class = True
            continue
        if not seen_function_or_class:
            continue
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            continue
        tail = body[idx:]
        if any(has_top_level_test_marker(tail_stmt) for tail_stmt in tail):
            if any(isinstance(tail_stmt, ast.Assert) for tail_stmt in tail):
                reasons.append("top-level assert after function/class")
            if any(
                isinstance(tail_stmt, ast.Expr)
                and isinstance(tail_stmt.value, ast.Call)
                for tail_stmt in tail
            ):
                reasons.append("top-level call/print after function/class")
            if any(
                isinstance(tail_stmt, ast.If)
                and "__name__" in unparse_short(tail_stmt.test, 120)
                and "__main__" in unparse_short(tail_stmt.test, 120)
                for tail_stmt in tail
            ):
                reasons.append("__main__ guard after function/class")
            return idx, sorted(set(reasons)) or [
                "top-level examples after function/class"
            ]
    return None, []


def build_visualization_tree(
    tree: ast.Module,
    source: str,
    ast_records: List[Dict[str, Any]],
    ast_obj_to_id: Dict[int, int],
) -> Tuple[List[Dict[str, Any]], Dict[int, int], Dict[str, Any], Dict[str, Any]]:
    viz_nodes: List[Dict[str, Any]] = []
    ast_to_viz: Dict[int, int] = {}
    docstring_records: List[Dict[str, Any]] = []

    def create_viz_node(
        *,
        node_type: str,
        label: str,
        parent_id: Optional[int],
        ast_node_id: Optional[int],
        ast_node_ids: Optional[List[int]] = None,
        synthetic: bool = False,
        line_span: Optional[Tuple[Optional[int], Optional[int]]] = None,
        char_span: Optional[Tuple[Optional[int], Optional[int]]] = None,
    ) -> int:
        node_id = len(viz_nodes)
        if ast_node_id is not None:
            ast_rec = ast_records[ast_node_id]
            lineno = ast_rec["lineno"]
            end_lineno = ast_rec["end_lineno"]
            char_start = ast_rec["char_start"]
            char_end = ast_rec["char_end"]
        else:
            lineno = line_span[0] if line_span else None
            end_lineno = line_span[1] if line_span else None
            char_start = char_span[0] if char_span else None
            char_end = char_span[1] if char_span else None

        node = {
            "viz_node_id": node_id,
            "type": node_type,
            "label": label,
            "parent_id": parent_id,
            "children": [],
            "depth": 0 if parent_id is None else viz_nodes[parent_id]["depth"] + 1,
            "ast_node_id": ast_node_id,
            "ast_node_ids": ast_node_ids
            or ([] if ast_node_id is None else [ast_node_id]),
            "synthetic": synthetic,
            "lineno": lineno,
            "end_lineno": end_lineno,
            "char_start": char_start,
            "char_end": char_end,
            "source_snippet": compact_snippet(span_text(source, char_start, char_end)),
            "token_indices": [],
            "direct_token_indices": [],
            "subtree_token_indices": [],
            "docstring_collapsed": False,
        }
        viz_nodes.append(node)
        if parent_id is not None:
            viz_nodes[parent_id]["children"].append(node_id)
        if ast_node_id is not None:
            ast_to_viz[ast_node_id] = node_id
        return node_id

    def register_docstring(owner_node: ast.AST, owner_viz_id: int) -> Optional[int]:
        body = getattr(owner_node, "body", None)
        if not body or not is_docstring_stmt(body[0]):
            return None
        stmt = body[0]
        stmt_id = ast_obj_to_id[id(stmt)]
        value_id = ast_obj_to_id.get(id(stmt.value))
        rec = ast_records[stmt_id]
        docstring_records.append(
            {
                "owner_ast_node_id": ast_obj_to_id[id(owner_node)],
                "owner_viz_node_id": owner_viz_id,
                "docstring_ast_node_id": stmt_id,
                "docstring_value_ast_node_id": value_id,
                "line_span": [rec["lineno"], rec["end_lineno"]],
                "char_span": [rec["char_start"], rec["char_end"]],
                "action": "collapsed_into_owner_visualization_node",
                "text_preview": compact_snippet(
                    ast.get_docstring(owner_node, clean=False) or "", 160
                ),
            }
        )
        viz_nodes[owner_viz_id]["docstring_collapsed"] = True
        return stmt_id

    def map_ast_subtree_to_viz(ast_id: int, viz_id: int) -> None:
        for descendant_id in all_descendant_ast_ids(ast_records, ast_id):
            ast_to_viz[descendant_id] = viz_id

    def create_generated_tests_node(
        stmts: Sequence[ast.stmt],
        parent_viz_id: int,
        reasons: Sequence[str],
    ) -> int:
        ast_ids = [ast_obj_to_id[id(stmt)] for stmt in stmts]
        char_starts = [ast_records[node_id]["char_start"] for node_id in ast_ids]
        char_ends = [ast_records[node_id]["char_end"] for node_id in ast_ids]
        line_starts = [ast_records[node_id]["lineno"] for node_id in ast_ids]
        line_ends = [ast_records[node_id]["end_lineno"] for node_id in ast_ids]
        char_start = min(value for value in char_starts if value is not None)
        char_end = max(value for value in char_ends if value is not None)
        line_start = min(value for value in line_starts if value is not None)
        line_end = max(value for value in line_ends if value is not None)
        viz_id = create_viz_node(
            node_type="Generated Tests",
            label="Generated Tests / Examples",
            parent_id=parent_viz_id,
            ast_node_id=None,
            ast_node_ids=ast_ids,
            synthetic=True,
            line_span=(line_start, line_end),
            char_span=(char_start, char_end),
        )
        viz_nodes[viz_id]["detection_reasons"] = list(reasons)
        for ast_id in ast_ids:
            map_ast_subtree_to_viz(ast_id, viz_id)
        return viz_id

    def process_body(
        owner_node: ast.AST, parent_viz_id: int, *, top_level: bool = False
    ) -> None:
        body = list(getattr(owner_node, "body", []) or [])
        docstring_stmt_id = register_docstring(owner_node, parent_viz_id)

        start_idx = 1 if docstring_stmt_id is not None else 0
        if top_level:
            test_start, reasons = detect_generated_test_start(body[start_idx:])
            if test_start is not None:
                test_start += start_idx
            else:
                reasons = []
        else:
            test_start, reasons = None, []

        idx = start_idx
        while idx < len(body):
            if test_start is not None and idx == test_start:
                create_generated_tests_node(body[idx:], parent_viz_id, reasons)
                break
            process_statement(body[idx], parent_viz_id)
            idx += 1

    def process_statement(stmt: ast.stmt, parent_viz_id: int) -> None:
        ast_id = ast_obj_to_id[id(stmt)]
        node_type = type(stmt).__name__
        if node_type not in VISUAL_STATEMENT_TYPES:
            node_type = type(stmt).__name__
        viz_id = create_viz_node(
            node_type=node_type,
            label=ast_records[ast_id]["label"],
            parent_id=parent_viz_id,
            ast_node_id=ast_id,
        )

        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            process_body(stmt, viz_id)
        elif isinstance(stmt, (ast.For, ast.AsyncFor, ast.While, ast.If)):
            for child in getattr(stmt, "body", []) or []:
                if not is_docstring_stmt(child):
                    process_statement(child, viz_id)
            for child in getattr(stmt, "orelse", []) or []:
                process_statement(child, viz_id)
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            for child in stmt.body:
                process_statement(child, viz_id)
        elif isinstance(stmt, ast.Try):
            for child in stmt.body:
                process_statement(child, viz_id)
            for handler in stmt.handlers:
                for child in handler.body:
                    process_statement(child, viz_id)
            for child in stmt.orelse:
                process_statement(child, viz_id)
            for child in stmt.finalbody:
                process_statement(child, viz_id)
        elif hasattr(ast, "Match") and isinstance(stmt, ast.Match):
            for case in stmt.cases:
                for child in case.body:
                    process_statement(child, viz_id)

    module_ast_id = ast_obj_to_id[id(tree)]
    module_viz_id = create_viz_node(
        node_type="Module",
        label="Module",
        parent_id=None,
        ast_node_id=module_ast_id,
    )
    process_body(tree, module_viz_id, top_level=True)

    docstring_info = {
        "docstring_collapsed": bool(docstring_records),
        "records": docstring_records,
    }
    generated_test_nodes = [
        node for node in viz_nodes if node["type"] == "Generated Tests"
    ]
    test_info = {
        "has_generated_tests_or_examples": bool(generated_test_nodes),
        "nodes": [
            {
                "viz_node_id": node["viz_node_id"],
                "line_span": [node["lineno"], node["end_lineno"]],
                "ast_node_ids": node["ast_node_ids"],
                "detection_reasons": node.get("detection_reasons", []),
            }
            for node in generated_test_nodes
        ],
    }

    return viz_nodes, ast_to_viz, docstring_info, test_info


# ---------------------------------------------------------------------------
# Tokenization (std-lib `tokenize`) and span -> node mapping
# ---------------------------------------------------------------------------


def tokenize_source(
    source: str, starts: Sequence[int], lines: Sequence[str]
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    tokens: List[Dict[str, Any]] = []
    try:
        generated = tokenize.generate_tokens(io.StringIO(source).readline)
        for tok in generated:
            tok_type = tok.type
            if tok_type in {token_mod.ENCODING, token_mod.ENDMARKER}:
                continue
            start = position_to_offset(starts, lines, tok.start[0], tok.start[1])
            end = position_to_offset(starts, lines, tok.end[0], tok.end[1])
            token_index = len(tokens)
            tokens.append(
                {
                    "token_index": token_index,
                    "string": tok.string,
                    "type": token_mod.tok_name.get(tok_type, str(tok_type)),
                    "start_line": tok.start[0],
                    "start_col": tok.start[1],
                    "end_line": tok.end[0],
                    "end_col": tok.end[1],
                    "char_start": start,
                    "char_end": end,
                    "line": tok.start[0],
                    "deep_node_id": None,
                    "viz_node_id": None,
                    "unmapped": True,
                }
            )
    except tokenize.TokenError as exc:
        return tokens, str(exc)
    except IndentationError as exc:
        return tokens, str(exc)
    return tokens, None


def spans_intersect(
    token_start: Optional[int],
    token_end: Optional[int],
    node_start: Optional[int],
    node_end: Optional[int],
) -> bool:
    if (
        token_start is None
        or token_end is None
        or node_start is None
        or node_end is None
    ):
        return False
    if node_end < node_start:
        return False
    if token_end <= token_start:
        return node_start <= token_start <= node_end
    return max(token_start, node_start) < min(token_end, node_end)


def nearest_visual_node(
    ast_id: Optional[int],
    ast_parent: Dict[int, Optional[int]],
    ast_to_viz: Dict[int, int],
) -> Optional[int]:
    current = ast_id
    while current is not None:
        if current in ast_to_viz:
            return ast_to_viz[current]
        current = ast_parent.get(current)
    return None


def map_tokens_to_nodes(
    tokens: List[Dict[str, Any]],
    ast_records: List[Dict[str, Any]],
    viz_nodes: List[Dict[str, Any]],
    ast_parent: Dict[int, Optional[int]],
    ast_to_viz: Dict[int, int],
) -> Tuple[Dict[str, Optional[int]], Dict[str, Optional[int]]]:
    loc_nodes = [
        rec
        for rec in ast_records
        if rec["type"] != "Module"
        and rec["char_start"] is not None
        and rec["char_end"] is not None
    ]

    token_to_deep: Dict[str, Optional[int]] = {}
    token_to_viz: Dict[str, Optional[int]] = {}
    viz_by_id = {node["viz_node_id"]: node for node in viz_nodes}

    for token in tokens:
        candidates = [
            rec
            for rec in loc_nodes
            if spans_intersect(
                token["char_start"], token["char_end"],
                rec["char_start"], rec["char_end"],
            )
        ]
        if candidates:
            candidates.sort(
                key=lambda rec: (
                    rec["depth"],
                    -((rec["char_end"] or 0) - (rec["char_start"] or 0)),
                ),
                reverse=True,
            )
            deep_id: Optional[int] = candidates[0]["node_id"]
        else:
            deep_id = None

        viz_id = nearest_visual_node(deep_id, ast_parent, ast_to_viz)
        token["deep_node_id"] = deep_id
        token["viz_node_id"] = viz_id
        token["unmapped"] = deep_id is None
        token_to_deep[str(token["token_index"])] = deep_id
        token_to_viz[str(token["token_index"])] = viz_id

        token_index = token["token_index"]
        if deep_id is not None:
            ast_records[deep_id]["direct_token_indices"].append(token_index)
            current = deep_id
            while current is not None:
                ast_records[current]["subtree_token_indices"].append(token_index)
                current = ast_parent.get(current)
        if viz_id is not None:
            viz_by_id[viz_id]["direct_token_indices"].append(token_index)
            current_viz = viz_id
            while current_viz is not None:
                viz_by_id[current_viz]["subtree_token_indices"].append(token_index)
                current_viz = viz_by_id[current_viz]["parent_id"]

    for rec in ast_records:
        rec["direct_token_indices"] = sorted(set(rec["direct_token_indices"]))
        rec["subtree_token_indices"] = sorted(set(rec["subtree_token_indices"]))
        rec["token_indices"] = rec["subtree_token_indices"]
        rec["token_count"] = len(rec["token_indices"])
    for node in viz_nodes:
        node["direct_token_indices"] = sorted(set(node["direct_token_indices"]))
        node["subtree_token_indices"] = sorted(set(node["subtree_token_indices"]))
        node["token_indices"] = node["direct_token_indices"]
        node["token_count"] = len(node["subtree_token_indices"])

    return token_to_deep, token_to_viz


def countable_token_indices(tokens: Sequence[Dict[str, Any]]) -> List[int]:
    return [
        token["token_index"]
        for token in tokens
        if token.get("char_start") is not None
        and token.get("char_end") is not None
        and token["char_end"] > token["char_start"]
    ]


# ---------------------------------------------------------------------------
# Snapshot -> visible-token alignment (used by the HTML pipeline; kept
# here only for ``analyze_sample`` callers that pass per-step decoded
# answer strings — our ``.pt`` pipeline uses position-level lineage
# tracking instead, in ``compute_metrics.py``.)
# ---------------------------------------------------------------------------


def visible_final_tokens_from_snapshot(
    final_source: str,
    tokens: Sequence[Dict[str, Any]],
    snapshot_answer: str,
    *,
    token_threshold: float = 0.80,
) -> Tuple[List[int], int, float]:
    snapshot_text = clean_snapshot_answer_for_alignment(snapshot_answer)
    known_text = snapshot_text.replace("[M]", "")
    if not final_source or not known_text:
        return [], 0, 0.0

    covered = bytearray(len(final_source))
    matcher = difflib.SequenceMatcher(None, final_source, known_text, autojunk=False)
    for block in matcher.get_matching_blocks():
        if block.size <= 0:
            continue
        covered[block.a : block.a + block.size] = b"\x01" * block.size

    visible_tokens: List[int] = []
    for token in tokens:
        start = token.get("char_start")
        end = token.get("char_end")
        if start is None or end is None or end <= start:
            continue
        token_len = end - start
        covered_count = sum(covered[start:end])
        if token_len and covered_count / token_len >= token_threshold:
            visible_tokens.append(token["token_index"])

    covered_chars = int(sum(covered))
    return visible_tokens, covered_chars, covered_chars / max(1, len(final_source))


# ---------------------------------------------------------------------------
# Source parsing (with fenced-code-block fallback)
# ---------------------------------------------------------------------------


def extract_fenced_python(text: str) -> Optional[str]:
    fence_pattern = re.compile(
        r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL
    )
    matches = fence_pattern.findall(text)
    if not matches:
        return None
    return matches[0].strip()


def parse_with_fallbacks(
    cleaned_source: str,
) -> Tuple[Optional[ast.Module], str, Optional[str], str]:
    """Parse ``cleaned_source`` (the model's final source) into an AST.

    Tries (in order):
      1. ``cleaned_source`` as-is.
      2. The first fenced ``python`` block (handles outputs that wrap the
         function in a markdown fence).
      3. Everything before the first closing ``\\`\\`\\``` fence — handles the
         case where the opening fence was already in the prompt and only the
         closing one appears in the model's output (the typical DC-instruct
         shape on HumanEval).

    Returns ``(tree, parse_source, error_or_None, source_kind)``.
    """
    attempts: List[Tuple[str, str]] = [("cleaned_answer", cleaned_source)]

    fenced = extract_fenced_python(cleaned_source)
    if fenced is not None and fenced != cleaned_source:
        attempts.append(("fenced_code_block", fenced))

    fence = "`" * 3
    if fence in cleaned_source:
        before_close = cleaned_source.split(fence, 1)[0].strip()
        if before_close and before_close != cleaned_source and before_close != fenced:
            attempts.append(("before_closing_fence", before_close))

    errors: List[str] = []
    for source_kind, source in attempts:
        try:
            return ast.parse(source), source, None, source_kind
        except SyntaxError as exc:
            errors.append(
                f"{source_kind}: {exc.msg} at line {exc.lineno}, column {exc.offset}"
            )

    # Final fallback: longest-valid-prefix extractor. Useful when the model
    # output is truncated mid-statement (e.g. max_new_tokens cut a trailing
    # `assert`). Loaded from the vendored sanitize helper (see
    # evals/humaneval_compare/_sanitize_utils.py).
    extract_longest_valid_code = None
    try:
        import importlib.util as _ilu
        import os as _os
        _san_path = _os.path.join(
            _os.path.dirname(__file__), "..", "humaneval_compare", "_sanitize_utils.py"
        )
        _spec = _ilu.spec_from_file_location("_flexmdm_sanitize_utils", _san_path)
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        extract_longest_valid_code = _mod.extract_longest_valid_code
    except Exception:
        pass
    if extract_longest_valid_code is not None:
        snippet = extract_longest_valid_code(cleaned_source)
        if snippet and snippet != cleaned_source:
            try:
                tree = ast.parse(snippet)
                return tree, snippet, None, "longest_valid_prefix"
            except SyntaxError as exc:
                errors.append(
                    f"longest_valid_prefix: {exc.msg} at line {exc.lineno}, column {exc.offset}"
                )

    return None, cleaned_source, "; ".join(errors), "cleaned_answer"


# ---------------------------------------------------------------------------
# Metric helpers (per-node + aggregation)
# ---------------------------------------------------------------------------


def is_reference_viz_node(node: Dict[str, Any]) -> bool:
    node_type = str(node.get("type", ""))
    label = str(node.get("label", "")).lower()
    return (
        node_type in REFERENCE_NODE_TYPES
        or "generated tests" in label
        or "test cases" in label
        or "examples" in label
        or "reference" in label
    )


def metric_completion(entry: Dict[str, Any], node_id: int) -> Dict[str, Any]:
    return (entry.get("viz_completion") or {}).get(str(node_id)) or {}


def child_generated(entry: Dict[str, Any], child_id: int) -> int:
    return int(metric_completion(entry, child_id).get("generated", 0) or 0)


def child_new(entry: Dict[str, Any], child_id: int) -> int:
    return int(metric_completion(entry, child_id).get("new", 0) or 0)


def child_total(entries: Sequence[Dict[str, Any]], child_id: int) -> int:
    total = 0
    for entry in entries:
        total = max(
            total, int(metric_completion(entry, child_id).get("total", 0) or 0)
        )
    return total


def avg_or_none(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def depth_aggregate(
    local_metrics: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    by_depth: Dict[int, List[Dict[str, Any]]] = {}
    for metric in local_metrics:
        by_depth.setdefault(int(metric["depth"]), []).append(metric)
    out: Dict[str, Dict[str, Any]] = {}
    for depth, metrics in sorted(by_depth.items()):
        out[str(depth)] = {
            "node_count": len(metrics),
            "split_node_count": sum(1 for metric in metrics if metric["is_split"]),
            "cbc": avg_or_none([metric["cbc"] for metric in metrics]),
            "rub": avg_or_none([metric["rub"] for metric in metrics]),
            "rub_plus": avg_or_none([metric["rub_plus"] for metric in metrics]),
            "obw": avg_or_none([metric["obw"] for metric in metrics]),
        }
    return out


def aggregate_tree_order_metrics(
    local_metrics: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    split_metrics = [metric for metric in local_metrics if metric["is_split"]]
    return {
        "nonleaf_node_count": len(local_metrics),
        "split_node_count": len(split_metrics),
        "overall": {
            "cbc": avg_or_none([metric["cbc"] for metric in local_metrics]),
            "rub": avg_or_none([metric["rub"] for metric in local_metrics]),
            "rub_plus": avg_or_none([metric["rub_plus"] for metric in local_metrics]),
            "obw": avg_or_none([metric["obw"] for metric in local_metrics]),
        },
        "split_only": {
            "cbc": avg_or_none([metric["cbc"] for metric in split_metrics]),
            "rub": avg_or_none([metric["rub"] for metric in split_metrics]),
            "rub_plus": avg_or_none([metric["rub_plus"] for metric in split_metrics]),
            "obw": avg_or_none([metric["obw"] for metric in split_metrics]),
        },
        "by_depth": depth_aggregate(local_metrics),
    }


def compute_tree_order_metrics_variant(
    viz_nodes: Sequence[Dict[str, Any]],
    generation_progress: Dict[str, Any],
    *,
    include_reference_nodes: bool,
) -> Dict[str, Any]:
    """Compute CBC / RUB / OBW for each non-leaf viz node.

    ``generation_progress`` must contain an ``entries`` list, where each entry
    has a ``viz_completion[str(node_id)] = {"generated", "new", "total"}``
    snapshot of how many tokens of that node's subtree have been revealed at
    that step. ``compute_metrics.build_generation_progress`` produces this
    structure from our ``.pt`` records.
    """
    entries = generation_progress.get("entries") or []
    nodes_by_id = {node["viz_node_id"]: node for node in viz_nodes}
    excluded: set[int] = set()
    if not include_reference_nodes:
        stack = [
            node["viz_node_id"]
            for node in viz_nodes
            if is_reference_viz_node(node)
        ]
        while stack:
            node_id = stack.pop()
            if node_id in excluded:
                continue
            excluded.add(node_id)
            stack.extend(nodes_by_id.get(node_id, {}).get("children", []))

    local_metrics: List[Dict[str, Any]] = []
    per_node: Dict[str, Dict[str, Any]] = {}

    for node in viz_nodes:
        node_id = node["viz_node_id"]
        if node_id in excluded:
            continue
        children = [
            child_id
            for child_id in node.get("children", [])
            if child_id not in excluded
        ]
        if not children:
            continue

        k = len(children)
        if k == 1:
            child_id = children[0]
            total = child_total(entries, child_id)
            metric = {
                "viz_node_id": node_id,
                "depth": node.get("depth", 0),
                "child_count": k,
                "is_split": False,
                "cbc": 1.0,
                "rub": 1.0,
                "rub_plus": 1.0,
                "obw": 1.0,
                "commit_step": None,
                "commit_progress_index": None,
                "started_at_commit_count": 1 if total > 0 else 0,
                "returned_child_count": 1 if total > 0 else 0,
                "child_visit_counts": {str(child_id): 1 if total > 0 else 0},
                "max_open_child_count": 1 if total > 0 else 0,
            }
            local_metrics.append(metric)
            per_node[str(node_id)] = metric
            continue

        totals = {child_id: child_total(entries, child_id) for child_id in children}

        commit_entry: Optional[Dict[str, Any]] = None
        for entry in entries:
            if any(
                totals[child_id] > 0
                and child_generated(entry, child_id) >= totals[child_id]
                for child_id in children
            ):
                commit_entry = entry
                break

        if commit_entry is None:
            started_at_commit_count = 0
            commit_step = None
            commit_index = None
        else:
            started_at_commit_count = sum(
                1
                for child_id in children
                if child_generated(commit_entry, child_id) > 0
            )
            commit_step = commit_entry.get("step")
            commit_index = commit_entry.get("progress_index")

        returned_child_count = 0
        rub_plus_sum = 0.0
        child_visit_counts: Dict[str, int] = {}
        for child_id in children:
            total = totals[child_id]
            if total <= 0:
                child_visit_counts[str(child_id)] = 0
                continue
            open_index: Optional[int] = None
            finish_index: Optional[int] = None
            for entry_index, entry in enumerate(entries):
                generated = child_generated(entry, child_id)
                if open_index is None and generated > 0:
                    open_index = entry_index
                if finish_index is None and generated >= total:
                    finish_index = entry_index
                    break
            # Count maximal contiguous runs of c's reveal activity, restricted
            # to entries where some (non-excluded) child of v is being
            # revealed. Entries dominated by excluded subtrees (e.g. Generated
            # Tests) don't break a run -- this matches RUB's "left for a
            # sibling and came back" semantics.
            runs = 0
            in_run = False
            for entry in entries:
                v_children_active = any(
                    child_new(entry, other_id) > 0 for other_id in children
                )
                if not v_children_active:
                    continue
                if child_new(entry, child_id) > 0:
                    if not in_run:
                        runs += 1
                        in_run = True
                else:
                    in_run = False
            child_visit_counts[str(child_id)] = runs
            rub_plus_sum += min(max(runs - 1, 0), 2) / 2.0
            if (
                open_index is None
                or finish_index is None
                or finish_index <= open_index
            ):
                continue
            sibling_activity = any(
                any(
                    other_id != child_id
                    and child_new(entry, other_id) > 0
                    for other_id in children
                )
                for entry in entries[open_index + 1 : finish_index]
            )
            if sibling_activity:
                returned_child_count += 1

        max_open_child_count = 0
        any_generated_child = False
        for entry in entries:
            open_count = 0
            for child_id in children:
                total = totals[child_id]
                if total <= 0:
                    continue
                generated = child_generated(entry, child_id)
                if generated > 0:
                    any_generated_child = True
                if 0 < generated < total:
                    open_count += 1
            max_open_child_count = max(max_open_child_count, open_count)
        if max_open_child_count == 0 and any_generated_child:
            max_open_child_count = 1

        metric = {
            "viz_node_id": node_id,
            "depth": node.get("depth", 0),
            "child_count": k,
            "is_split": True,
            "cbc": started_at_commit_count / k,
            "rub": returned_child_count / k,
            "rub_plus": rub_plus_sum / k,
            "obw": max_open_child_count / k,
            "commit_step": commit_step,
            "commit_progress_index": commit_index,
            "started_at_commit_count": started_at_commit_count,
            "returned_child_count": returned_child_count,
            "child_visit_counts": child_visit_counts,
            "max_open_child_count": max_open_child_count,
        }
        local_metrics.append(metric)
        per_node[str(node_id)] = metric

    return {
        "variant": "reference_aware" if include_reference_nodes else "code_only",
        "local": local_metrics,
        "per_node": per_node,
        "aggregate": aggregate_tree_order_metrics(local_metrics),
        "excluded_reference_node_ids": sorted(excluded),
        "notes": (
            "reference-aware uses the simplified visualization tree as shown; "
            "code-only excludes generated tests/examples nodes. Docstrings are "
            "currently collapsed into owner nodes."
        ),
    }


def attach_tree_order_metrics(
    viz_nodes: List[Dict[str, Any]],
    generation_progress: Dict[str, Any],
) -> Dict[str, Any]:
    metrics = {
        "reference_aware": compute_tree_order_metrics_variant(
            viz_nodes, generation_progress, include_reference_nodes=True
        ),
        "code_only": compute_tree_order_metrics_variant(
            viz_nodes, generation_progress, include_reference_nodes=False
        ),
    }
    reference_per_node = metrics["reference_aware"]["per_node"]
    code_only_per_node = metrics["code_only"]["per_node"]
    for node in viz_nodes:
        node_id = str(node["viz_node_id"])
        node["tree_order_metrics"] = reference_per_node.get(node_id)
        if node_id in code_only_per_node:
            node["code_only_tree_order_metrics"] = code_only_per_node[node_id]
    return metrics


# ---------------------------------------------------------------------------
# Per-sample summary (small dict suitable for CSV-style reporting)
# ---------------------------------------------------------------------------


def line_count(source: str) -> int:
    if not source:
        return 0
    return len(source.splitlines())


def summarize_sample(result: Dict[str, Any]) -> Dict[str, Any]:
    ast_types = {node["type"] for node in result.get("ast_nodes", [])}
    viz_types = [node["type"] for node in result.get("visualization_nodes", [])]
    parse_error = result.get("parse_error")
    tree_metrics = result.get("tree_order_metrics", {})
    ref_agg = tree_metrics.get("reference_aware", {}).get("aggregate", {})
    ref_overall = ref_agg.get("overall", {})
    ref_split = ref_agg.get("split_only", {})
    code_agg = tree_metrics.get("code_only", {}).get("aggregate", {})
    code_overall = code_agg.get("overall", {})
    code_split = code_agg.get("split_only", {})
    return {
        "sample_idx": result.get("sample_idx"),
        "parse_success": result.get("parse_success", False),
        "line_count": line_count(
            result.get("parsed_source") or result.get("cleaned_final_answer", "")
        ),
        "token_count": len(result.get("tokens", [])),
        "raw_ast_node_count": len(result.get("ast_nodes", [])),
        "visualization_node_count": len(result.get("visualization_nodes", [])),
        "control_flow_node_count": sum(
            1 for typ in viz_types if typ in CONTROL_FLOW_TYPES
        ),
        "has_function_def": "FunctionDef" in ast_types
        or "AsyncFunctionDef" in ast_types,
        "has_loop": bool({"For", "AsyncFor", "While"} & ast_types),
        "has_if": "If" in ast_types,
        "has_generated_tests_or_examples": result.get(
            "test_case_detection_info", {}
        ).get("has_generated_tests_or_examples", False),
        "docstring_collapsed": result.get("docstring_handling_info", {}).get(
            "docstring_collapsed", False
        ),
        "ref_overall_cbc": ref_overall.get("cbc"),
        "ref_overall_rub": ref_overall.get("rub"),
        "ref_overall_rub_plus": ref_overall.get("rub_plus"),
        "ref_overall_obw": ref_overall.get("obw"),
        "ref_split_cbc": ref_split.get("cbc"),
        "ref_split_rub": ref_split.get("rub"),
        "ref_split_rub_plus": ref_split.get("rub_plus"),
        "ref_split_obw": ref_split.get("obw"),
        "code_overall_cbc": code_overall.get("cbc"),
        "code_overall_rub": code_overall.get("rub"),
        "code_overall_rub_plus": code_overall.get("rub_plus"),
        "code_overall_obw": code_overall.get("obw"),
        "code_split_cbc": code_split.get("cbc"),
        "code_split_rub": code_split.get("rub"),
        "code_split_rub_plus": code_split.get("rub_plus"),
        "code_split_obw": code_split.get("obw"),
        "parse_error": parse_error or "",
    }
