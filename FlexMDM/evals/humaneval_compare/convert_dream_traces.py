"""Convert Dream / Diffu-Coder generation_history.rank*.jsonl traces into the
standard .pt + tasks.json layout consumed by `evals/tree_analysis/cli.py`.

Input layout:
    <input_dir>/generation_history.rank{0,1,2,3}.jsonl

Each row has:
    rank, batch_index, sample_index, prompt, final_response,
    history_decode_every, generation_order, history

We write:
    <output_root>/<gen_dataset>/tasks.json
    <output_root>/<gen_dataset>/raw/<TaskName>__<model>__<alg>/sample_00.pt

The .pt schema matches `flexmdm/inference.py` outputs (so the existing
anyorder pipeline reads it unchanged):
    prompt, prompt_len, sequences (S, L) int32, attention_masks (S, L) bool,
    insertion_masks (S, L) bool (all False for non-flex MDMs),
    mask_id, pad_id, meta

For HE the task_id is matched against the canonical evals/humaneval
list (164 tasks). For MBPP the trace data uses the original 500-task
test split and prompts are 3-shot — we derive a synthetic task_id from
the entry-point parsed out of the *last* "Your code should pass these
tests:" block (the actual task, ignoring few-shot examples). Collisions
across the 500 traces are disambiguated with a numeric suffix.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch


MASK_ID = 151666
PAD_ID = 151643
BOS_TOKEN = "<|beginoftext|>"


def _strip_bos(prompt: str) -> str:
    return prompt[len(BOS_TOKEN):] if prompt.startswith(BOS_TOKEN) else prompt


def _last_test_block(prompt: str) -> str:
    """Return everything after the LAST 'Your code should pass these tests:' marker.
    For 3-shot MBPP prompts, this isolates the actual target task."""
    idx = prompt.rfind('Your code should pass these tests:')
    return prompt[idx:] if idx >= 0 else ''


def _entry_point_from_test_block(block: str) -> Optional[str]:
    """Extract the entry-point name from the first `assert <name>(...)` in the
    given test block."""
    m = re.search(r'assert\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(', block)
    return m.group(1) if m else None


def _build_record(
    *,
    prompt: str,
    prompt_token_ids: List[int],
    history: List[Dict[str, Any]],
    task_id: str,
    entry_point: str,
    model: str,
    alg: str,
    sample_k: int,
    dataset: str,
) -> Dict[str, Any]:
    """Build a .pt record with the prompt tokens *prepended* to each step's
    response. We set ``prompt_len = 0`` so the anyorder pipeline parses
    prompt + response as a unit (this matters for HE-style prompts that
    provide the function header — without this, the body-only response has
    no ``def``-line for the AST to anchor to)."""
    if not history:
        raise ValueError(f"empty history for {task_id}")

    response_len = len(history[0]['response_token_ids'])
    prompt_len_native = len(prompt_token_ids)
    total_len = prompt_len_native + response_len
    n_steps = len(history)

    prompt_tensor = torch.tensor(prompt_token_ids, dtype=torch.int32)
    sequences = torch.empty((n_steps, total_len), dtype=torch.int32)
    sequences[:, :prompt_len_native] = prompt_tensor.unsqueeze(0)
    for i, h in enumerate(history):
        sequences[i, prompt_len_native:] = torch.tensor(h['response_token_ids'], dtype=torch.int32)

    attention_masks = torch.ones((n_steps, total_len), dtype=torch.bool)
    insertion_masks = torch.zeros((n_steps, total_len), dtype=torch.bool)

    return {
        'prompt': prompt,
        'prompt_len': 0,  # parse prompt + response together
        'sequences': sequences,
        'attention_masks': attention_masks,
        'insertion_masks': insertion_masks,
        'mask_id': MASK_ID,
        'pad_id': PAD_ID,
        'meta': {
            'task_id': task_id,
            'entry_point': entry_point,
            'dataset': dataset,
            'model': model,
            'alg': alg,
            'sample_k': int(sample_k),
            'steps': int(n_steps),
            'temperature': 0.0,
            'source': 'dream_jsonl_converted',
            'prompt_len_native': int(prompt_len_native),
        },
    }


def _record_path(output_root: str, gen_dataset: str, task_id: str, model: str,
                 alg: str, sample_k: int) -> str:
    safe_tid = task_id.replace('/', '_')
    name = f"{safe_tid}__{model}__{alg}"
    return os.path.join(output_root, gen_dataset, 'raw', name, f"sample_{sample_k:02d}.pt")


def _load_tokenizer(repo: str):
    from transformers import AutoTokenizer  # noqa: PLC0415
    return AutoTokenizer.from_pretrained(repo, trust_remote_code=True)


def _tokenize_prompt(tokenizer: Any, prompt: str) -> List[int]:
    """Tokenize the prompt without adding extra special tokens (the trace
    text already contains <|beginoftext|> at the start)."""
    return tokenizer.encode(prompt, add_special_tokens=False)


def _convert_he(*,
                input_dir: str,
                output_root: str,
                model: str,
                alg: str,
                canonical_tasks_json: str,
                tokenizer_repo: str) -> Tuple[int, int]:
    """Convert HE trace rows. Match each row to a canonical HE task by prompt
    text. Returns (n_matched, n_total)."""
    canonical = json.load(open(canonical_tasks_json))
    canon_by_prompt = {t['prompt']: t for t in canonical}
    tokenizer = _load_tokenizer(tokenizer_repo)

    gen_dataset = 'humaneval'
    out_dir = os.path.join(output_root, gen_dataset)
    os.makedirs(os.path.join(out_dir, 'raw'), exist_ok=True)

    used_tasks: List[Dict[str, Any]] = []
    matched, total = 0, 0
    seen_task_ids = set()
    for path in sorted(glob.glob(os.path.join(input_dir, 'generation_history.rank*.jsonl'))):
        for line in open(path):
            row = json.loads(line)
            total += 1
            stripped = _strip_bos(row['prompt'])
            canon = canon_by_prompt.get(stripped)
            if canon is None:
                continue
            task_id = canon['task_id']
            if task_id in seen_task_ids:
                continue
            seen_task_ids.add(task_id)
            entry_point = canon['entry_point']
            prompt_tokens = _tokenize_prompt(tokenizer, row['prompt'])
            record = _build_record(
                prompt=row['prompt'],
                prompt_token_ids=prompt_tokens,
                history=row['history'],
                task_id=task_id,
                entry_point=entry_point,
                model=model,
                alg=alg,
                sample_k=0,
                dataset=gen_dataset,
            )
            rp = _record_path(output_root, gen_dataset, task_id, model, alg, 0)
            os.makedirs(os.path.dirname(rp), exist_ok=True)
            torch.save(record, rp)
            used_tasks.append(canon)
            matched += 1

    used_tasks_sorted = sorted(used_tasks, key=lambda t: int(t['task_id'].split('/')[-1]))
    with open(os.path.join(out_dir, 'tasks.json'), 'w') as fh:
        json.dump(used_tasks_sorted, fh)
    return matched, total


def _convert_mbpp(*,
                  input_dir: str,
                  output_root: str,
                  model: str,
                  alg: str,
                  tokenizer_repo: str) -> Tuple[int, int]:
    """Convert MBPP trace rows. Trace prompts are 3-shot; we derive task_id
    from the entry-point parsed out of the LAST test block of each prompt.
    Collisions get numeric suffixes. Returns (n_matched, n_total)."""
    gen_dataset = 'mbpp'
    out_dir = os.path.join(output_root, gen_dataset)
    os.makedirs(os.path.join(out_dir, 'raw'), exist_ok=True)
    tokenizer = _load_tokenizer(tokenizer_repo)

    tasks_meta: List[Dict[str, Any]] = []
    matched, total = 0, 0
    ep_count: Dict[str, int] = {}
    for path in sorted(glob.glob(os.path.join(input_dir, 'generation_history.rank*.jsonl'))):
        for line in open(path):
            row = json.loads(line)
            total += 1
            block = _last_test_block(row['prompt'])
            ep = _entry_point_from_test_block(block)
            if ep is None:
                continue
            n = ep_count.get(ep, 0)
            ep_count[ep] = n + 1
            task_id = f"Mbpp/{ep}" if n == 0 else f"Mbpp/{ep}__{n}"
            prompt_tokens = _tokenize_prompt(tokenizer, row['prompt'])
            record = _build_record(
                prompt=row['prompt'],
                prompt_token_ids=prompt_tokens,
                history=row['history'],
                task_id=task_id,
                entry_point=ep,
                model=model,
                alg=alg,
                sample_k=0,
                dataset=gen_dataset,
            )
            rp = _record_path(output_root, gen_dataset, task_id, model, alg, 0)
            os.makedirs(os.path.dirname(rp), exist_ok=True)
            torch.save(record, rp)
            tasks_meta.append({
                'dataset': gen_dataset,
                'index': len(tasks_meta),
                'task_id': task_id,
                'prompt': row['prompt'],
                'entry_point': ep,
                'canonical_solution': '',
                'test_base': '',
                'test_plus': '',
                'test_list': [],
                'test_imports': [],
            })
            matched += 1

    with open(os.path.join(out_dir, 'tasks.json'), 'w') as fh:
        json.dump(tasks_meta, fh)
    return matched, total


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', required=True,
                   help='Dir containing generation_history.rank*.jsonl')
    p.add_argument('--output-root', required=True)
    p.add_argument('--model', required=True, choices=['dream', 'diffucoder'])
    p.add_argument('--alg', default='mgp')
    p.add_argument('--dataset', required=True, choices=['humaneval', 'mbpp'])
    p.add_argument('--canonical-tasks-json', default=None,
                   help='Required for HE: path to a tasks.json with canonical '
                        'HE entries (e.g., from any existing FlexMDM HE run).')
    p.add_argument('--tokenizer', required=True,
                   help='Tokenizer repo id (e.g., Dream-org/Dream-v0-Base-7B).')
    args = p.parse_args(argv)

    os.makedirs(args.output_root, exist_ok=True)
    if args.dataset == 'humaneval':
        if not args.canonical_tasks_json:
            raise SystemExit('--canonical-tasks-json required for humaneval')
        m, n = _convert_he(
            input_dir=args.input_dir,
            output_root=args.output_root,
            model=args.model,
            alg=args.alg,
            canonical_tasks_json=args.canonical_tasks_json,
            tokenizer_repo=args.tokenizer,
        )
    else:
        m, n = _convert_mbpp(
            input_dir=args.input_dir,
            output_root=args.output_root,
            model=args.model,
            alg=args.alg,
            tokenizer_repo=args.tokenizer,
        )
    print(f'[converted] {m} / {n} rows from {args.input_dir} → {args.output_root}/{args.dataset}/')


if __name__ == '__main__':
    main()
