import ast
import gc
import os, json
import numpy as np
import torch
from tqdm.auto import tqdm
from datasets import load_dataset, load_dataset_builder
from transformers import AutoTokenizer
from torch.utils.data import Dataset, random_split

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _get_train_size_fallback():
    return 11_846_109


def remove_first_function_docstring(code: str) -> tuple[str, bool, bool]:
    """
    Remove the leading docstring from the first function in a TinyGSM code sample.

    Returns:
      (code_without_docstring, removed_docstring, parse_failed)
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, False, True

    function_node = next(
        (node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))),
        None,
    )
    if function_node is None or not function_node.body:
        return code, False, False

    first_stmt = function_node.body[0]
    if not (
        isinstance(first_stmt, ast.Expr)
        and isinstance(first_stmt.value, ast.Constant)
        and isinstance(first_stmt.value.value, str)
    ):
        return code, False, False

    if first_stmt.end_lineno is None:
        return code, False, True

    lines = code.splitlines(keepends=True)
    del lines[first_stmt.lineno - 1 : first_stmt.end_lineno]
    return "".join(lines).strip(), True, False


def remove_redundant_blank_lines(code: str) -> tuple[str, int]:
    """Remove whitespace-only lines left by generated formatting."""
    lines = code.splitlines()
    kept_lines = [line for line in lines if line.strip()]
    return "\n".join(kept_lines).strip(), len(lines) - len(kept_lines)

def pretokenize_tinygsm_split(
    out_dir: str,
    tokenizer_name: str = "Qwen/Qwen2-0.5B",
    max_len: int = 512,
    max_prompt_len: int = 512,
    max_seg_len: int = 32,
    max_seg_num: int = 16,
    sep: str = "\n",
    batch_size: int = 2048,
    streaming: bool = True,
    limit: int | None = None,
    drop_overflow: bool = True,
):
    """
    Tokenize TinyGSM into prompt tensors and fixed-size answer segment tensors.

    The duplicated prompt docstring and redundant blank lines are removed from
    the raw TinyGSM code before segmentation. Newline is the only delimiter, and
    it is kept in the segment that ends at the delimiter:
      "a\nb\nc" -> ["a\n", "b\n", "c"]

    Saved files:
      - labels.bin       uint32 [N, max_len], flat prompt+answer labels
      - prompt_mask.bin  uint8  [N, max_len/8], packed prompt mask
      - prompt.bin       uint32 [N, max_prompt_len], contains prompt_ids + sep_ids
      - split_labels.bin uint32 [N, max_seg_num, max_seg_len]

    EOS is used as the padding token for prompts, short segments, and unused
    segment slots.

    If drop_overflow=True, samples are counted and skipped when the prompt is
    longer than max_prompt_len, any segment reaches max_seg_len, or the segment
    count is greater than max_seg_num.
    """
    os.makedirs(out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    eos_id = tok.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no eos_token_id")

    sep_ids = tok(sep, add_special_tokens=False).input_ids

    # Determine source dataset size (upper bound for output memmap allocation).
    if not streaming:
        ds = load_dataset("TinyGSM/TinyGSM", split="train", streaming=False)
        source_N = len(ds)
    else:
        try:
            builder = load_dataset_builder("TinyGSM/TinyGSM")
            source_N = builder.info.splits["train"].num_examples
        except Exception:
            source_N = _get_train_size_fallback()

        ds = load_dataset("TinyGSM/TinyGSM", split="train", streaming=True)

    if limit is not None:
        source_N = min(source_N, int(limit))

    if max_len <= 0:
        raise ValueError("max_len must be positive.")
    if max_len % 8 != 0:
        raise ValueError("max_len must be divisible by 8 to pack prompt_mask cleanly.")
    if max_prompt_len <= 0:
        raise ValueError("max_prompt_len must be positive.")
    if max_seg_len <= 0:
        raise ValueError("max_seg_len must be positive.")
    if max_seg_num <= 0:
        raise ValueError("max_seg_num must be positive.")

    flat_labels_path = os.path.join(out_dir, "labels.bin")
    mask_path = os.path.join(out_dir, "prompt_mask.bin")
    prompt_path = os.path.join(out_dir, "prompt.bin")
    split_labels_path = os.path.join(out_dir, "split_labels.bin")
    meta_path = os.path.join(out_dir, "meta.json")
    mask_bytes = max_len // 8

    # Allocate memmaps
    flat_labels_mm = np.memmap(
        flat_labels_path,
        mode="w+",
        dtype=np.uint32,
        shape=(source_N, max_len),
    )
    mask_mm = np.memmap(
        mask_path,
        mode="w+",
        dtype=np.uint8,
        shape=(source_N, mask_bytes),
    )
    prompt_mm = np.memmap(
        prompt_path,
        mode="w+",
        dtype=np.uint32,
        shape=(source_N, max_prompt_len),
    )
    split_labels_mm = np.memmap(
        split_labels_path,
        mode="w+",
        dtype=np.uint32,
        shape=(source_N, max_seg_num, max_seg_len),
    )

    def batched(it, n):
        buf = []
        for x in it:
            buf.append(x)
            if len(buf) == n:
                yield buf
                buf = []
        if buf:
            yield buf

    def split_keep_delimiters(text: str, delimiters=("\n",)) -> list[str]:
        segments = []
        current = []
        delimiter_set = set(delimiters)
        for ch in text:
            current.append(ch)
            if ch in delimiter_set:
                segments.append("".join(current))
                current = []
        if current:
            segments.append("".join(current))
        return segments

    def pack_mask(mask_bool_1d: np.ndarray) -> np.ndarray:
        return np.packbits(mask_bool_1d.astype(np.uint8), axis=-1, bitorder="little")

    source_seen = 0
    written = 0
    samples_exceeding_max_prompt_len = 0
    samples_exceeding_max_seg_len = 0
    segments_exceeding_max_seg_len = 0
    samples_reaching_max_seg_len = 0
    segments_reaching_max_seg_len = 0
    samples_exceeding_max_seg_num = 0
    extra_segments_beyond_max_seg_num = 0
    skipped_overflow_samples = 0
    docstrings_removed = 0
    docstrings_missing = 0
    docstring_parse_failures = 0
    blank_lines_removed = 0
    max_observed_prompt_len = 0
    max_observed_answer_len = 0
    max_observed_total_len = 0
    max_observed_seg_len = 0
    max_observed_seg_num = 0
    pbar_total = source_N
    pbar = tqdm(total=pbar_total, desc=f"Pretokenizing TinyGSM -> {out_dir}")

    for batch in batched(ds, batch_size):
        if source_seen >= source_N:
            break
        if source_seen + len(batch) > source_N:
            batch = batch[: (source_N - source_seen)]

        prompts = [(ex.get("question") or "").strip() for ex in batch]
        cleaned_codes = []
        code_segments = []
        for ex in batch:
            code, removed, parse_failed = remove_first_function_docstring(
                (ex.get("code") or "").strip()
            )
            code, removed_blank_lines = remove_redundant_blank_lines(code)
            cleaned_codes.append(code)
            code_segments.append(split_keep_delimiters(code))
            if removed:
                docstrings_removed += 1
            elif parse_failed:
                docstring_parse_failures += 1
            else:
                docstrings_missing += 1
            blank_lines_removed += removed_blank_lines

        # Batched tokenize
        p_ids_batch = tok(prompts, add_special_tokens=False).input_ids
        a_ids_batch = tok(cleaned_codes, add_special_tokens=False).input_ids
        flat_segments = [seg for sample_segments in code_segments for seg in sample_segments]
        flat_segment_ids = (
            tok(flat_segments, add_special_tokens=False).input_ids
            if flat_segments
            else []
        )

        seg_cursor = 0
        for p_ids, a_ids, sample_segments in zip(
            p_ids_batch,
            a_ids_batch,
            code_segments,
        ):
            prompt_ids = p_ids + sep_ids
            raw_ids = prompt_ids + a_ids
            max_observed_prompt_len = max(max_observed_prompt_len, len(prompt_ids))
            max_observed_answer_len = max(max_observed_answer_len, len(a_ids))
            max_observed_total_len = max(max_observed_total_len, len(raw_ids))
            prompt_overflow = len(prompt_ids) > max_prompt_len
            if len(prompt_ids) > max_prompt_len:
                samples_exceeding_max_prompt_len += 1

            n_seg = len(sample_segments)
            max_observed_seg_num = max(max_observed_seg_num, n_seg)
            seg_num_overflow = n_seg > max_seg_num
            if seg_num_overflow:
                samples_exceeding_max_seg_num += 1
                extra_segments_beyond_max_seg_num += n_seg - max_seg_num

            sample_segment_ids = flat_segment_ids[seg_cursor : seg_cursor + n_seg]
            seg_cursor += n_seg
            has_long_segment = False
            has_segment_reaching_limit = False
            for seg_ids in sample_segment_ids:
                max_observed_seg_len = max(max_observed_seg_len, len(seg_ids))
                if len(seg_ids) > max_seg_len:
                    has_long_segment = True
                    segments_exceeding_max_seg_len += 1
                if len(seg_ids) >= max_seg_len:
                    has_segment_reaching_limit = True
                    segments_reaching_max_seg_len += 1
            if has_long_segment:
                samples_exceeding_max_seg_len += 1
            if has_segment_reaching_limit:
                samples_reaching_max_seg_len += 1

            sample_overflow = prompt_overflow or has_segment_reaching_limit or seg_num_overflow
            if drop_overflow and sample_overflow:
                skipped_overflow_samples += 1
                source_seen += 1
                pbar.update(1)
                continue

            flat_labels_mm[written].fill(eos_id)
            prompt_mm[written].fill(eos_id)
            split_labels_mm[written].fill(eos_id)

            pm = np.zeros((max_len,), dtype=np.bool_)
            if len(raw_ids) >= max_len:
                flat_ids = raw_ids[: max_len - 1] + [eos_id]
                prompt_boundary = min(len(prompt_ids), max_len - 1)
            else:
                flat_ids = raw_ids + [eos_id] * (max_len - len(raw_ids))
                prompt_boundary = min(len(prompt_ids), max_len)
            if prompt_boundary > 0:
                pm[:prompt_boundary] = True
            flat_labels_mm[written, :] = np.asarray(flat_ids, dtype=np.uint32)
            mask_mm[written, :] = pack_mask(pm)

            prompt_ids = prompt_ids[:max_prompt_len]
            prompt_mm[written, : len(prompt_ids)] = np.asarray(prompt_ids, dtype=np.uint32)
            for seg_idx, seg_ids in enumerate(sample_segment_ids[:max_seg_num]):
                if len(seg_ids) >= max_seg_len:
                    seg_ids = seg_ids[: max_seg_len - 1] + [eos_id]
                split_labels_mm[written, seg_idx, : len(seg_ids)] = np.asarray(
                    seg_ids,
                    dtype=np.uint32,
                )
            written += 1
            source_seen += 1
            pbar.update(1)

            if source_seen >= source_N:
                break

    pbar.close()

    # Flush to disk
    flat_labels_mm.flush()
    mask_mm.flush()
    prompt_mm.flush()
    split_labels_mm.flush()
    del flat_labels_mm
    del mask_mm
    del prompt_mm
    del split_labels_mm

    dtype_size = np.dtype(np.uint32).itemsize
    os.truncate(flat_labels_path, written * max_len * dtype_size)
    os.truncate(mask_path, written * mask_bytes * np.dtype(np.uint8).itemsize)
    os.truncate(prompt_path, written * max_prompt_len * dtype_size)
    os.truncate(split_labels_path, written * max_seg_num * max_seg_len * dtype_size)

    meta = {
        "dataset": "TinyGSM/TinyGSM",
        "split": "train",
        "tokenizer": tokenizer_name,
        "max_len": max_len,
        "max_prompt_len": max_prompt_len,
        "max_seg_len": max_seg_len,
        "max_seg_num": max_seg_num,
        "segment_delimiters": ["\\n"],
        "segment_delimiters_kept": True,
        "removed_code_docstring": True,
        "removed_redundant_blank_lines": True,
        "sep": sep,
        "eos_id": int(eos_id),
        "num_source_examples": int(source_seen),
        "num_examples": int(written),
        "drop_overflow": bool(drop_overflow),
        "filter_exceeding_max_prompt_len": bool(drop_overflow),
        "filter_segment_len_ge_max_seg_len": bool(drop_overflow),
        "filter_exceeding_max_seg_num": bool(drop_overflow),
        "segment_len_ge_max_policy": "filter_sample",
        "segment_num_gt_max_policy": "filter_sample",
        "labels_dtype": "uint32",
        "prompt_dtype": "uint32",
        "split_labels_dtype": "uint32",
        "labels_shape": [int(written), int(max_len)],
        "prompt_mask_packed": True,
        "prompt_mask_bitorder": "little",
        "prompt_mask_shape": [int(written), int(mask_bytes)],
        "prompt_shape": [int(written), int(max_prompt_len)],
        "split_labels_shape": [int(written), int(max_seg_num), int(max_seg_len)],
        "samples_exceeding_max_prompt_len": int(samples_exceeding_max_prompt_len),
        "samples_exceeding_max_seg_len": int(samples_exceeding_max_seg_len),
        "segments_exceeding_max_seg_len": int(segments_exceeding_max_seg_len),
        "samples_reaching_max_seg_len": int(samples_reaching_max_seg_len),
        "segments_reaching_max_seg_len": int(segments_reaching_max_seg_len),
        "samples_exceeding_max_seg_num": int(samples_exceeding_max_seg_num),
        "extra_segments_beyond_max_seg_num": int(extra_segments_beyond_max_seg_num),
        "skipped_overflow_samples": int(skipped_overflow_samples),
        "docstrings_removed": int(docstrings_removed),
        "docstrings_missing": int(docstrings_missing),
        "docstring_parse_failures": int(docstring_parse_failures),
        "blank_lines_removed": int(blank_lines_removed),
        "max_observed_prompt_len": int(max_observed_prompt_len),
        "max_observed_answer_len": int(max_observed_answer_len),
        "max_observed_total_len": int(max_observed_total_len),
        "max_observed_seg_len": int(max_observed_seg_len),
        "max_observed_seg_num": int(max_observed_seg_num),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. Scanned {source_seen:,} examples and wrote {written:,} examples")
    print(f"- {flat_labels_path} (uint32) shape=({written},{max_len})")
    print(f"- {mask_path}   (packed uint8) shape=({written},{mask_bytes})")
    print(f"- {prompt_path} (uint32) shape=({written},{max_prompt_len})")
    print(f"- {split_labels_path} (uint32) shape=({written},{max_seg_num},{max_seg_len})")
    print(f"- {meta_path}")
    print(
        "Stats: "
        f"samples_exceeding_max_prompt_len={samples_exceeding_max_prompt_len}, "
        f"samples_exceeding_max_seg_len={samples_exceeding_max_seg_len}, "
        f"segments_exceeding_max_seg_len={segments_exceeding_max_seg_len}, "
        f"samples_reaching_max_seg_len={samples_reaching_max_seg_len}, "
        f"segments_reaching_max_seg_len={segments_reaching_max_seg_len}, "
        f"samples_exceeding_max_seg_num={samples_exceeding_max_seg_num}, "
        f"extra_segments_beyond_max_seg_num={extra_segments_beyond_max_seg_num}, "
        f"skipped_overflow_samples={skipped_overflow_samples}, "
        f"docstrings_removed={docstrings_removed}, "
        f"docstrings_missing={docstrings_missing}, "
        f"docstring_parse_failures={docstring_parse_failures}, "
        f"blank_lines_removed={blank_lines_removed}, "
        f"max_observed_prompt_len={max_observed_prompt_len}, "
        f"max_observed_answer_len={max_observed_answer_len}, "
        f"max_observed_total_len={max_observed_total_len}, "
        f"max_observed_seg_len={max_observed_seg_len}, "
        f"max_observed_seg_num={max_observed_seg_num}"
    )
    del ds
    del tok
    gc.collect()



class TinyGSMSplitDataset(Dataset):
    """
    Loads segmented TinyGSM from:
      - prompt_mask.bin  uint8  [N, max_len/8] (packed bits)
      - prompt.bin       uint32 [N, max_prompt_len]
      - split_labels.bin uint32 [N, max_seg_num, max_seg_len]
      - meta.json
    Returns:
      {"prompt": LongTensor[max_prompt_len],
       "split_labels": LongTensor[max_seg_num, max_seg_len],
       "prompt_mask": BoolTensor[max_len],
       "prompt_len": LongTensor[]}
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        meta_path = os.path.join(data_dir, "meta.json")
        with open(meta_path, "r") as f:
            self.meta = json.load(f)

        self.N = int(self.meta["num_examples"])
        self.max_len = int(self.meta["max_len"])
        self.max_prompt_len = int(self.meta["max_prompt_len"])
        self.max_seg_len = int(self.meta["max_seg_len"])
        self.max_seg_num = int(self.meta["max_seg_num"])
        self.bitorder = self.meta.get("prompt_mask_bitorder", "little")

        mask_path = os.path.join(data_dir, "prompt_mask.bin")
        prompt_path = os.path.join(data_dir, "prompt.bin")
        split_labels_path = os.path.join(data_dir, "split_labels.bin")

        self.mask_mm = np.memmap(
            mask_path,
            mode="r",
            dtype=np.uint8,
            shape=(self.N, self.max_len // 8),
        )
        self.prompt_mm = np.memmap(
            prompt_path,
            mode="r",
            dtype=np.uint32,
            shape=(self.N, self.max_prompt_len),
        )
        self.labels_mm = np.memmap(
            split_labels_path,
            mode="r",
            dtype=np.uint32,
            shape=(self.N, self.max_seg_num, self.max_seg_len),
        )

    def __len__(self):
        return self.N

    def __getitem__(self, idx: int):
        packed = self.mask_mm[idx]
        mask = np.unpackbits(packed, bitorder=self.bitorder)[: self.max_len].astype(np.bool_)
        prompt_mask = torch.from_numpy(mask)
        prompt_len = torch.tensor(int(mask.sum()), dtype=torch.long)
        prompt = torch.from_numpy(self.prompt_mm[idx].astype(np.int64))
        split_labels = torch.from_numpy(self.labels_mm[idx].astype(np.int64))

        return {
            "prompt": prompt,
            "split_labels": split_labels,
            "prompt_mask": prompt_mask,
            "prompt_len": prompt_len,
        }

def split_tinygsm(data_dir: str, val_ratio: float = 0.05, seed: int = 2025):
    dataset = TinyGSMSplitDataset(data_dir)
    n = len(dataset)

    n_val = int(n * val_ratio)
    n_train = n - n_val

    g = torch.Generator().manual_seed(seed)
    train_data, val_data = random_split(dataset, [n_train, n_val], generator=g)
    return train_data, val_data



if __name__ == "__main__":
    out_dir = "data/tiny_gsm_split_v2_limit2048"
    pretokenize_tinygsm_split(out_dir=out_dir, limit=2048)
