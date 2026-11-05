"""Read-only WAV header and six-vowel inventory audit using the standard library.

This does not decode audio, assign splits, train models, or validate clinical IDs.
All output is written outside the source audio directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path


CLASS_MAP = {
    "NV": "NV", "SD": "SD", "UVFP": "UVFP", "BPLVF": "BSVFL",
    "VFS": "BSVFL", "MVFT": "MVFT",
}
VOWELS = ("a", "e", "i", "o", "u", "v")


def wav_header(path: Path) -> dict:
    size = path.stat().st_size
    with path.open("rb") as stream:
        if stream.read(4) != b"RIFF":
            raise ValueError("expected little-endian RIFF WAV")
        stream.read(4)
        if stream.read(4) != b"WAVE":
            raise ValueError("missing WAVE signature")
        fmt = None
        data_bytes = 0
        while stream.tell() + 8 <= size:
            chunk_id = stream.read(4)
            chunk_size = struct.unpack("<I", stream.read(4))[0]
            start = stream.tell()
            if start + chunk_size > size:
                raise ValueError("chunk extends beyond file size")
            if chunk_id == b"fmt ":
                payload = stream.read(min(chunk_size, 64))
                if len(payload) < 16:
                    raise ValueError("short fmt chunk")
                tag, channels, sr, byte_rate, block_align, bits = struct.unpack(
                    "<HHIIHH", payload[:16]
                )
                effective_tag = tag
                if tag == 65534 and len(payload) >= 40:
                    effective_tag = struct.unpack("<I", payload[24:28])[0]
                fmt = {
                    "format_tag": tag, "effective_format_tag": effective_tag,
                    "channels": channels, "sample_rate_hz": sr,
                    "byte_rate": byte_rate, "block_align": block_align,
                    "bits_per_sample": bits,
                }
            elif chunk_id == b"data":
                data_bytes += chunk_size
            stream.seek(start + chunk_size + chunk_size % 2)
        if fmt is None or data_bytes == 0:
            raise ValueError("missing fmt or nonempty data chunk")
        if fmt["effective_format_tag"] not in (1, 3):
            raise ValueError("non-PCM/non-IEEE-float format requires decoder review")
        if min(fmt["channels"], fmt["sample_rate_hz"], fmt["block_align"]) <= 0:
            raise ValueError("invalid channels, sample rate or block alignment")
        if data_bytes % fmt["block_align"]:
            raise ValueError("data size is not an integral number of frames")
        frames = data_bytes // fmt["block_align"]
        fmt.update(data_bytes=data_bytes, frames=frames,
                   duration_seconds=frames / fmt["sample_rate_hz"])
        return fmt


def quantiles(values: list[float]) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)
    result = {}
    for name, fraction in (("min", 0), ("p05", .05), ("p25", .25),
                           ("median", .5), ("p75", .75), ("p95", .95), ("max", 1)):
        position = (len(ordered) - 1) * fraction
        lo = int(position)
        hi = min(lo + 1, len(ordered) - 1)
        result[name] = ordered[lo] + (position - lo) * (ordered[hi] - ordered[lo])
    return result


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hash-files", action="store_true")
    args = parser.parse_args()
    root, output = args.data_root.resolve(), args.output.resolve()
    if output == root or root in output.parents:
        parser.error("output must be outside the source audio directory")
    if not root.is_dir():
        parser.error("data root does not exist")
    output.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict]] = defaultdict(list)
    folders = {}
    unknown_folders = []
    for class_folder in sorted(root.iterdir()):
        if not class_folder.is_dir():
            continue
        if class_folder.name not in CLASS_MAP:
            unknown_folders.append(class_folder.name)
            continue
        for subject_folder in sorted(class_folder.iterdir()):
            if subject_folder.is_dir():
                group = f"{class_folder.name}:{subject_folder.name}"
                folders[group] = (class_folder.name, subject_folder.name)
    records = []
    layout_errors = []
    non_wav_count = 0
    duplicate_hashes: dict[str, list[str]] = defaultdict(list)
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() != ".wav":
            non_wav_count += 1
            continue
        relative = path.relative_to(root)
        parts = relative.parts
        if len(parts) != 3 or parts[0] not in CLASS_MAP:
            layout_errors.append(relative.as_posix())
            continue
        original, folder = parts[0], parts[1]
        group = f"{original}:{folder}"
        vowel = path.stem.rsplit("_", 1)[-1].lower()
        stat = path.stat()
        record = {
            "relative_path": relative.as_posix(),
            "candidate_group_id": group,
            "identity_status": "folder-derived; clinical identity not verified",
            "original_label": original, "label": CLASS_MAP[original],
            "vowel_file_token": vowel,
            "vowel_display": "ü" if vowel == "v" else vowel,
            "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "header_ok": False,
        }
        try:
            record.update(wav_header(path))
            record["header_ok"] = True
        except (OSError, ValueError, struct.error) as error:
            record["header_error"] = str(error)
        if args.hash_files:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            record["file_sha256"] = digest.hexdigest()
            duplicate_hashes[digest.hexdigest()].append(relative.as_posix())
        records.append(record)
        grouped[group].append(record)
    subjects = []
    for group, (original, folder) in sorted(folders.items()):
        files = grouped[group]
        tokens = Counter(record["vowel_file_token"] for record in files)
        missing = [v for v in VOWELS if tokens[v] == 0]
        unexpected = sorted(set(tokens) - set(VOWELS))
        repeated = sorted(v for v, count in tokens.items() if count > 1)
        complete = len(files) == 6 and not (missing or unexpected or repeated)
        subjects.append({
            "candidate_group_id": group, "original_label": original,
            "label": CLASS_MAP[original], "folder": folder,
            "file_count": len(files), "missing_vowels": missing,
            "unexpected_vowels": unexpected, "repeated_vowels": repeated,
            "complete_six_filenames": complete,
            "all_headers_ok": bool(files) and all(r["header_ok"] for r in files),
            "candidate_for_complete_six_analysis": complete and all(r["header_ok"] for r in files),
            "identity_verified": False, "waveform_quality_verified": False,
        })
    file_jsonl = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records)
    subject_jsonl = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in subjects)
    (output / "audio_manifest.jsonl").write_text(file_jsonl, encoding="utf-8")
    (output / "subject_inventory.jsonl").write_text(subject_jsonl, encoding="utf-8")
    summary = {
        "audit_time_utc_plus_8": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "data_root": str(root), "scope": "directory and WAV header audit; no waveform decoding",
        "candidate_subject_folders": len(subjects), "wav_files": len(records),
        "complete_six_filename_folders": sum(s["complete_six_filenames"] for s in subjects),
        "complete_six_and_header_ok_folders": sum(s["candidate_for_complete_six_analysis"] for s in subjects),
        "header_error_files": sum(not r["header_ok"] for r in records),
        "non_wav_files": non_wav_count, "layout_errors": layout_errors,
        "unknown_class_folders": unknown_folders,
        "content_hashing_performed": args.hash_files,
        "duplicate_file_hash_groups": [v for v in duplicate_hashes.values() if len(v) > 1] if args.hash_files else None,
        "sample_rate_counts": dict(Counter(str(r["sample_rate_hz"]) for r in records if r["header_ok"])),
        "channel_counts": dict(Counter(str(r["channels"]) for r in records if r["header_ok"])),
        "duration_seconds": quantiles([r["duration_seconds"] for r in records if r["header_ok"]]),
        "by_class": {},
        "irregular_folders": [s for s in subjects if not s["candidate_for_complete_six_analysis"]],
        "audio_manifest_sha256": hashlib.sha256(file_jsonl.encode("utf-8")).hexdigest(),
        "fingerprint_scope": "manifest contents; raw audio bytes hashed only if --hash-files is used",
        "training_started": False, "split_assigned": False,
    }
    for label in ("NV", "SD", "UVFP", "BSVFL", "MVFT"):
        selected = [s for s in subjects if s["label"] == label]
        summary["by_class"][label] = {
            "candidate_folders": len(selected),
            "complete_six_and_header_ok": sum(s["candidate_for_complete_six_analysis"] for s in selected),
            "wav_files": sum(s["file_count"] for s in selected),
        }
    dump_json(output / "inventory_summary.json", summary)
    print(json.dumps({key: summary[key] for key in (
        "candidate_subject_folders", "wav_files", "complete_six_and_header_ok_folders",
        "header_error_files", "sample_rate_counts", "channel_counts", "duration_seconds", "by_class"
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
