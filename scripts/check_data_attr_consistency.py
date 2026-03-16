#!/usr/bin/env python3
"""Check consistency between data/*.json and attr/*.json files.

Rule:
- For each class/split pair, all IDs appearing in data samples
  (target + references) should exist as keys in the corresponding attr file.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


CLASSES = ["all", "dress", "shirt", "toptee"]
SPLITS = ["train", "val"]


def extract_ids_from_sample(sample: object) -> set[str]:
    ids: set[str] = set()
    if not isinstance(sample, dict):
        return ids

    target = sample.get("target")
    if isinstance(target, list):
        if len(target) >= 2:
            ids.add(str(target[1]))
        elif len(target) == 1:
            ids.add(str(target[0]))
    elif isinstance(target, dict) and "id" in target:
        ids.add(str(target["id"]))
    elif target is not None:
        ids.add(str(target))

    references = sample.get("reference", [])
    if isinstance(references, list):
        for ref in references:
            if isinstance(ref, list):
                if len(ref) >= 3:
                    ids.add(str(ref[2]))
                elif len(ref) == 1:
                    ids.add(str(ref[0]))
            elif isinstance(ref, dict) and "id" in ref:
                ids.add(str(ref["id"]))

    return ids


def compute_rows(
    root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    data_dir = root / "data"
    attr_dir = root / "attr"
    split_dir = root / "image_splits"

    rows: list[dict[str, object]] = []
    missing_details: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []

    for cls in CLASSES:
        for split in SPLITS:
            data_path = data_dir / f"{cls}.{split}.json"
            attr_path = attr_dir / f"asin2attr.{cls}.{split}.new.json"
            split_path = split_dir / f"split.{cls}.{split}.json"

            if not data_path.exists() or not attr_path.exists():
                rows.append(
                    {
                        "class": cls,
                        "split": split,
                        "data_count": "MISSING",
                        "attr_count": "MISSING",
                        "unique_ids": "MISSING",
                        "missing_count": "MISSING",
                        "coverage": "MISSING",
                        "status": "WARN",
                    }
                )
                continue

            with data_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            with attr_path.open("r", encoding="utf-8") as f:
                attr = json.load(f)
            with split_path.open("r", encoding="utf-8") as f:
                split_ids_raw = json.load(f)

            if not isinstance(data, list):
                raise TypeError(f"{data_path} expected list, got {type(data)}")
            if not isinstance(attr, dict):
                raise TypeError(f"{attr_path} expected dict, got {type(attr)}")

            id_set: set[str] = set()
            for sample in data:
                id_set.update(extract_ids_from_sample(sample))

            attr_keys = {str(k) for k in attr.keys()}
            split_ids = {str(x) for x in split_ids_raw}
            missing = sorted(id_set - attr_keys)
            unique_ids = len(id_set)
            missing_count = len(missing)
            coverage = 1.0 if unique_ids == 0 else (unique_ids - missing_count) / float(unique_ids)

            data_minus_split = sorted(id_set - split_ids)
            split_minus_attr = sorted(split_ids - attr_keys)
            attr_minus_split = sorted(attr_keys - split_ids)
            split_coverage = 1.0 if unique_ids == 0 else (unique_ids - len(data_minus_split)) / float(unique_ids)

            row = {
                "class": cls,
                "split": split,
                "data_count": len(data),
                "attr_count": len(attr_keys),
                "unique_ids": unique_ids,
                "missing_count": missing_count,
                "coverage": coverage,
                "status": "PASS" if missing_count == 0 else "WARN",
            }
            rows.append(row)

            if missing_count > 0:
                missing_details.append(
                    {
                        "class": cls,
                        "split": split,
                        "missing_count": missing_count,
                        "examples": missing[:20],
                    }
                )

            split_rows.append(
                {
                    "class": cls,
                    "split": split,
                    "split_count": len(split_ids),
                    "attr_count": len(attr_keys),
                    "split_minus_attr": len(split_minus_attr),
                    "attr_minus_split": len(attr_minus_split),
                    "data_minus_split": len(data_minus_split),
                    "data_minus_attr": missing_count,
                    "data_in_split_coverage": split_coverage,
                    "status": "PASS" if len(data_minus_split) == 0 else "WARN",
                }
            )

    return rows, missing_details, split_rows


def render_report(
    root: Path,
    rows: list[dict[str, object]],
    missing_details: list[dict[str, object]],
    split_rows: list[dict[str, object]],
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Data vs Attr Train/Val Consistency Report",
        "",
        f"- Generated at: {now}",
        f"- Workspace: {root}",
        "- Environment: conda env `mfr-py3`",
        "- Check rule: IDs in `data` (target + references) must exist in corresponding `attr` keys.",
        "",
        "## Results",
        "",
        "| class | split | data_count | attr_count | unique_ids_in_data | missing_in_attr | coverage | status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    for row in rows:
        coverage = row["coverage"]
        coverage_str = f"{coverage * 100:.2f}%" if isinstance(coverage, float) else str(coverage)
        lines.append(
            f"| {row['class']} | {row['split']} | {row['data_count']} | {row['attr_count']} | {row['unique_ids']} | {row['missing_count']} | {coverage_str} | {row['status']} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `unique_ids_in_data`: Number of distinct product IDs appearing in the data file (`target` + all `reference` IDs).",
            "- `data_count` and `attr_count` are not expected to be equal.",
            "- Correctness criterion: `missing_in_attr == 0`.",
            "",
            "## Image Splits Alignment",
            "",
            "- Check rule: all IDs used in `data` should be present in `image_splits/split.{class}.{split}.json`.",
            "- Also report relationship between split IDs and attr keys.",
            "",
            "| class | split | split_count | attr_count | split_minus_attr | attr_minus_split | data_minus_split | data_minus_attr | data_in_split_coverage | status |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )

    for row in split_rows:
        split_cov = row["data_in_split_coverage"]
        split_cov_str = f"{split_cov * 100:.2f}%" if isinstance(split_cov, float) else str(split_cov)
        lines.append(
            f"| {row['class']} | {row['split']} | {row['split_count']} | {row['attr_count']} | {row['split_minus_attr']} | {row['attr_minus_split']} | {row['data_minus_split']} | {row['data_minus_attr']} | {split_cov_str} | {row['status']} |"
        )

    lines.extend(
        [
            "",
            "## Missing ID Details",
            "",
        ]
    )

    if missing_details:
        for detail in missing_details:
            lines.append(f"### {detail['class']} / {detail['split']}")
            lines.append(f"- Missing count: {detail['missing_count']}")
            lines.append(f"- Example missing IDs (up to 20): `{', '.join(detail['examples'])}`")
            lines.append("")
    else:
        lines.append("- No missing IDs found. All checks passed.")
        lines.append("")

    lines.extend(
        [
            "## Reproduce",
            "",
            "```bash",
            f"cd {root}",
            "conda run -n mfr-py3 python scripts/check_data_attr_consistency.py",
            "```",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    docs_dir = root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    rows, missing_details, split_rows = compute_rows(root)
    report = render_report(root, rows, missing_details, split_rows)

    out_path = docs_dir / "data_attr_stats.md"
    out_path.write_text(report, encoding="utf-8")

    print(f"Wrote {out_path}")
    print("SUMMARY_START")
    for row in rows:
        coverage = row["coverage"]
        coverage_str = f"{coverage * 100:.2f}%" if isinstance(coverage, float) else str(coverage)
        print(
            f"{row['class']}\t{row['split']}\tdata={row['data_count']}\tattr={row['attr_count']}\t"
            f"unique={row['unique_ids']}\tmissing={row['missing_count']}\tcoverage={coverage_str}\tstatus={row['status']}"
        )
    print("SUMMARY_END")


if __name__ == "__main__":
    main()
