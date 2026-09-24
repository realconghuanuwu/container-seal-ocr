"""Build a reproducible recognition dataset from manually named full-frame photos."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.merge_manual_dataset import merge
from scripts.prepare_manual_seal_dataset import prepare


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="Full-frame JPG photos; filename stem is the manual label")
    parser.add_argument("--base", type=Path, required=True,
                        help="Frozen recognition-v3 dataset directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector", type=Path, default=Path("models/seal-detector-v1/best.onnx"))
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="manual-recognition-", dir=args.output.parent) as temp:
        manual = Path(temp) / "manual"
        prepare(args.input, manual, args.detector, args.seed, use_text_crop=True)
        result = merge(args.base, manual, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
