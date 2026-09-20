"""Upload Container Seal OCR to Hugging Face (Spaces or Model Hub).

Usage:
  1. Upload web demo & API to Hugging Face Spaces:
     python scripts/upload_to_hf.py --type space --repo-id <your-hf-username>/container-seal-ocr

  2. Upload model weights & model card to Hugging Face Models:
     python scripts/upload_to_hf.py --type model --repo-id <your-hf-username>/seal-ocr-rec-v1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, login


SPACE_README_FRONTMATTER = """---
title: Container Seal OCR
emoji: 🚢
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# 🚢 Container Seal OCR (Production v1)

Production OCR service for reading container seal numbers based on YOLOv8 Seal Detector and fine-tuned PP-OCRv6 SVTR_LCNet.

- **Accuracy (Exact Match)**: 82.67% (248/300 on frozen benchmark)
- **Character Error Rate (CER)**: 7.33%
- **Average Latency**: ~760ms / request
- **Interactive UI**: Supports Single OCR, Batch OCR, and 300-image Benchmark.

### API Usage

```bash
curl -X POST "https://{space_host}/api/v1/ocr/seal" \\
  -F "file=@seal.jpg;type=image/jpeg"
```
"""

MODEL_README = """---
language:
- vi
- en
license: mit
tags:
- ocr
- paddleocr
- container-seal
- logistics
- yolo
pipeline_tag: image-to-text
metrics:
- accuracy: 82.67
- cer: 7.33
---

# 🚢 Container Seal OCR Recognition Model v1

Fine-tuned recognition model for container seal characters (OOCL, SITC, Yang Ming, etc.) based on PP-OCRv6 SVTR_LCNet architecture.

## Model Performance (300 Frozen Benchmark)

| Metric | Base Model | Seal OCR Production v1 | Improvement |
| :--- | :---: | :---: | :---: |
| **Exact Match** | 37.67% (113/300) | **82.67%** (248/300) | **+45.00%** |
| **CER (Char Error Rate)** | 44.65% | **7.33%** | **-83.6% error** |
| **Review Cases** | 106 | **15** | **-85.8%** |
| **Average Latency** | 598 ms | **764 ms** | Production Ready |

## Files in this Repository

- `models/seal-ocr-rec-v1/`: PaddleOCR inference model (SVTR_LCNet).
  - `inference.pdiparams`
  - `inference.json`
  - `inference.yml`
  - `config.yml`
- `models/seal-detector-v1/best.onnx`: YOLOv8 seal bounding box detector (ONNX format).
"""


def main():
    parser = argparse.ArgumentParser(description="Upload to Hugging Face")
    parser.add_argument(
        "--type",
        choices=["space", "model"],
        default="space",
        help="Upload type: 'space' for live demo web/API, or 'model' for weights hub.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository ID on Hugging Face (e.g. your-username/container-seal-ocr)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face User Access Token (with WRITE permission). Defaults to $HF_TOKEN or stored token.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private.",
    )
    args = parser.parse_args()

    token = args.token
    if not token:
        token = os.getenv("HF_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token
            token = get_token()
        except Exception:
            token = None

    if not token:
        print("⚠️ Chưa tìm thấy Hugging Face Token.")
        print("Vui lòng lấy token WRITE tại: https://huggingface.co/settings/tokens")
        token = input("Nhập Hugging Face Write Token: ").strip()

    if not token:
        print("❌ Hủy bỏ: Cần token để xác thực với Hugging Face.")
        sys.exit(1)

    login(token=token, add_to_git_credential=False)
    api = HfApi(token=token)

    root_dir = Path(__file__).resolve().parent.parent

    if args.type == "space":
        print(f"🚀 Đang khởi tạo Space '{args.repo_id}' (Docker SDK)...")
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="space",
            space_sdk="docker",
            private=args.private,
            exist_ok=True,
        )

        space_readme = root_dir / "README_SPACE.md"
        with open(space_readme, "w", encoding="utf-8") as f:
            f.write(SPACE_README_FRONTMATTER)

        print(f"📦 Đang tải mã nguồn, models và giao diện lên Hugging Face Space '{args.repo_id}'...")
        api.upload_file(
            path_or_fileobj=str(space_readme),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="space",
        )

        ignore_patterns = [
            ".venv/**",
            "__pycache__/**",
            ".pytest_cache/**",
            "*.pyc",
            "tests/**",
            "training/**",
            "training_runs/**",
            "training_source/**",
            "recognition_dataset*/**",
            "*.zip",
            ".git/**",
            "README_SPACE.md",
        ]

        api.upload_folder(
            folder_path=str(root_dir),
            repo_id=args.repo_id,
            repo_type="space",
            ignore_patterns=ignore_patterns,
            commit_message="Deploy Container Seal OCR to Hugging Face Spaces",
        )

        if space_readme.exists():
            space_readme.unlink()

        user_name, space_name = args.repo_id.split("/")
        print("\n✅ ĐÃ TẢI LÊN HUGGING FACE SPACES THÀNH CÔNG!")
        print(f"🔗 Link Giao diện Web: https://huggingface.co/spaces/{args.repo_id}")
        print(f"🔗 Link Trực tiếp API:  https://{user_name}-{space_name}.hf.space/api/v1/ocr/seal")
        print(f"🔗 Link OpenAPI Docs:  https://{user_name}-{space_name}.hf.space/docs")

    elif args.type == "model":
        print(f"🚀 Đang khởi tạo Model Repository '{args.repo_id}'...")
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            private=args.private,
            exist_ok=True,
        )

        model_readme = root_dir / "README_MODEL.md"
        with open(model_readme, "w", encoding="utf-8") as f:
            f.write(MODEL_README)

        print(f"📦 Đang tải weights và cấu hình model lên '{args.repo_id}'...")
        api.upload_file(
            path_or_fileobj=str(model_readme),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
        )

        api.upload_folder(
            folder_path=str(root_dir / "models"),
            path_in_repo="models",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Upload Container Seal OCR weights and YOLOv8 detector",
        )

        if model_readme.exists():
            model_readme.unlink()

        print("\n✅ ĐÃ TẢI MODEL LÊN HUGGING FACE MODEL HUB THÀNH CÔNG!")
        print(f"🔗 Link Model Card: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
