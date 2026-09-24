# Standard seal OCR training workflow

## 1. Add manually labeled photos

Put full-frame JPG photos in a folder. The filename stem must be the verified seal number:

```text
FX40295831.jpg
SITR722123.jpg
```

Run from `container-seal-ocr/`:

```powershell
python scripts/prepare_dataset.py `
  --input E:\Code\python\new-seal-images `
  --output E:\Code\python\prepared_dataset `
  --crop `
  --val-ratio 0.2
```

The script automatically validates seal numbers from filenames (or `labels.csv`), crops the seal using YOLO detector, and outputs standard PaddleOCR `images/`, `train.txt`, `val.txt`.

## 2. Upload to Kaggle

Create a Kaggle Dataset from the contents of `E:\Code\python\ocr\archive\ocr-service-artifacts\2026-09-21\seal-recognition-v4-manual\recognition_dataset_v4_complete`, or upload `E:\Code\python\ocr\archive\ocr-service-artifacts\2026-09-21\seal-recognition-v4-manual\recognition_dataset_v4_manual.zip` and unzip it in the notebook. The extracted root must contain:

```text
metadata.json
manifest.csv
train.txt
val.txt
images/
```

Expected v4 counts for the current dataset are 14,800 total, 13,237 train, and 1,563 validation samples.

## 3. Rebuild v1 checkpoint and fine-tune on two GPUs

Attach only `seal-recognition-v4-manual`. It already contains the full frozen v3 split (14,500 samples) plus the manual data. The reusable notebook reconstructs the original 13,000/1,500 v3 train/validation lists from `manifest.csv`, so a separate v3 Kaggle dataset is unnecessary.

Select Kaggle's two-GPU accelerator. Run once with `RUN_MODE = 'smoke'`. After both one-epoch stages complete, change it to `RUN_MODE = 'full'` for the 60-epoch control followed by an 8-epoch low-learning-rate fine-tune. Each GPU uses base batch 32, preserving the old global base batch of 64.

The notebook creates `seal-ocr-rec-v1-rebuilt.zip` and `seal-ocr-rec-v5-candidate.zip`. Both contain inference artifacts and resumable `checkpoints/best_accuracy.pdparams`, `.pdopt`, and `.states`. Preserve both ZIP files outside the repository.

Benchmark both exported inference folders against the frozen 300-image benchmark. Keep production v1 unless the rebuilt control is close to 248/300, and promote v5 only if it reaches at least 270/300 with no CER regression.
