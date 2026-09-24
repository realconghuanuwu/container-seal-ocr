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

## 2. Upload to Kaggle / Google Colab

Create a Kaggle Dataset (or upload to Colab) from the contents of:
`E:\Code\python\ocr\archive\recognition_dataset_rec_v1.0.3` (or zip file).

The dataset directory must contain:

```text
metadata.json
manifest.csv
train.txt
val.txt
images/
```

Total counts: 14,800 total, 13,237 train, and 1,563 validation samples.

## 3. Training on Kaggle / Google Colab (1 GPU or 2 GPUs)

Open `training/seal_ocr_finetune_standard.ipynb` in Kaggle or Google Colab.

### Hardware Selection (`GPU_MODE`):
- **1 GPU (`GPU_MODE = 1`)**:
  - **Recommended** for Google Colab, Kaggle single GPU (P100 / T4), or whenever you want 100% reliable training without distributed overhead.
  - Bypasses `paddle.distributed.launch` completely and runs `tools/train.py` directly.
  - Global batch size: 64 (or 32 if VRAM constrained).
- **2 GPUs (`GPU_MODE = 2`)**:
  - For Kaggle 2x T4 accelerator.
  - Uses `paddle.distributed.launch --gpus 0,1` with 32 batch size per card (= 64 global batch size).
  - **Hang Fixes Applied**: Automatically sets `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1` to prevent NCCL virtual PCIe deadlocks, and `num_workers=0` to prevent Docker shared memory IPC deadlocks.
  - Live stdout streaming is enabled so all iteration steps and losses show in real time in the notebook.

### Execution Steps:
1. Run once with `RUN_MODE = 'smoke'` to verify data loading, 1-epoch training, and model export.
2. Change `RUN_MODE = 'full'` for the 60-epoch control baseline followed by 8-epoch low-learning-rate fine-tuning.
   - Optional: set `SKIP_STAGE_1 = True` if you only want to fine-tune directly using the official pretrained checkpoint.
3. The notebook exports:
   - `seal-rec-v1.0.0-rebuilt.zip` (legacy: `seal-ocr-rec-v1-rebuilt.zip`)
   - `seal-rec-v1.0.3.zip` (legacy: `seal-ocr-rec-v5-candidate.zip`)
4. Download the exported models and benchmark against the benchmark suite. Promote `seal-rec-v1.0.3` if accuracy improves without regressions.
