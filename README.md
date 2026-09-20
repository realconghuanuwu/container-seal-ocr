# Container Seal OCR

Internal FastAPI service for reading container seal numbers. The current production pipeline uses the custom seal detector plus Recognition Model v1 (Production Release), with barcode fast-path, rotation TTA, horizontal TTA and rule-based postprocessing. The HTTP API and response schema remain stable.

## Run the current model

Build the image from this directory:

```powershell
docker build -t seal-ocr:det-v1-rec-v1 .
```

The model binaries are intentionally not stored in Git. Restore `models/seal-detector-v1/best.onnx` and the Paddle inference files under `models/seal-ocr-rec-v1/`, then start the service:

```powershell
docker stop seal-ocr-local 2>$null
docker run -d --rm --name seal-ocr-local `
  -p 127.0.0.1:8000:8000 `
  -v "${PWD}/models/seal-detector-v1:/models/seal-detector-v1:ro" `
  -v "${PWD}/models/seal-ocr-rec-v1:/models/seal-ocr-rec-v1:ro" `
  -e SEAL_DETECTOR_MODEL=/models/seal-detector-v1/best.onnx `
  -e RECOGNITION_MODEL_DIR=/models/seal-ocr-rec-v1 `
  -e MODEL_VERSION=seal-ocr-det-v1-rec-v1 `
  seal-ocr:det-v1-rec-v1
```

Wait for `http://localhost:8000/health/ready` to return `status: healthy` and `modelVersion: seal-ocr-det-v1-rec-v1`. The browser test page is at `http://localhost:8000/test`; OpenAPI documentation is at `http://localhost:8000/docs`.

The image has one Uvicorn process and one long-lived OCR worker. A simultaneous inference can return HTTP 503 `SERVICE_BUSY`. A request exceeding the nine-second processing deadline returns HTTP 504 `OCR_TIMEOUT` and restarts the worker. Keep the service on a private network and configure callers for the request deadline plus upload time.

### Environments & Conventions

The system uses a single unified `.env` configuration file (copied from [`.env.example`](.env.example)):
- Set `APP_ENV=prod` (default) for Production release (`seal-ocr-det-v1-rec-v1`, port `8000`).
- Set `APP_ENV=uat` for QA/Staging testing (`seal-ocr-det-v1-rec-v1-uat`, port `8080`).
- Set `APP_ENV=dev` for Development/R&D (`seal-ocr-det-v1-rec-v1-dev`, port `8001`).

See [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md) for full naming conventions across models, Docker containers, ports, and Git branches.

## API

```powershell
curl.exe -X POST http://localhost:8000/api/v1/ocr/seal `
  -H "X-Request-ID: example-123" `
  -F "file=@path/to/seal.jpg;type=image/jpeg"
```

The response status is `SUCCESS`, `REVIEW`, `RECAPTURE`, or `ERROR`. `RECAPTURE` is HTTP 200 with a machine-readable reason; invalid uploads use 400/413/415, and busy or timed-out requests use 503/504. Barcode results use `source: BARCODE` and a null confidence. The service preserves OCR confidence for status classification and does not expose internal candidate scores.

Useful operations:

```powershell
docker logs -f seal-ocr-local
docker stop seal-ocr-local
```

## Tests and frozen benchmark

```powershell
python -m pip install -r requirements-test.txt
python -m pytest -q
python -m scripts.benchmark --freeze
```

The frozen evaluation set is the 300 images in `benchmark/images/` plus `benchmark/labels.csv`. `benchmark/manifest.json` protects their hashes. Never use these images for training or tuning.

The canonical Production Release v1 result is committed as:

- `baselines/seal-ocr-det-v1-rec-v1.json`
- `baselines/seal-ocr-det-v1-rec-v1.csv`

The previous v2 result is kept in `baselines/seal-ocr-det-v1-rec-v2.{json,csv}` for historical regression tracking.

Run a comparable benchmark against the active service with an explicit immutable baseline path:

```powershell
python -m scripts.benchmark `
  --expected-model seal-ocr-det-v1-rec-v1 `
  --model-dir models/seal-ocr-rec-v1 `
  --detector-model models/seal-detector-v1/best.onnx `
  --compare-with baselines/seal-ocr-det-v1-rec-v1
```

Generated reports are written under ignored `reports/`. Historical milestones are summarized in `docs/BENCHMARK_HISTORY.md`.

## Dataset and model artifacts

Training data, intermediate review state, legacy reports and model binaries live outside Git. Their authoritative relative paths, file counts and SHA-256 hashes are recorded in `artifacts/datasets.json`. A local archive bundle can be placed in an external directory, for example:

```text
data/archive/ocr-service-artifacts/2026-09-19
```

Verify an archive with its adjacent `.sha256` file before extraction. To restore and validate Recognition Dataset v2:

```powershell
Expand-Archive `
  data/archive/ocr-service-artifacts/2026-09-19/recognition_dataset_v2.zip `
  recognition_dataset_v2
python -m scripts.training_model validate --dataset recognition_dataset_v2
```

The v2 creation, review and finalization tools remain available as the frozen input to v3.

## Recognition Dataset v3

Recognition v3 combines the 5,000 verified v2 crops with 15,000 new crops. Staging and working directories stay outside Git (e.g. under `data/` or a custom workspace):

```text
data/recognition_dataset_v3_work
data/recognition_dataset_v3
```

Restore and validate v2 before running the pipeline. The full AGY pilot is a hard gate: `build` refuses to run until `agy_profile.json` records a passing 500-image calibration.

```powershell
# Fast media-transport check; this does not unlock production labeling.
python -m scripts.create_recognition_dataset_v3 pilot `
  --v2-dataset data/recognition_dataset_v2 `
  --transport-only --smoke-count 20 --transport-batch-size 4

# Quality calibration: tries batch 8, then 4, then 1 and stops at the first passing size.
python -m scripts.create_recognition_dataset_v3 pilot `
  --v2-dataset data/recognition_dataset_v2

# Resumable source scan, crop extraction and AGY/v2 consensus labeling.
python -m scripts.create_recognition_dataset_v3 build `
  --v2-dataset data/recognition_dataset_v2

python -m scripts.create_recognition_dataset_v3 status
```

AGY runs inside a dedicated staging project with only copied crops. The pipeline uses a scoped Antigravity project rather than `--dangerously-skip-permissions`; its project id is cached under the ignored work directory.

Only unresolved OCR disagreements appear in the reviewer:

```powershell
python -m scripts.review_tool_v2 `
  --work-dir data/recognition_dataset_v3_work `
  --source-dir data/new_dataset
```

Finalize only after all three new-data quotas are complete:

```powershell
python -m scripts.finalize_dataset_v3 `
  --v2-dataset data/recognition_dataset_v2

python -m scripts.training_model validate `
  --dataset data/recognition_dataset_v3
```

Upload `recognition_dataset_v3.zip` as the private Kaggle dataset `seal-recognition-v3-20k`, attach it to `training/phase4_kaggle_v3.ipynb`, and run `control` and `primary` in separate sessions. Set `SMOKE_EPOCHS = 1` first; set it back to `0` for the 60-epoch run. Only run `spacing-050` and then `spacing-025` if the first two experiments miss the frozen-benchmark gate.

After downloading the selected model, extract it under ignored `models/seal-ocr-rec-v1/`, set `RECOGNITION_MODEL_DIR` and `MODEL_VERSION=seal-ocr-det-v1-rec-v1`, then run the same immutable 300-image benchmark.

## Local development

Use Python 3.11 or newer:

```powershell
python -m pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

PaddlePaddle may require its platform-specific CPU wheel. Copy settings from `.env.example` into the process environment. Model and dataset directories are ignored so local restoration cannot accidentally add them to Git.

The repository history was rewritten during the 2026-09-19 cleanup. Old checkouts must be replaced with a fresh clone rather than merged into the rewritten `master`. The recovery bundle named in `artifacts/datasets.json` can restore the pre-cleanup history.
