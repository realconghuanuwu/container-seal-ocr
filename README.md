# Container Seal OCR

An application for recognizing **container seal numbers** from images. It automatically locates the seal, reads its barcode or printed characters, cleans the result, and evaluates its confidence.
[![Untitled.png](https://i.postimg.cc/2S5D0CR0/Untitled.png)](https://postimg.cc/4mjqdk3c)
## Features

- Recognize a single image or process multiple images from the web interface.
- Automatically detect and crop the seal region.
- Read barcodes first and fall back to OCR when no valid barcode is found.
- Handle rotated seals and difficult character spacing.
- Detect images that are too small, dark, bright, or blurry and request a new photo.
- Return the seal number, confidence, processing time, and detected region.
- Provide an HTTP API for integration with other systems.

## Quick start with Docker

Make sure [Docker](https://www.docker.com/) and Git LFS are installed, then run these commands from the project directory:

```powershell
docker build -t container-seal-ocr .
docker run -d --rm --name container-seal-ocr -p 8000:7860 container-seal-ocr
```

After the service starts, open:

- Web interface: <http://localhost:8000/test>
- API documentation: <http://localhost:8000/docs>
- Service status: <http://localhost:8000/health/ready>

To stop the application:

```powershell
docker stop container-seal-ocr
```

## Use the web interface

1. Open <http://localhost:8000/test>.
2. Select or drag and drop a seal image into the upload area.
3. Click **Recognize**.
4. Review the seal number, confidence, and result status.

The interface also supports batch recognition, result filtering, and CSV export.

Supported image formats are JPEG, PNG, and WebP. The default maximum file size is 10 MB.

## Use the API

Recognition endpoint:

```text
POST /api/v1/ocr/seal
```

PowerShell example:

```powershell
curl.exe -X POST http://localhost:8000/api/v1/ocr/seal `
  -F "file=@C:\images\seal.jpg;type=image/jpeg"
```

Example response:

```json
{
  "status": "SUCCESS",
  "sealNumber": "OOLKCK28059",
  "rawText": "OOLKCK28059",
  "confidence": 0.98,
  "source": "OCR",
  "processingTimeMs": 420,
  "modelVersion": "seal-ocr-det-v1-rec-v1",
  "reason": null,
  "imageWidth": 1280,
  "imageHeight": 720,
  "detections": []
}
```

### Result statuses

| Status | Meaning |
|---|---|
| `SUCCESS` | The seal was recognized with high confidence. |
| `REVIEW` | A seal number was found, but it should be checked manually. |
| `RECAPTURE` | The image quality is insufficient or no valid seal was found. Take another photo. |
| `ERROR` | The upload or recognition request failed. |

`source` is `BARCODE` when the result comes from a barcode and `OCR` when it comes from printed characters.

## Run with Python

Python 3.11 or newer is required:

```powershell
python -m pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Then open <http://localhost:8000/test>.

To change the model, image limits, or confidence thresholds, copy `.env.example` and set the required environment variables before starting the service.

## Run the tests

```powershell
python -m pip install -r requirements-test.txt
python -m pytest tests/ -q
```

## Prepare dataset for training

To prepare a training dataset in the simplest way possible:

### Option 1: Filename as label (Recommended)
Place seal photos in a folder and name each file with its seal number (e.g. `FX40295831.jpg`, `SITR722123.jpg`, `FX40295831_1.jpg`).

```powershell
python scripts/prepare_dataset.py --input path/to/images --output path/to/prepared_dataset
```

### Option 2: Folder of images + `labels.csv`
Place images in a folder and provide a 2-column CSV (`image,label`):

```powershell
python scripts/prepare_dataset.py --input path/to/images --output path/to/prepared_dataset
```

Add `--crop` to automatically crop seal regions using the YOLO detector. The script generates standard PaddleOCR `images/`, `train.txt`, `val.txt`, and `metadata.json`.

