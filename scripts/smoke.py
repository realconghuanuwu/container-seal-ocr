import argparse
import time
from pathlib import Path

import httpx


def run(image: Path, base_url: str) -> None:
    with httpx.Client(timeout=15) as client:
        for _ in range(120):
            try:
                if client.get(f"{base_url}/health/ready").status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(1)
        else:
            raise RuntimeError("OCR model did not become ready within 120 seconds")
        with image.open("rb") as source:
            response = client.post(
                f"{base_url}/api/v1/ocr/seal",
                files={"file": (image.name, source, "image/jpeg")},
            )
        response.raise_for_status()
        result = response.json()
        assert result["status"] in {"SUCCESS", "REVIEW", "RECAPTURE"}
        assert result["modelVersion"]
        print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, help="Close-up JPEG seal image")
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()
    run(args.image, args.base_url)
