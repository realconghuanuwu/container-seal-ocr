# Frozen benchmark history

All results use the same human-reviewed 300-image benchmark. The frozen manifest SHA-256 is `51d04479efd558e19e002481e288b1df6cc7cd461e3e26a9355e042e1407b3d7`.

| Date (UTC) | Pipeline milestone | Exact | Accuracy | CER | Avg ms | P95 ms |
|---|---|---:|---:|---:|---:|---:|
| 2026-09-17 | PP-OCRv6 Medium base | 113/300 | 37.67% | 0.4465 | 598.8 | 938.6 |
| 2026-09-18 | Recognition v1 | 117/300 | 39.00% | 0.3881 | 810.5 | 1234.8 |
| 2026-09-18 | Detector v1 + Recognition v1 | 128/300 | 42.67% | 0.3134 | 564.2 | 890.0 |
| 2026-09-19 | Detector v1 + Recognition v2 | 217/300 | 72.33% | 0.0988 | 610.0 | 859.8 |
| 2026-09-19 | v2 postprocessing checkpoint | 223/300 | 74.33% | 0.1027 | 658.9 | 1015.0 |
| 2026-09-19 | v2 + targeted horizontal TTA | 227/300 | 75.67% | 0.1020 | 645.0 | 1078.0 |
| 2026-09-20 | Production Release v1 (14.5k crops) | 248/300 | 82.67% | 0.0733 | 764.4 | 1204.7 |
| 2026-09-25 | **Production Release v1.0.4** (2-Stage + Dual Fallback + Stitching) | **275/300** | **91.67%** | **0.0298** | **820.5** | **1340.2** |

The canonical committed production release baseline is `models/seal-rec-v1.0.4`, trained with 2-Stage Sequential Curriculum (45 foundation epochs + 10 oversampled epochs) and powered by Dual-Crop Fallback and Prefix-Serial Multi-line Stitching. It achieves **91.67% exact match accuracy (275/300)** on the frozen real-world benchmark.

The previous v1 baseline is preserved in `baselines/seal-ocr-det-v1-rec-v1.{json,csv}` for historical reference. The complete generated report history is preserved in `reports/`.
