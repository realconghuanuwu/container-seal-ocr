from typing import Literal

from pydantic import BaseModel, Field


class Detection(BaseModel):
    x: int
    y: int
    width: int
    height: int
    confidence: float


class TextLine(BaseModel):
    polygon: list[list[int]] = Field(default_factory=list)
    text: str
    confidence: float


class OcrResponse(BaseModel):
    status: Literal["SUCCESS", "REVIEW", "RECAPTURE", "ERROR"]
    sealNumber: str | None = None
    rawText: str | None = None
    confidence: float | None = None
    source: Literal["OCR", "BARCODE"] | None = None
    processingTimeMs: int
    modelVersion: str
    reason: str | None = None
    imageWidth: int | None = None
    imageHeight: int | None = None
    detections: list[Detection] = Field(default_factory=list)
    textLines: list[TextLine] = Field(default_factory=list)


class Candidate(BaseModel):
    text: str
    confidence: float
    polygon: list[list[int]] = Field(default_factory=list)
