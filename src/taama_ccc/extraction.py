from __future__ import annotations

import base64
from pathlib import Path

import pymupdf
from openai import OpenAI

from taama_ccc.config import Settings
from taama_ccc.models import ExtractedClaims

_EXTRACTION_SYSTEM_PROMPT = (
    "You extract product marketing/label claims for a regulatory compliance "
    "check. Given the input (a product label photo, a listing screenshot, a "
    "PDF page render, or raw text), do two things:\n"
    "\n"
    "1. Identify the product name if it is stated or clearly inferable.\n"
    "2. Enumerate every DISTINCT claim being made — health, function, "
    "ingredient-property, or marketing claims. Split compound sentences into "
    "separate claims when they assert separate things (e.g. 'boosts "
    "immunity and helps prevent colds' is two claims, not one). Do NOT "
    "include neutral facts with no claim content (a bare ingredient list, "
    "storage instructions, net weight) unless the fact itself asserts "
    "something.\n"
    "\n"
    "Transcribe claims close to their original wording — do not paraphrase "
    "away the specific language. Claim wording strength is itself "
    "compliance-relevant ('helps reduce the risk of X' and 'supports X' are "
    "different claim strengths, and softening one into the other during "
    "extraction would silently change the verdict downstream)."
)

_MULTI_FILE_INSTRUCTION = (
    "Extract the product name and every distinct claim visible across all "
    "images/pages above — treat them as one product, not separate items. "
    "Note any internal contradictions between images (e.g. a marketing "
    "claim vs. a disclaimer) as part of the claim you extract, if relevant."
)

_IMAGE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}

SUPPORTED_SUFFIXES = frozenset(_IMAGE_MEDIA_TYPES) | {".pdf"}


def _image_content_block(image_bytes: bytes, media_type: str) -> dict:
    encoded = base64.b64decode(image_bytes).decode("utf-8")

    return {
        "type": "input_image",
        "image_url": f"data:{media_type};base64,{encoded}",
    }


def _pdf_to_image_blocks(
    path: Path,
    *,
    max_pages: int = 3,
    dpi: int = 150,
) -> list[dict]:
    blocks: list[dict] = []
    doc = pymupdf.open(str(path))

    try:
        for page in doc[:max_pages]:
            pixmap = page.get_pixmap(dpi=dpi)
            blocks.append(_image_content_block(pixmap.tobytes("png"), "image/png"))
    finally:
        doc.close()

    return blocks


def extract_claims(
    client: OpenAI,
    settings: Settings,
    *,
    text: str | None = None,
    file_paths: list[Path] | None = None,
) -> ExtractedClaims:
    if (text is None) == (not file_paths):
        raise ValueError("provide exactly one of `text` or `file_paths`")

    content: list[dict] = []

    if text is not None:
        content.append(
            {
                "type": "input_text",
                "text": text,
            }
        )
    else:
        for file_path in file_paths:
            suffix = file_path.suffix.lower()

            if suffix == ".pdf":
                content.extend(_pdf_to_image_blocks(file_path))
            elif suffix in _IMAGE_MEDIA_TYPES:
                content.append(
                    _image_content_block(
                        file_path.read_bytes(), _IMAGE_MEDIA_TYPES[suffix]
                    )
                )
            else:
                raise ValueError(
                    f"unsupported file type for claim extraction: {suffix}"
                )

        content.append(
            {
                "type": "input_text",
                "text": _MULTI_FILE_INSTRUCTION,
            }
        )

    response = client.responses.parse(
        model=settings.openai_model,
        inputs=[
            {
                "role": "system",
                "content": _EXTRACTION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": content,
            },
        ],
        text_format=ExtractedClaims,
    )

    return response.output_parsed
