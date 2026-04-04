import base64
import io
import os
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import pdfplumber
from google.genai import Client, types
from tidylog import get_logger

logger = get_logger(__name__)


def classify_pdf_sync(path: Path) -> Dict[str, Any]:
    """Classify PDF content synchronously using pdfplumber."""
    has_text = False
    has_image = False

    try:
        with pdfplumber.open(path) as pdf:
            total_pages = len(pdf.pages)

            for page in pdf.pages:
                if not has_text:
                    text = page.extract_text() or ""
                    if text.strip():
                        has_text = True

                if not has_image:
                    if page.images:
                        has_image = True

        return {
            "has_text": has_text,
            "has_image": has_image,
            "total_pages": total_pages,
        }

    except Exception as exc:
        logger.error("Failed to classify PDF", exc_info=exc)
        return {
            "has_text": False,
            "has_image": False,
            "total_pages": 0,
        }


def extract_images_base64(pdf_path: str) -> Dict[int, List[str]]:
    extracted = defaultdict(list)
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                for img_index, img in enumerate(page.images):
                    try:
                        # Safest method: Crop the page at the image's coordinates
                        # and convert it to a PIL image to guarantee valid formatting
                        bbox = (img["x0"], img["top"], img["x1"], img["bottom"])
                        pil_img = page.crop(bbox).to_image(resolution=150).original

                        # Save the PIL image into a byte stream as a valid JPEG
                        img_byte_arr = io.BytesIO()
                        pil_img.save(img_byte_arr, format="JPEG")
                        img_bytes = img_byte_arr.getvalue()

                        # Encode to base64 string
                        extracted[page_num].append(
                            base64.b64encode(img_bytes).decode("utf-8")
                        )

                    except Exception as exc:
                        logger.error(
                            "Failed to extract image",
                            extra={"page": page_num, "img_idx": img_index},
                            exc_info=exc,
                        )
                        continue

    except Exception as exc:
        logger.error("Error opening PDF with pdfplumber", exc_info=exc)

    return extracted


def get_pdf_page_to_image_map(
    pdf_path: Path,
) -> Dict[int, list[Dict[str, Any]]]:
    """Extract and describe all images in a PDF using Gemini vision."""
    extracted: Dict[int, list[str]] = extract_images_base64(str(pdf_path))

    if not extracted:
        logger.info("No images found in PDF", extra={"path": pdf_path})
        return {}

    client = Client(api_key=os.environ.get("GEMINI_API_KEY"))
    results = defaultdict(list)

    for page_num, img_b64_list in extracted.items():
        for img_index, img_b64 in enumerate(img_b64_list, start=1):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[
                        types.Part.from_bytes(
                            data=base64.b64decode(img_b64),
                            mime_type="image/jpeg",
                        ),
                        textwrap.dedent(
                            """
                            Analyze the image and provide a comprehensive description in pure plain text.

                            Focus your analysis on the following:
                            - What the image visually shows
                            - Any visible text (OCR)
                            - Structural elements like charts, diagrams, tables, or UI components
                            - Inferred visual metadata (e.g., style, medium, lighting)
                            - The likely purpose or context of the image
                            - Any uncertainties or ambiguities in the image

                            STRICT CONSTRAINTS:
                            - Return ONLY plain text.
                            - Do NOT use any Markdown formatting whatsoever (no asterisks, no hash symbols, no bullet points, no bold/italics).
                            - Do NOT return JSON, dictionaries, or any structured data formats.
                            - Do NOT invent, guess, or hallucinate EXIF or file metadata.
                            - Provide the final output as standard, readable paragraphs.
                            """
                        ),
                    ],
                )

                results[page_num].append(
                    {"img_index": img_index, "description": response.text}
                )

                logger.info(
                    "Image described",
                    extra={"page": page_num, "img_idx": img_index},
                )

            except Exception as exc:
                logger.error(
                    "Failed to describe image",
                    extra={"page": page_num, "img_idx": img_index},
                    exc_info=exc,
                )

    return dict(results)
