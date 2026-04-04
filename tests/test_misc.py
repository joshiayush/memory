import logging
import os

import pytest
from dotenv import load_dotenv

load_dotenv(".env")

from livekit_memory.content.loaders import _read_pdf_sync
from livekit_memory.utils.misc import extract_images_base64

logger = logging.getLogger(__name__)

# Create a path relative to the current script's directory
script_dir = os.path.dirname(__file__)
rel_path = "pdfs\\attention-is-all-you-need.pdf"
abs_file_path = os.path.join(script_dir, rel_path)


def test_extract_images_base64_no_images():
    """Test extracting images from a PDF with no images."""
    images = extract_images_base64(abs_file_path)
    assert len(images) == 2

    # checking no. of images in a specfic page
    for i in images:
        # third page contains 1 image
        if i == 3:
            assert len(images[i]) == 1
        # fourth page contains 2 images
        elif i == 4:
            assert len(images[i]) == 2


def test_read_pdf_sync():
    result = _read_pdf_sync(abs_file_path)

    assert "[Page: 3, Image 1 Description]" in result
    assert "[Page: 4, Image 1 Description]" in result
    assert "[Page: 4, Image 2 Description]" in result
