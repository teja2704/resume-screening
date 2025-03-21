# app/utils/parser.py

import pdfminer
from pdfminer.high_level import extract_text

def parse_resume(file):
    """
    Parse resume from a PDF file and extract text.
    """
    try:
        # Convert file-like object to bytes for pdfminer
        text = extract_text(file.stream)
        return {"text": text}
    except Exception as e:
        return {"error": str(e)}
