"""
data_ingestion.py  –  PDF text extraction
------------------------------------------
Uses pypdf library to extract text from PDF documents.
"""

from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content from a PDF file.
    
    Parameters
    ----------
    pdf_path : str
        Path to the PDF file
        
    Returns
    -------
    str
        Concatenated text from all pages in the PDF
        
    Raises
    ------
    Exception
        If the PDF file cannot be read or parsed
    """
    text = ""

    reader = PdfReader(pdf_path)

    for page in reader.pages:
        extracted = page.extract_text()

        if extracted:
            text += extracted

    return text
