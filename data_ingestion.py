from pypdf import PdfReader

def extract_text_from_pdf(pdf_path):

    text = ""

    reader = PdfReader(pdf_path)

    for page in reader.pages:
        extracted = page.extract_text()

        if extracted:
            text += extracted

    return text