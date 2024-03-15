from pdfminer.high_level import extract_text
import fitz


def save_to_file(text, file_path):
    with open(file_path, "w") as file:
        file.write(text)


def extract_text_from_doc(method="pdfminer", path=None):
    text = ""
    if path != None:
        if method == "pdfminer":
            text = extract_text(path)
        elif method == "pymupdf":
            text = fitz.open(path)
    return text


# # local testing
# file_path = "./data/2022_Annual_Report.pdf"
# text = extract_text_from_doc(path=file_path)
# save_to_file(text=text, file_path="example-2022_Annual_Report.txt")
