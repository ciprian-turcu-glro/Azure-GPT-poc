import fitz


# extracting the text from a file path
def read_pdf_with_pyMuPDF(file_path):
    text = ""
    try:
        # Open the PDF file
        with fitz.open(file_path) as pdf_file:
            # Iterate through each page
            for page_number in range(pdf_file.page_count):
                # Get the current page
                page = pdf_file.load_page(page_number)
                # Extract text from the current page
                text += page.get_text()
    except Exception as e:
        print(f"Error reading PDF file: {e}")

    return text


pdf_file_path = "../../data/BD-D100_D120GV_XGV.pdf"
pdf_text = read_pdf_with_pyMuPDF(pdf_file_path)

# writing the extracted text to a file
with open("../../data/D100_D120GV_XGV-pymupdf.txt", "w") as file:
    file.write(pdf_text)

print("String has been written to the file.")
