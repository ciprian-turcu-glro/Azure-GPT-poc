import fitz

def read_pdf(file_path):
    text = ""
    try:
        # Open the PDF file
        with fitz.open(file_path) as pdf_file:
            # Iterate through each page
            for page_number in range(pdf_file.page_count):
                # Get the page
                page = pdf_file.load_page(page_number)
                # Extract text from the page
                text += page.get_text()
    except Exception as e:
        print(f"Error reading PDF file: {e}")
    
    return text
pdf_file_path = "../../data/2022_Annual_Report.pdf"
pdf_text = read_pdf(pdf_file_path)
with open("../../data/pymupdf_contents.txt", "w") as file:
    file.write(pdf_text)

print("String has been written to the file.")
