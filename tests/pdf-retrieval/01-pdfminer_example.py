from pdfminer.high_level import extract_text
extracted_text = extract_text('../../data/2022_Annual_Report.pdf')

with open("../../data/pdfminer.txt", "w") as file:
    file.write(extracted_text)

print("String has been written to the file.")