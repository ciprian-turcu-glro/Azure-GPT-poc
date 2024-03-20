from pdfminer.high_level import extract_text

# extract the text
extracted_text = extract_text("../../data/BD-D100_D120GV_XGV.pdf")

# writing the extracted text to a file
with open("../../data/D100_D120GV_XGV-pdfminer.txt", "w") as file:
    file.write(extracted_text)

print("String has been written to the file.")
