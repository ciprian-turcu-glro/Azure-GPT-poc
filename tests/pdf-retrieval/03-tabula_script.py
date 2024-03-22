import tabula

# Read pdf into list of DataFrame
# dfs = tabula.read_pdf("../../data/BD-D100_D120GV_XGV.pdf", pages='36')
# with open("../../data/tabula.txt", "w") as file:
#     file.write(dfs)
tabula.convert_into("../../data/BD-D100_D120GV_XGV.pdf", "../../data/tabula.csv", output_format="csv", pages='36')
# tell me what is the Duvet washing ammount set in the table data extracted in this csv format: