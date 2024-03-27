import tabula
  
# Path to your pdf  
file = "../../data/BD-D100_D120GV_XGV.pdf"
  
# Read pdf into list of DataFrame  
dfs = tabula.read_pdf(file, pages=36)  
  
# Now dfs is a list of dataframes, where each dataframe corresponds to a table.  
# Loop through the list and print the dataframes  
for i, df in enumerate(dfs):  
    print(f"Table {i+1}:\n{df}")  
