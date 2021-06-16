import pandas as pd

print("Read xlsx")

xlsx = pd.read_excel('./datas.xlsx')

print(xlsx.head())
