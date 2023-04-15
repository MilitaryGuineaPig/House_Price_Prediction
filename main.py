import pandas as pd

data = pd.read_csv("/Users/monkey/Public/Python/Jop_prepar/Task_1/archive/AB_NYC_2019.csv")
pd.set_option('display.max_columns', None)

columns_name = data.columns
print(data.head())
print(columns_name)