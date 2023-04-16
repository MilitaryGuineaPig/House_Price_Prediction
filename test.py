import data_cleaning
import pandas as pd


data = pd.read_csv("./archive/AB_NYC_2019.csv")

data_cleaning.clean_data(data)
