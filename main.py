# coding: utf8
import pandas as pd
import utils

loaded_df = utils.readfile("data/", "housing.csv")
print(loaded_df.head())
loaded_df2 = utils.readfile("C:/Users/Hanna/Documents/DSH/GitHub/Predictive/temperature/", "monthly_json.json")
print(loaded_df2.head())

if __name__ == '__main__':  # bei Direktaufruf soll das ausgeführt werden, nicht beim Import!
    print('Hello')
