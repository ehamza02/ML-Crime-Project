import pandas as pd
import numpy as np
import Layer 

# Loading data
file_path = "full_data.csv"

# These are the variables we want
filtered = [0, 1, 5, 7, 8, 9, 10, 11, 12, 13, 14, 17, 25, 33, 34, 35, 36, 37, 38, 96, 97, 110, 145, 146]

renamed = {
    0: 'City',
    1: 'State',
    5: 'Population',
    7: 'Race:Black',
    8: 'Race:White',
    9: 'Race:Asian',
    10: 'Race:Hispanic',
    11: 'Population_pct:12-21',
    12: 'Population_pct:12-29',
    13: 'Population_pct:16-24',
    14: 'Population_pct:65+',
    17: 'Median_Income',
    25: 'Per_Capita_Income',
    33: 'Pct_Under_Poverty_Line',
    34: 'Pct_Under_9th_Grade',
    35: 'Pct_No_Highschool',
    36: 'Pct_Higher_Education',
    37: 'Pct_Unemployed',
    38: 'Pct_Employed',
    96: 'Num_in_Shelters',
    97: 'Num_on_Street',
    110: 'Police_per_Population',
    145: 'Violent_Crime_per_Population',
    146: 'Nonviolent_Crime_per_Population'
}

df = pd.read_csv(file_path)
df_selected = df.iloc[:, filtered] 

# Data preprocessing
df_selected.columns = renamed.values()

print(df_selected.head())

# Searching for missing data
for col in df_selected:
    unique, counts = np.unique(df_selected[col].values, return_counts=True)
    num_missing = dict(zip(unique, counts)).get("?")

    if num_missing == None:
        num_missing = 0

    print("Column name: {0}, Missing: {1}".format(col, num_missing / len(df_selected[col].values)))

# Dropping missing data
df_selected = df_selected.drop('Police_per_Population', axis=1)
df_selected = df_selected.query("Violent_Crime_per_Population != '?' and Nonviolent_Crime_per_Population != '?'")

print(df_selected.shape)

# 1902 x 23
# Input layer: 19 x 1
# Output layer: 2 x 1 or 1 x 1
# Number of total examples = 1902 (80/20 train/test split) 

NN = Layer.NeuralNetwork("ReLU", 1, [19, 22, 1], df_selected.iloc[:, 2:21].to_numpy, df_selected.iloc[:, 21:22].to_numpy)