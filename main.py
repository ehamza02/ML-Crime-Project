import pandas as pd
import numpy as np
import math
import Layer 
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
    17: 'Median_Income', # rescale 
    25: 'Per_Capita_Income', # rescale 
    33: 'Pct_Under_Poverty_Line',
    34: 'Pct_Under_9th_Grade',
    35: 'Pct_No_Highschool',
    36: 'Pct_Higher_Education',
    37: 'Pct_Unemployed',
    38: 'Pct_Employed',
    96: 'Num_in_Shelters', # standardize/normalize
    97: 'Num_on_Street', # standardize/normalize
    110: 'Police_per_Population',
    145: 'Violent_Crime_per_Population',
    146: 'Nonviolent_Crime_per_Population'
}

df = pd.read_csv(file_path)
df_selected = df.iloc[:, filtered] 

# Data preprocessing
df_selected.columns = renamed.values()

# df_selected['Median_Income'] = df_selected['Median_Income'] / 100
# df_selected['Per_Capita_Income'] = df_selected['Per_Capita_Income'] / 100

# df_selected['Num_in_Shelters'] = (df_selected['Num_in_Shelters'] - df_selected['Num_in_Shelters'].mean()) / df_selected['Num_in_Shelters'].std()
# df_selected['Num_on_Street'] = (df_selected['Num_on_Street'] - df_selected['Num_on_Street'].mean()) / df_selected['Num_on_Street'].std()

# print(df_selected.head())

# Searching for missing data
for col in df_selected:
    unique, counts = np.unique(df_selected[col].values, return_counts=True)
    num_missing = dict(zip(unique, counts)).get("?")

    if num_missing == None:
        num_missing = 0

    # print("Column name: {0}, Missing: {1}".format(col, num_missing / len(df_selected[col].values)))

# Dropping missing data
df_selected = df_selected.drop('Police_per_Population', axis=1)
df_selected = df_selected.query("Violent_Crime_per_Population != '?' and Nonviolent_Crime_per_Population != '?'")

print(df_selected.head())

# Convert all columns to numeric values
df_selected = df_selected.apply(pd.to_numeric, errors='ignore')

# print(df_selected.to_numpy())

# 1902 x 23
# Input layer: 19 x 1
# Output layer: 2 x 1 or 1 x 1
# Number of total examples = 1902 (80/20 train/test split) 


# print(df_selected.iloc[:, 2:21].to_numpy())
# print(df_selected.iloc[:, 21:22].to_numpy())

min_max_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

# using minmax and standard scaler
min_max_cols = ['Median_Income', 'Per_Capita_Income']
standard_cols = ['Num_in_Shelters', 'Num_on_Street']

df_selected[min_max_cols] = min_max_scaler.fit_transform(df_selected[min_max_cols])
df_selected[standard_cols] = standard_scaler.fit_transform(df_selected[standard_cols])

remaining = ['Population', 'Race:Black', 'Race:White', 'Race:Asian', 'Race:Hispanic',
             'Population_pct:12-21', 'Population_pct:12-29', 'Population_pct:16-24',
             'Population_pct:65+', 'Pct_Under_Poverty_Line', 'Pct_Under_9th_Grade',
             'Pct_No_Highschool', 'Pct_Higher_Education', 'Pct_Unemployed',
             'Pct_Employed']

df_selected[remaining] = min_max_scaler.fit_transform(df_selected[remaining])

# targets = ['Violent_Crime_per_Population', 'Nonviolent_Crime_per_Population']
targets = ['Violent_Crime_per_Population', 'Nonviolent_Crime_per_Population']
features = [col for col in df_selected.columns if col not in targets + ['City', 'State']]

X = df_selected[features].to_numpy()
print(np.shape(X))
y = df_selected[targets[0]].to_numpy()
print(np.shape(y))


NN = Layer.NeuralNetwork("ReLU", 8, [19, 17, 15, 13, 11, 9, 7, 5, 3, 1], X, y)

# NN = Layer.NeuralNetwork("ReLU", 1, [19, 22, 1], df_selected.iloc[:, 2:21].to_numpy(), df_selected.iloc[:, 21:22].to_numpy())

# NN = Layer.NeuralNetwork("ReLU", 0, [19, 1], df_selected.iloc[:, 2:21].to_numpy(), df_selected.iloc[:, 21:22].to_numpy())

# First example
# NN = Layer.NeuralNetwork("ReLU", 8, [19, 17, 15, 13, 11, 9, 7, 5, 3, 1], df_selected.iloc[:, 2:21].head(1).to_numpy(), df_selected.iloc[:, 21:22].head(1).to_numpy())


NN.train(learn_rate=0.001, max_iter=200)
print(NN.test())