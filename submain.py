import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

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

df_selected.columns = renamed.values()

# Initialize scalers
min_max_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

# Searching for missing data
for col in df_selected:
    unique, counts = np.unique(df_selected[col].values, return_counts=True)
    num_missing = dict(zip(unique, counts)).get("?")

    if num_missing == None:
        num_missing = 0

# using minmax and standard scaler, but we can change back to what we did earlier
min_max_cols = ['Median_Income', 'Per_Capita_Income']

standard_cols = ['Num_in_Shelters', 'Num_on_Street']

df_selected[min_max_cols] = min_max_scaler.fit_transform(df_selected[min_max_cols])

df_selected[standard_cols] = standard_scaler.fit_transform(df_selected[standard_cols])

remaining = ['Population', 'Race_Black', 'Race_White', 'Race_Asian', 'Race_Hispanic',
                  'Population_pct_12_21', 'Population_pct_12_29', 'Population_pct_16_24',
                  'Population_pct_65_up', 'Pct_Under_Poverty_Line', 'Pct_Under_9th_Grade',
                  'Pct_No_Highschool', 'Pct_Higher_Education', 'Pct_Unemployed',
                  'Pct_Employed', 'Police_per_Population']

df_selected[remaining] = min_max_scaler.fit_transform(df_selected[remaining])

targets = ['Violent_Crime_per_Population', 'Nonviolent_Crime_per_Population']
features = [col for col in df_selected.columns if col not in targets + ['City', 'State']]

X = df_selected[features]
y = df_selected[targets]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='relu'))




