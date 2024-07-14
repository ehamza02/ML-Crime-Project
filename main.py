import pandas as pd
import numpy as np

file_path = "full_data.csv"

# These are the variables we want
filtered = [0, 1, 5, 7, 8, 9, 10, 11, 12, 13, 14, 17, 25, 33, 34, 35, 36, 37, 38, 96, 97, 110, 145, 146]

renamed = {
    0: 'City',
    1: 'State',
    5: 'population',
    7: 'Race: Black',
    8: 'Race: White',
    9: 'Race: Asian',
    10: 'Race: Hispanic',
    11: 'Population pct: 12-21',
    12: 'Population pct: 12-29',
    13: 'Population pct: 16-24',
    14: 'Population pct: 65+',
    17: 'Median Income',
    25: 'Per Capita Income',
    33: 'Pct under poverty line',
    34: 'Pct under 9th grade',
    35: 'Pct No Highschool',
    36: 'Pct: Higher Education',
    37: 'Pct: Unemployed',
    38: 'Pct: Employed',
    96: 'Number in shelters',
    97: 'Number on street',
    110: 'Police per Population',
    145: 'Violent Crime per population',
    146: 'Nonviolent crime per population'
}

df = pd.read_csv(file_path)
df_selected = df.iloc[:, filtered] 
df_selected.columns = renamed.values()

print(df_selected.head())

for col in df_selected:
    unique, counts = np.unique(df_selected[col].values, return_counts=True)
    num_missing = dict(zip(unique, counts)).get("?")

    if num_missing == None:
        num_missing = 0

    print("Column name: {0}, Missing: {1}".format(col, num_missing / len(df_selected[col].values)))