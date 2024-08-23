import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2
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

# Dropping missing data
df_selected = df_selected.drop('Police_per_Population', axis=1)
df_selected = df_selected.query("Violent_Crime_per_Population != '?' and Nonviolent_Crime_per_Population != '?'")

# Convert all columns to numeric values
df_selected = df_selected.apply(pd.to_numeric, errors='ignore')

# Initialize scalers
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

targets = ['Violent_Crime_per_Population', 'Nonviolent_Crime_per_Population']
features = [col for col in df_selected.columns if col not in targets + ['City', 'State']]

X = df_selected[features]
y = df_selected[targets]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Architecture
model = Sequential()
model.add(Dense(16, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(2))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.01), loss=Huber(), metrics=['mae'])

# Training
history = model.fit(X_train, y_train, epochs=400, batch_size=32, validation_data=(X_test, y_test))

predictions = model.predict(X_test)

# Predictions and Evaluation
loss, mae = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test MAE: {mae}')

# Plot training & validation loss values
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Plot actual vs predicted values

plt.figure(figsize=(12, 6))
plt.plot(predictions[:, 0], label='Test Predictions')
plt.plot(y_test.to_numpy()[:, 0], label='Actual Values')
plt.title('Violent Crime per Population')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(predictions[:, 1], label='Test Predictions')
plt.plot(y_test.to_numpy()[:, 1], label='Actual Values')
plt.title('Non-Violent Crime per Population')
plt.legend()
plt.show()