import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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

# using minmax and standard scaler
# min_max_cols = ['Median_Income', 'Per_Capita_Income']
# standard_cols = ['Num_in_Shelters', 'Num_on_Street']

# df_selected[min_max_cols] = min_max_scaler.fit_transform(df_selected[min_max_cols])
# df_selected[standard_cols] = standard_scaler.fit_transform(df_selected[standard_cols])

# remaining = ['Population', 'Race:Black', 'Race:White', 'Race:Asian', 'Race:Hispanic',
#              'Population_pct:12-21', 'Population_pct:12-29', 'Population_pct:16-24',
#              'Population_pct:65+', 'Pct_Under_Poverty_Line', 'Pct_Under_9th_Grade',
#              'Pct_No_Highschool', 'Pct_Higher_Education', 'Pct_Unemployed',
#              'Pct_Employed']

# df_selected[remaining] = min_max_scaler.fit_transform(df_selected[remaining])

targets = ['Violent_Crime_per_Population', 'Nonviolent_Crime_per_Population']
features = [col for col in df_selected.columns if col not in targets + ['City', 'State']]

X = df_selected[features]
X = min_max_scaler.fit_transform(X)
y = df_selected[targets]
y = min_max_scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create datasets and dataloaders
batch_size = 512

class CrimeDataset(Dataset):
    def __init__(self, data, label, idx):
        self.data = data
        self.label = label
        self.idx = idx

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, ind):
        x = self.data[ind]
        y = self.label[ind][self.idx]
        return x, y

train_loader = DataLoader(CrimeDataset(X_train, y_train, 0), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(CrimeDataset(X_test, y_test, 0), batch_size=batch_size, shuffle=False)

# TensorFlow
# # Model Architecture
# model = Sequential()
# model.add(Dense(16, input_dim=X_train.shape[1], activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(2))

# # Compile the model
# model.compile(optimizer=Adam(learning_rate=0.01), loss=Huber(), metrics=['mae'])

# # Training
# history = model.fit(X_train, y_train, epochs=400, batch_size=32, validation_data=(X_test, y_test))

# predictions = model.predict(X_test)

# # Predictions and Evaluation
# loss, mae = model.evaluate(X_test, y_test)
# print(f'Test Loss: {loss}')
# print(f'Test MAE: {mae}')

# # Plot training & validation loss values
# plt.figure(figsize=(12, 6))
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend()
# plt.show()

# # Plot actual vs predicted values

# plt.figure(figsize=(12, 6))
# plt.plot(predictions[:, 0], label='Test Predictions')
# plt.plot(y_test.to_numpy()[:, 0], label='Actual Values')
# plt.title('Violent Crime per Population')
# plt.legend()
# plt.show()

# plt.figure(figsize=(12, 6))
# plt.plot(predictions[:, 1], label='Test Predictions')
# plt.plot(y_test.to_numpy()[:, 1], label='Actual Values')
# plt.title('Non-Violent Crime per Population')
# plt.legend()
# plt.show()

# GPU(cuda) or CPU(cpu)
device = torch.device("cuda")

# Model definition
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 32)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(32)
        self.output = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.output(x)

        return x

model = MLP(X.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
criterion = nn.MSELoss()

# Model training
epochs = 10000

model.train()
losses = []
for epoch in range(epochs):
    epoch_loss = 0

    for _, input_data in enumerate(train_loader):
        optimizer.zero_grad()
        x, y = input_data
        x = x.float().to(device)
        y = y.float().to(device)

        output = model(x).float()
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() 
    losses.append(epoch_loss / len(train_loader))

    print('Epoch %d | Loss %6.4f' % (epoch, losses[epoch]))

# Look at predictions
model.eval()
predictions = []
actuals = []

for _, input_data in enumerate(test_loader):
    x, y = input_data
    x = x.float().to(device)
    y = y.float().to(device)

    with torch.no_grad():
        output = model(x).cpu().numpy()
    
    predictions.append(output)
    actuals.append(y.cpu().numpy())

predictions = np.concatenate(predictions, axis=0)
predictions = predictions.flatten().tolist()
actuals = np.concatenate(actuals, axis=0).tolist()

for i in range(len(predictions)):
    print(predictions[i], actuals[i])

# # Plot predictions vs actuals
# plt.figure(figsize=(12, 6))
# plt.scatter(actuals, predictions, alpha=0.5)
# plt.plot([min(actuals), max(actuals)],
#          [min(actuals), max(actuals)],
#          'r--', label='Perfect Prediction')
# plt.xlabel('Actual Values')
# plt.ylabel('Predicted Values')
# plt.title('Predictions vs Actuals')
# plt.legend()
# plt.show()