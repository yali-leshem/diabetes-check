import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

class data(torch.utils.data.Dataset):
    def __init__(self, add_precents, data_src, parameters, target_cols):
        # import data from csv file if necessary
        self.data = data_src
        # store info about data source
        self.parameters = parameters   
        self.target_cols = target_cols
        self.data['Y'] = self.data['Y'].rank(method='first')
        if add_precents == 0:
            self.data['Class'] = pd.qcut(self.data['Y'], 100, labels = False) + 1
        else:
            self.data['Class'] = pd.qcut(self.data['Y'], 10, labels = False, duplicates='drop') + 1
        

    def __getitem__(self, idx):
        # get the parameters x and the target y
        x = self.data[self.parameters].iloc[idx]
        y = self.data[self.target_cols].iloc[idx]
        x = torch.tensor(x.values, dtype=torch.float32)
        y = torch.tensor(y.values, dtype=torch.float32)

        return x, y

    def __len__(self):
        return self.data.shape[0] # Returning length (shape)
    
class dataY(torch.utils.data.Dataset):
    def __init__(self, add_precents, data_src, parameters, target_cols):
        # import data from csv file if necessary
        self.data = data_src
        # store info about data source  
        self.target_cols = target_cols
        self.parameters = parameters
        self.data['Y'] = self.data['Y'].rank(method='first')
        if add_precents == 0:
                self.data['Class'] = pd.qcut(self.data['Y'], 100, labels = False) + 1
        else:
                self.data['Class'] = pd.qcut(self.data['Y'], 10, labels = False, duplicates='drop') + 1
        
    def __getitem__(self, idx):
        # get the parameters x and the target y
        x = self.data[self.parameters].iloc[idx]
        y = self.data[self.target_cols].iloc[idx]
        x = torch.tensor(x.values, dtype=torch.float32)
        y = torch.tensor(y.values, dtype=torch.float32)

        return x, y

    def __len__(self):
        return self.data.shape[0] # Returning length (shape)

def train_model(model, dataloader, epochs, optimizer, loss_fn):
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device).long().squeeze()
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

def evaluate_model(model, dataloader):
    model.eval()  # Set the model to evaluation mode
    total, correct = 0, 0
    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device).long().squeeze()
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return 100 * correct / total

# Model without Y definition
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.Sequential(
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.LogSoftmax(dim=1)
).to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelY = nn.Sequential(
    nn.Linear(11, 10),  # Adjust input size to match the number of input features
    nn.ReLU(),
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.LogSoftmax(dim=1)
).to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelPrecents = nn.Sequential(
    nn.Linear(10, 100),  # Adjust input size to match the number of input features
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.LogSoftmax(dim=1)
).to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelPrecentsY = nn.Sequential(
    nn.Linear(11, 100),  # Adjust input size to match the number of input features
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.LogSoftmax(dim=1)
).to(device)

# Data loading
csv_file = r"C:\Users\zlesh\Downloads\diabetes.csv"
dataSet = pd.read_csv(csv_file, sep = '\t')
train_data, test_data = train_test_split(dataSet, train_size = 0.8, shuffle = True) # Load random parts of dataset
parameters = ['AGE', 'SEX', 'BMI', 'BP', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6']
target_cols = ['Class']
diabetes_dataset = data(1, train_data, parameters, target_cols)
diabetes_dataloader = DataLoader(diabetes_dataset, batch_size = 10)
stats, targets = next(iter(diabetes_dataloader))
print("First Mini-Batch example: ")
print(stats)
print(targets)

epochs = 750 # number of single passes on the network
loss_fn = nn.NLLLoss(ignore_index = 10)
loss_fnPrecents = nn.NLLLoss(ignore_index = 100) # negative log likehood loss function while ignoring the precentiles classes
optimizer = torch.optim.SGD(model.parameters(), lr=0.00035) # optimizer for gradient decent with a learning rate of 0.01

# Training for Class without 'Y' 
train_model(model, diabetes_dataloader, epochs, optimizer, loss_fn)
accuracy = evaluate_model(model, diabetes_dataloader)
print(f"Accuracy of the model excluding 'Y' column on the TRAINED dataset: {accuracy:.2f}%")

parameters.append('Y')
diabetesY_dataset = dataY(1, train_data, parameters, target_cols)
diabetesY_dataloader = DataLoader(diabetesY_dataset, batch_size = 10)
stats, targets = next(iter(diabetesY_dataloader))
# Training for Class with 'Y'
train_model(modelY, diabetesY_dataloader, epochs, optimizer, loss_fn)
accuracy = evaluate_model(modelY, diabetesY_dataloader)
print(f"Accuracy of the model including 'Y' column on the TRAINED dataset: {accuracy:.2f}%")

# Training with remaining random 20% of the database
parameters.remove('Y')
TrainData = data(1, test_data, parameters, target_cols)
diabetes_dataloader = DataLoader(TrainData, batch_size = len(TrainData))
DataTest, TargetsTest = next(iter(diabetes_dataloader))
DataTest = DataTest.to(device)
TargetsTest = TargetsTest.to(device)
accuracy = evaluate_model(model, diabetes_dataloader)
print(f"Accuracy of the model excluding 'Y' column on the TESTED dataset: {accuracy:.2f}%") # test data without 'Y'

parameters.append('Y')
TrainYData = dataY(1, test_data, parameters, target_cols)
diabetesY_dataloader = DataLoader(TrainYData, batch_size = len(TrainYData))
DataYTest, TargetsYTest = next(iter(diabetesY_dataloader))
accuracy = evaluate_model(modelY, diabetesY_dataloader)
print(f"Accuracy of the model including 'Y' column on the TESTED dataset: {accuracy:.2f}%") # test data with 'Y'

parameters.remove('Y')
diabetes_dataset = data(0, train_data, parameters, target_cols)
diabetes_dataloader = DataLoader(diabetes_dataset, batch_size = 10)
stats, targets = next(iter(diabetes_dataloader))

# Training for Class without 'Y' 
train_model(modelPrecents, diabetes_dataloader, epochs, optimizer, loss_fnPrecents)
accuracy = evaluate_model(modelPrecents, diabetes_dataloader)
print(f"Accuracy of the model excluding 'Y' column on the TRAINED dataset (precentiles 1-100): {accuracy:.2f}%")

parameters.append('Y')
diabetes_datasetY = dataY(0, train_data, parameters, target_cols)
diabetesY_dataloader = DataLoader(diabetesY_dataset, batch_size = 10)
# Training for Class with 'Y'
train_model(modelPrecentsY, diabetesY_dataloader, epochs, optimizer, loss_fnPrecents)
accuracy = evaluate_model(modelPrecentsY, diabetesY_dataloader)
print(f"Accuracy of the model including 'Y' column on the TRAINED dataset (precentiles 1-100): {accuracy:.2f}%")

parameters.remove('Y')
# Training with remaining random 20% of the database
TrainData = data(0, test_data, parameters, target_cols)
diabetes_dataloader = DataLoader(TrainData, batch_size = len(TrainData))
DataTest, TargetsTest = next(iter(diabetes_dataloader))
DataTest = DataTest.to(device)
TargetsTest = TargetsTest.to(device)
accuracy = evaluate_model(modelPrecents, diabetes_dataloader)
print(f"Accuracy of the model excluding 'Y' column on the TESTED dataset (precentiles 1-100): {accuracy:.2f}%") # test data without 'Y'

parameters.append('Y')
TrainYData = dataY(0, test_data, parameters, target_cols)
diabetesY_dataloader = DataLoader(TrainYData, batch_size = len(TrainYData))
DataYTest, TargetsYTest = next(iter(diabetesY_dataloader))
accuracy = evaluate_model(modelPrecentsY, diabetesY_dataloader)
print(f"Accuracy of the model including 'Y' column on the TESTED dataset (precentiles 1-100): {accuracy:.2f}%") # test data with 'Y'
