#!/usr/bin/env python
# coding: utf-8

# # XAI - Homework 1 - Barbara Bugała

# ## Task A

# Calculate Shapley values for player A given the following value function

# $v(\empty) = 0 \\
# v(A) = 20 \\
# v(B) = 20 \\
# v(C) = 60 \\
# v(A,B) = 60 \\
# v(A,C) = 70 \\
# v(B,C) = 70 \\
# v(A,B,C) = 100 $

# contributions of player A:
# 
# $v(A) - v(\empty) = 20$ \
# $v(A,B) - v(B) = 40$ \
# $v(A,C) - v(C) = 10$ \
# $v(A,B,C) - v(B,C) = 30$

# number of players $ \rightarrow |P| = 3$

# shap values for player A:
# 
# $\phi_A = \frac{2}{6} *20 + \frac{1}{6}* 40 + \frac{1}{6} *10 + \frac{2}{6}* 30 = 25$

# ## Task B

# In[223]:


# imports
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import shap
import itertools
import numpy as np


# In[230]:


# read data
data = pd.read_csv("phoneme.csv")
X = data.drop(["Unnamed: 0", "TARGET"], axis = 1).values
y = data["TARGET"].values

# split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# datasets and dataloaders
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train,dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
training_data = DataLoader(train_dataset, batch_size= 64, shuffle=True)
testing_data = DataLoader(test_dataset, batch_size=64, shuffle=True)


# ### XGBoost

# In[244]:


model_xgb = xgb.XGBRegressor(objective = "reg:squarederror")
model_xgb.fit(X_train, y_train)

y_pred = model_xgb.predict(X_test)


# In[245]:


# generate random numbers that will be used for shap analysis
random1 = random.randint(0, len(test_dataset) - 1)
random2 = random.randint(0, len(test_dataset) - 1)


# In[246]:


# calculation of f(\empty), taking a mean of all data samples
baseline = X_test.mean(axis = 0).reshape(1, 5)
f_baseline = model_xgb.predict(baseline)[0]
print(f_baseline)


# I will do Monte Carlo sampling over permutations of features.
# 
# For each permutation:
# Start from a baseline input.
# Add features one by one in that order.
# Record how much each feature changes the model’s output.
# Average these marginal changes across permutations.

# In[247]:


psi = np.zeros(5)

for perm in itertools.permutations([0, 1, 2, 3, 4]):
    x = baseline.copy()
    prev_pred = model_xgb.predict(x)[0]
    for i in perm:
        x[0][i] = X_test[random1][i]
        pred = model_xgb.predict(x)[0]
        psi[i] += pred - prev_pred
        prev_pred = pred


psi /= 120
print("Shap for attibutions: ", psi)
print("predicted models output: ", psi.sum() + f_baseline)
print("actual model output: ", model_xgb.predict(X_test[random1].reshape(1, -1))[0])


# Shap values from shap library

# In[248]:


explainer = shap.Explainer(model_xgb, baseline)
shap_values = explainer(X_test)


# In[249]:


shap.plots.waterfall(shap_values[random1])


# *The results are not identical, but they are quite similar. The overall impact on the final prediction of each feature seems to be preserved.*

# Second example, calculation from self-made shap 

# In[250]:


psi = np.zeros(5)

for perm in itertools.permutations([0, 1, 2, 3, 4]):
    x = baseline.copy()
    prev_pred = model_xgb.predict(x)[0]
    for i in perm:
        x[0][i] = X_test[random2][i]
        pred = model_xgb.predict(x)[0]
        psi[i] += pred - prev_pred
        prev_pred = pred


psi /= 120
print("Shap for attibutions: ", psi)
print("predicted models output: ", psi.sum() + f_baseline)
print("actual model output: ", model_xgb.predict(X_test[random2].reshape(1, -1))[0])


# In[251]:


shap.plots.waterfall(shap_values[random2])


# *In this example results also look very similar to the ones in the self-made shap calculation. There are small differences but it looks very good.*

# ### Neural net

# In[4]:


class Net(nn.Module):
    def __init__(self, in_params, out_params, hidden):
        super().__init__()
        self.fc1 = nn.Linear(in_params, hidden)
        self.fc2 = nn.Linear(hidden, out_params)
        self.tanh = nn.Tanh()
        self.bn1 = nn.BatchNorm1d(hidden)

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.tanh(x)

        return x


# In[5]:


model = Net(6, 1, 10)


# In[6]:


optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
criterion = torch.nn.MSELoss()

epochs = 5
losses = []


# In[7]:


model.train()
for epoch in range(epochs):
    for x, y in training_data:
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(y, output.squeeze(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().item())


# In[8]:


plt.plot(losses)
plt.show()


# In[9]:


model.eval()
correct = 0
all = 0
with torch.no_grad():
    for x, y in testing_data:
        optimizer.zero_grad()
        output = torch.sign(model(x)).view(-1)
        correct += (output == y).sum().item()
        all += y.size(0)

    print(correct / all  * 100)


# The accuraccy looks very good, but mostly because the dataset is very imbalanced with imbalance ratio over 40.

# In[51]:


random1 = random.randint(0, len(test_dataset) - 1)
random2 = random.randint(0, len(test_dataset) - 1)

random_sample1_torch = test_dataset[random1][0]
random_sample2_torch = test_dataset[random2][0]

random_sample1_numpy = X_test[random1]
random_sample2_numpy = X_test[random2]


# In[11]:


output_sample1 = model(random_sample1_torch.unsqueeze(0))
output_sample2 = model(random_sample2_torch.unsqueeze(0))


# In[12]:


print(random_sample1_torch)


# In[253]:


# model.eval()

# explainer = shap.GradientExplainer(model, test_dataset[:100][0])
# shap_values = explainer(test_dataset[:100][0])
# shap.plots.waterfall(shap_values[0])


# ## Task C

# In[212]:


# read data
data_more_imbalanced = pd.read_csv("mammography.csv")
X_mI = data_more_imbalanced.drop(["Unnamed: 0", "TARGET"], axis = 1).values
y_mI = data_more_imbalanced["TARGET"].values


# In[ ]:


model_xgb = xgb.XGBRegressor(objective = "reg:squarederror")
model_xgb.fit(X_mI, y_mI)

explainer = shap.Explainer(model_xgb)
shap_values = explainer(X_mI)
shap.plots.waterfall(shap_values[0])


# *The difference is visible, shap values are significantly smaller, with the maximum one being usually smaller than 0.03 in absolute value. In contrast, the values in last analysis were up to 0.5 in absolute value.*

# In[ ]:




