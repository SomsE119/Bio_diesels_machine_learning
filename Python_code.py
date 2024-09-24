#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd               #importing libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[34]:


df= pd.read_csv("ictac.csv")      # defining data set


# In[35]:


df.head()                         # exploratory data analysis


# In[36]:


df.tail()


# In[37]:


df.shape


# In[38]:


df.info()


# In[39]:


df.describe


# In[40]:


print(df.columns)

df["FUELS"].unique()


# In[41]:


df["FUELS"].value_counts()


# In[42]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["FUELS"]=le.fit_transform(df["FUELS"])


# In[43]:


df["FUELS"]


# In[44]:


x=df.drop(['BTE (%)'],axis=1)   # independent variables


# In[45]:


y=df["BTE (%)"]                 # dependent variable


# In[46]:


x


# In[47]:


y


# In[48]:


plt.figure(figsize=(5.5,4.5))                             # density plot
sns.kdeplot(y, fill=True, color='blue')          
plt.title(f'Density Plot ')
plt.show()


# In[49]:


g = sns.pairplot(df,x_vars=['HC', 'OPACITY'],y_vars='BTE (%)',height=4,aspect=1,kind='scatter')    # scatter plot
plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=0.5)
plt.show()


# In[50]:


# MACHINE LEARNONG MODEL REGRESSORS 

# LINEAR REGRESSOR

from sklearn.linear_model import LinearRegression                    
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(x, y, test_size= 0.2,random_state=30)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))


print('Mean Square Error:', mse)
print('Mean Absolute Error:', mae)
print('Root Mean Square Error:', rmse)


# In[51]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)


model = LinearRegression()


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print('Mean Squared Error (MSE):', mse)
print('Mean Absolute Error (MAE):', mae)
print('Root Mean Squared Error (RMSE):', rmse)


results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})


print(results.head())  
plt.figure(figsize=(5.5, 4.5))
plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2)
plt.title('Actual vs. Predicted Values (Multiple Linear Regression)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.grid(True)
plt.show()


# In[52]:


print('Intercept: ',model.intercept_)
list(zip(x, model.coef_))


# In[53]:


# RANDOM FOREST REGRESSOR 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)


rf = RandomForestRegressor(n_estimators=100, random_state=30)


rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

random_mse = mean_squared_error(y_test, y_pred)
random_mae = mean_absolute_error(y_test, y_pred)
random_rmse = np.sqrt(random_mse)

print(f"Mean Squared Error (MSE): {random_mse}")
print(f"Mean Absolute Error (MAE): {random_mae}")
print(f"Root Mean Squared Error (RMSE): {random_rmse}")


results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})


print(results.head())  
plt.figure(figsize=(5.5, 4.5))
plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2)
plt.title('Actual vs. Predicted Values (Random Forest Regressor)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.grid(True)
plt.show()


# In[54]:


# SUPPORT VECTOR REGRESSOR

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X_train,X_test,y_train,y_test = train_test_split(x, y, test_size= 0.2,random_state=30)


svr_pipe = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=1.0, epsilon=0.1))
svr_pipe.fit(X_train, y_train)
y_pred = svr_pipe.predict(X_test)


svr_mse = mean_squared_error(y_test, y_pred)
svr_mae = mean_absolute_error(y_test, y_pred)
svr_rmse = np.sqrt(mse)


print(f"Mean Squared Error: {svr_mse}")
print(f"Mean Absolute Error (MAE): {svr_mae}")
print(f"Root Mean Squared Error (RMSE): {svr_rmse}")


# In[55]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)


svr_pipe = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=1.0, epsilon=0.1))

svr_pipe.fit(X_train, y_train)

y_pred = svr_pipe.predict(X_test)


svr_mse = mean_squared_error(y_test, y_pred)
svr_mae = mean_absolute_error(y_test, y_pred)
svr_rmse = np.sqrt(svr_mse)

print(f"Mean Squared Error (MSE): {svr_mse}")
print(f"Mean Absolute Error (MAE): {svr_mae}")
print(f"Root Mean Squared Error (RMSE): {svr_rmse}")

results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})


print(results.head())  
plt.figure(figsize=(5.5, 4.5))
plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2)
plt.title('Actual vs. Predicted Values (Support Vector Regressor)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.grid(True)
plt.show()


# In[56]:


#DECISION TREE REGRESSOR 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)

model = DecisionTreeRegressor()


model.fit(X_train, y_train)


predictions = model.predict(X_test)

tree_mse = mean_squared_error(y_test, predictions)
tree_mae = mean_absolute_error(y_test, predictions)
tree_rmse = np.sqrt(tree_mse)

print(f"Decision Tree Regressor Mean Squared Error (MSE): {tree_mse}")
print(f"Decision Tree Regressor Mean Absolute Error (MAE): {tree_mae}")
print(f"Decision Tree Regressor Root Mean Squared Error (RMSE): {tree_rmse}")


results = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})

print(results.head())  
plt.figure(figsize=(5.5, 4.5))
plt.scatter(y_test, predictions, color='blue', edgecolor='k', alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2)
plt.title('Actual vs. Predicted Values (Decision Tree Regressor)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.grid(True)
plt.show()


# In[57]:


#KNN REGRESSOR

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)

model = KNeighborsRegressor()

model.fit(X_train, y_train)

predictions = model.predict(X_test)


knn_mse = mean_squared_error(y_test, predictions)
knn_mae = mean_absolute_error(y_test, predictions)
knn_rmse = np.sqrt(knn_mse)

print(f"k-NN Regressor Mean Squared Error (MSE): {knn_mse}")
print(f"k-NN Regressor Mean Absolute Error (MAE): {knn_mae}")
print(f"k-NN Regressor Root Mean Squared Error (RMSE): {knn_rmse}")

results = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})


print(results.head())  
plt.figure(figsize=(5.5, 4.5))
plt.scatter(y_test, predictions, color='blue', edgecolor='k', alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2)
plt.title('Actual vs. Predicted Values (k-NN Regressor)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.grid(True)
plt.show()


# In[58]:


#GRADIENT BOOSTING REGRESSOR 


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)


model = GradientBoostingRegressor()


model.fit(X_train, y_train)


predictions = model.predict(X_test)


gboost_mse = mean_squared_error(y_test, predictions)
gboost_mae = mean_absolute_error(y_test, predictions)
gboost_rmse = np.sqrt(gboost_mse)

print(f"Gradient Boosting Regressor Mean Squared Error (MSE): {gboost_mse}")
print(f"Gradient Boosting Regressor Mean Absolute Error (MAE): {gboost_mae}")
print(f"Gradient Boosting Regressor Root Mean Squared Error (RMSE): {gboost_rmse}")

results = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})

print(results.head())  
plt.figure(figsize=(5.5, 4.5))
plt.scatter(y_test, predictions, color='blue', edgecolor='k', alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2)
plt.title('Actual vs. Predicted Values (Gradient Boosting Regressor)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.grid(True)
plt.show()


# In[59]:


#XGBOOST REGRESSOR


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)


model = XGBRegressor()


model.fit(X_train, y_train)

predictions = model.predict(X_test)


xgb_mse = mean_squared_error(y_test, predictions)
xgb_mae = mean_absolute_error(y_test, predictions)
xgb_rmse = np.sqrt(xgb_mse)

print(f"XGBoost Regressor Mean Squared Error (MSE): {xgb_mse}")
print(f"XGBoost Regressor Mean Absolute Error (MAE): {xgb_mae}")
print(f"XGBoost Regressor Root Mean Squared Error (RMSE): {xgb_rmse}")


results = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})


print(results.head())  

plt.figure(figsize=(5.5, 4.5))
plt.scatter(y_test, predictions, color='blue', edgecolor='k', alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2)
plt.title('Actual vs. Predicted Values (XGBoost Regressor)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.grid(True)
plt.show()


# In[29]:


import tensorflow as tf               #ANN

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))  

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=500, batch_size=32, verbose=1, validation_split=0.2)


predictions = model.predict(X_test)

# Evaluate the model
neural_mse = mean_squared_error(y_test, predictions)
neural_mae = mean_absolute_error(y_test, predictions)
neural_rmse = np.sqrt(neural_mse)

print(f"Neural Network Regressor MSE: {neural_mse}")
print(f"Neural Network Regressor MAE: {neural_mae}")
print(f"Neural Network Regressor RMSE: {neural_rmse}")


results = pd.DataFrame({'Actual': y_test, 'Predicted': predictions.flatten()})


print(results.head())  

plt.figure(figsize=(5, 3))
plt.scatter(y_test, predictions, color='blue', edgecolor='k', alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2)
plt.title('Actual vs. Predicted Values (Neural Network)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.grid(True)
plt.show()



# In[30]:


import matplotlib.pyplot as plt    # RESULT COMPARISON
import numpy as np


regressors = ['MLR', 'Neural_networks', 'RFR', 'SVR', 'DT', 'KNN', 'GB', 'XGB']
mae_values = [mae, neural_mae, random_mae, svr_mae, tree_mae, knn_mae, gboost_mae, xgb_mae]  # Replace with your actual MAE values
mse_values = [mse, neural_mse, random_mse, svr_mse, tree_mse, knn_mse, gboost_mse, xgb_mse]  # Replace with your actual MSE values
rmse_values = [rmse, neural_rmse,random_rmse, svr_rmse, tree_rmse, knn_rmse, gboost_rmse, xgb_rmse]  # Replace with your actual RMSE values


bar_width = 0.2
x = np.arange(len(regressors))


plt.figure(figsize=(12,6))
bars1 = plt.bar(x - bar_width, mae_values, width=bar_width, label='MAE', color='skyblue')
bars2 = plt.bar(x, mse_values, width=bar_width, label='MSE', color='lightgreen')
bars3 = plt.bar(x + bar_width, rmse_values, width=bar_width, label='RMSE', color='lightcoral')


plt.xlabel('Regressor')
plt.ylabel('Error Values')
plt.title('Comparison of MAE, MSE, and RMSE for Different Regressors')
plt.xticks(x, regressors)
plt.legend()


for bar in bars1:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')

for bar in bars2:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')

for bar in bars3:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')


plt.show()


# In[60]:


import matplotlib.pyplot as plt   #ERROR STACKED BAR PLOT
import numpy as np


x = np.arange(len(regressors))
bar_width = 0.35


plt.figure(figsize=(12, 6))
bars1 = plt.bar(x, mae_values, width=bar_width, label='MAE', color='skyblue')
bars2 = plt.bar(x, mse_values, width=bar_width, label='MSE', color='lightgreen', bottom=mae_values)
bars3 = plt.bar(x, rmse_values, width=bar_width, label='RMSE', color='lightcoral', bottom=np.array(mae_values) + np.array(mse_values))


plt.xlabel('Regressor')
plt.ylabel('Error Values')
plt.title('Stacked Bar Plot: MAE, MSE, and RMSE for Different Regressors')
plt.xticks(x, regressors)
plt.legend()


for i in range(len(regressors)):
    plt.text(x[i], mae_values[i] / 2, round(mae_values[i], 4), ha='center', va='center', color='black')
    plt.text(x[i], mae_values[i] + mse_values[i] / 2, round(mse_values[i], 4), ha='center', va='center', color='black')
    plt.text(x[i], mae_values[i] + mse_values[i] + rmse_values[i] / 2, round(rmse_values[i], 4), ha='center', va='center', color='black')


    total = mae_values[i] + mse_values[i] + rmse_values[i]
    

    plt.text(x[i], total + 0.02 * total, round(total, 4), ha='center', va='bottom', fontweight='bold', color='black')   
    
    
 
plt.show()


# In[ ]:





# In[61]:


plt.figure(figsize=(4.5, 3.5))


data = [mae_values, mse_values, rmse_values]
plt.boxplot(data, labels=['MAE', 'MSE', 'RMSE'], patch_artist=False, notch=False)


plt.xlabel('Error Metrics')
plt.ylabel('Values')
plt.title('Box Plot: Distribution of MAE, MSE, and RMSE')


plt.show()

