#!/usr/bin/env python
# coding: utf-8

# # Assignment -05 Toyota car problem 
# # Consider only the below columns and prepare a prediction model for predicting Price.

# In[1]:


import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt


# In[19]:


df = pd.read_csv("ToyotaCorolla (1).csv")
df.head()


# In[3]:


df.count()


# In[4]:


df.describe()


# In[5]:


df.isnull().sum()


# In[6]:


df.dtypes


# In[7]:


df.hist()


# In[10]:


corr = df.corr()
fig, ax = plt.subplots(figsize=(8, 8))
#Generate Heat Map, allow annotations and place floats in map
sns.heatmap(corr, cmap='magma', annot=True, fmt=".2f")
#Apply xticks
plt.xticks(range(len(corr.columns)), corr.columns);
#Apply yticks
plt.yticks(range(len(corr.columns)), corr.columns)
#show plot
plt.show()


# In[29]:


df = pd.get_dummies(df)
df.head()


# # Modeling

# In[30]:


X_multi_lreg = df.drop('Price', axis = 1).values
y_multi_lreg = df["Price"].values.reshape(-1,1)


# In[32]:


from sklearn.model_selection import train_test_split
X_train_mlreg, X_test_mlreg, y_train_mlreg, y_test_mlreg = train_test_split(X_multi_lreg,y_multi_lreg, test_size = 0.25, random_state = 4)
print('Train Dataset : ', X_train_mlreg.shape, y_train_mlreg.shape)
print('Test Dataset : ', X_test_mlreg.shape, y_test_mlreg.shape)


# In[33]:


from sklearn.linear_model import LinearRegression
multi_lreg = LinearRegression()
multi_lreg.fit(X_train_mlreg, y_train_mlreg)
print('Intercept : ', multi_lreg.intercept_)
print('Slope : ', multi_lreg.coef_)


# In[34]:


# Use the model to predict the test dataset.
y_mlreg_pred_test = multi_lreg.predict(X_test_mlreg)

# Use the model to predict the train dataset.
y_mlreg_pred_train = multi_lreg.predict(X_train_mlreg)


# In[36]:


print(y_mlreg_pred_test[0:5])
print(y_test_mlreg[0:5])

print(y_mlreg_pred_train[0:5])
print(y_train_mlreg[0:5])


# In[43]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
r2_score_mlreg_train = r2_score(y_mlreg_pred_train, y_train_mlreg)
r2_score_mlreg_test = r2_score(y_mlreg_pred_test, y_test_mlreg)
rmse_mlreg = np.sqrt(mean_squared_error(y_mlreg_pred_test, y_test_mlreg)**2)
print('r2_ score for train dataset for multi linear reg : ', r2_score_mlreg_train)
print('r2_ score for test dataset for multi linear reg : ', r2_score_mlreg_test)
print('root mean squared error for multi linear reg : ', rmse_mlreg)


# In[ ]:





# In[ ]:




