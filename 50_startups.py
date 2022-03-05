#!/usr/bin/env python
# coding: utf-8

# # Multi-Linear Regression Assignment - 05

# # Prepare a prediction model for profit of 50_startups data.
# Do transformations for getting better predictions of profit and
# make a table containing R^2 value for each prepared model.
# 

# In[2]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv("50_Startups.csv")
df.head()


# In[4]:


df.shape[0]


# In[5]:


df.count()


# In[6]:


df.isnull().sum()


# In[7]:


df.dtypes


# In[8]:


n = len(pd.unique(df['State']))
n


# In[9]:


df.describe()


# In[10]:


df.hist()


# In[41]:


sns.distplot(df['Profit'],bins=5,kde=True)


# In[11]:



sns.set_palette('colorblind')
sns.pairplot(data=df, height=3)

#there is one outlier in profit


# In[12]:


plt.boxplot(df["Profit"]) # outlier in Profit


# In[13]:


plt.boxplot(df["R&D Spend"])


# In[14]:


plt.boxplot(df["Administration"])


# In[15]:


plt.boxplot(df["Marketing Spend"])


# In[16]:


#since the state doesnt affect the data 
#dropping the column
df=df.drop(['State'],axis=1)
df.head()


# # Training and Splitting data

# In[17]:


X = df.iloc[:, :-1].values # dependent parameters
y = df.iloc[:, -1].values # independent parameter


# In[37]:


print (X)
print(y)


# In[19]:


df=df.rename({'R&D Spend':'RD','Marketing Spend':'Marketing_Spend' },axis=1)


# In[20]:


df.corr()


# In[42]:


#gives positive & negative relation between categories
sns.heatmap(df.corr(), annot=True)


# In[22]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#train_data,test_data= train_test_split(df)


# In[24]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[25]:


y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
#the predicted and test values are almost the same 


# In[31]:


from sklearn.metrics import r2_score

rsquar = r2_score(y_test, y_pred)
print(rsquar)
# the accuracy is 93% which we find from r^2 value


# In[39]:


from sklearn.metrics import mean_squared_error
mean_sq = mean_squared_error(y_test,y_pred)
print(mean_sq)


# In[ ]:




