#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[3]:


df=pd.read_csv('winequality-red.csv')
df.head()


# In[4]:


msk=np.random.rand(len(df))<0.8
train=df[msk]
test=df[~msk]


# In[5]:


from sklearn import linear_model
gh=linear_model.LinearRegression()
train_x=np.asanyarray(train[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']])
train_y=np.asanyarray(train[['quality']])
r=gh.fit(train_x,train_y)
print("coefficeints ", gh.coef_)
print("Intercept ",gh.intercept_)
gh.predict([[7.5,0.500,0.36,6.1,0.071,17.0,102.0,0.9978,3.35,0.80,10.5]])


# In[7]:


test.head()


# In[8]:


y_hat= gh.predict(test[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']])
x = np.asanyarray(test[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']])
y = np.asanyarray(test[['quality']])
print("Residual sum of squares: %.2f"
      % np.mean((y_hat - y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % gh.score(x, y))


# In[ ]:




