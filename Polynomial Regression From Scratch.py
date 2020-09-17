#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
df = pd.read_csv('position_salaries.csv')
df.head()


# In[3]:


df = pd.concat([pd.Series(1, index = df.index, name='00'), df], axis = 1)
df.head()


# In[4]:


df = df.drop(columns='Position')


# In[5]:


y = df['Salary']
X = df.drop(columns = 'Salary')
X.head()


# In[6]:


X['Level1'] = X['Level']**2
X['Level2'] = X['Level']**3
X.head()


# In[7]:


m = len(X)
X = X/X.max()


# In[8]:


X


# In[9]:


def hypothesis(X, theta):
    y1 = theta * X
    return np.sum(y1, axis = 1)


# In[10]:


def cost(X, y, theta):
    y1 = hypothesis(X, theta)
    return sum(np.sqrt((y1-y)**2)) / (2 * m)


# In[11]:


def gradientDescent(X, y, theta, alpha, epoch):
    J = []
    k = 0
    while k < epoch:
        y1 = hypothesis(X, theta)
        for c in range(0, len(X.columns)):
            theta[c] = theta[c] - alpha*sum((y1-y)* X.iloc[:, c])/m
        j = cost(X, y, theta)
        J.append(j)
        k += 1
    return J, theta


# In[12]:


theta = np.array([0.0]*len(X.columns))
J, theta = gradientDescent(X, y, theta, 0.05, 700)


# In[13]:


y_hat = hypothesis(X, theta)


# In[14]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(x=X['Level'],y= y)           
plt.scatter(x=X['Level'], y=y_hat)
plt.show()


# In[15]:


plt.figure()
plt.scatter(x=list(range(0, 700)), y=J)
plt.show()


# In[ ]:




