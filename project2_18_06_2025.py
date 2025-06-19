#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import pandas.testing as pd_testing
#import pandos.util.testing as rm
import seaborn as sns
import matplotlib.pyplot as plt


# In[23]:


df= pd.read_csv("/home/mathematics/Downloads/dataset (2)/dataset//Boston.csv")
df.head()


# In[18]:


df.isnull().sum()


# In[19]:


df.describe()


# In[20]:


sns.histplot(df["INDUS"])


# In[25]:


sns.heatmap(df.corr(),annot=True)


# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


target_y=df["INDUS"]
features_x=df.drop('INDUS',axis=1,inplace=False)
features_x.head()
                   


# In[28]:


train_x,test_x,train_y,test_y=train_test_split(features_x,target_y,test_size=0.2)


# In[29]:


test_x.shape


# In[30]:


test_y.shape


# In[31]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(train_x,train_y)


# In[33]:


print('Co-efficient:',model.coef_)
print('Intercept;',model.intercept_)


# In[34]:


predicted_y=model.predict(test_x)


# In[35]:


predicted_y.shape


# In[37]:


from sklearn.metrics import r2_score
print("accuracy:",r2_score(test_y,predicted_y))


# In[46]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle


# In[47]:


X_train, X_test, y_train, y_test = train_test_split(df[['INDUS']], df['PRICE'], test_size=0.3, random_state=109)


# In[48]:


linearRegr =  LinearRegression()
linearRegr.fit(X_train, y_train)
y_pred = linearRegr.predict(X_test)


# In[49]:


y_pred


# In[50]:


from sklearn.metrics import r2_score
print(r2_score(y_pred, y_test))


# In[51]:


#Saving the Model
pickle_out = open("linearRegr_windows_new.pkl", "wb") 
pickle.dump(linearRegr, pickle_out) 
pickle_out.close()



# In[52]:


pwd


# In[53]:


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


# In[ ]:




