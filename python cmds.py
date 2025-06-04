#!/usr/bin/env python
# coding: utf-8

# In[69]:


import numpy as np


# In[2]:


# if numpy is not istalled,then try initial using pip3 install numpy


# In[70]:


# create a list integer  upto 9
I=[0,1,2,3,4,6,6,7,8,9]
                    


# In[4]:


type(I)


# In[5]:


#calculate memory consumption 
import sys
print ("Memory consubed by a single element",sys.getsizeof(I[0]))


# In[6]:


print ("total memory consumption:",len(I)*sys.getsizeof(I[1]))


# In[7]:


#compute the memory consumption by a numpy array
arr=np.array(I)
arr


# In[8]:


type(arr)


# In[9]:


print("Memory consumption by single element in array:",arr.itemsize)


# In[10]:


print("total consumption in numpy arr:",arr.itemsize*arr.size)


# In[11]:


a=[[1,2,3],[4,5,6],[3,4,5]]
type(a)


# In[12]:


arr1=np.array(a)
arr1


# In[13]:


arr1.ndim


# In[14]:


#installation
# sudo apt-get install python3-pandas


# In[15]:


import pandas as pd


# In[16]:


df=pd.read_csv("/home/mathematics/Downloads/dataset (2)/dataset//Iris.csv")


# In[17]:


df.head()


# In[18]:


df["Species"].unique()


# In[19]:


df["Species"].unique()


# In[20]:


type(df)


# In[21]:


df.describe()


# In[22]:


df=pd.read_csv("/home/mathematics/Downloads/dataset (2)/dataset//tips.csv")


# In[23]:


df.head()


# In[86]:


df["day"].unique()


# In[87]:


df.head(4)


# In[88]:


df["tip"].unique()


# In[ ]:





# In[32]:


df.describe()


# In[90]:


df.tail()


# In[91]:


df.tail(10)


# In[25]:


df.head()


# In[35]:


df["time"].unique()


# In[34]:


#visualation
import seaborn as sns


# In[58]:


sns.jointplot(df["total_bill"])


# In[41]:


sns.distplot(df["tip"])


# In[68]:


#bivariate plot
sns.jointplot(x="total_bill",y="tip",data=df,kind="scatter",hue="sex")


# In[73]:


c=df[['total_bill','tip','size']]
sns.heatmap(c.corr(),annot=True)


# In[74]:


sns.pairplot(df,hue="day")


# In[76]:


sns.boxplot(df)


# In[78]:


sns.boxplot(df["total_bill"])


# In[85]:


sns.boxplot(df[["tip","size","total_bill"]])


# In[82]:


sns.boxplot(df[["tip","size"]])


# In[ ]:




