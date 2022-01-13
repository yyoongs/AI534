#!/usr/bin/env python
# coding: utf-8

# In[2]:

# get_ipython().system('pip3 install numpy')
# get_ipython().system('pip3 install pandas')
# get_ipython().system('pip install matplotlib')
# get_ipython().system('pip install seaborn')

# In[13]:

# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# In[2]:

table = pd.read_csv('pa0(train-only).csv')

print(table)
# In[96]:


table = table.drop('id', axis=1)

print("\n==================================================================================================\n")
print(table)

# In[3]:


table[['month','day','year']] = table['date'].str.split("/",expand=True)

# date.rename(columns = {0:'month',1:'day',2:'year'})
# In[4]:


table = table[['month','day','year','bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15','price']]


# In[119]:


table.columns


# In[5]:

print("\n==================================================================================================\n")
print(table)

# 
# table.groupby('bedrooms')['price']
# In[6]:



table.boxplot(column=['price'], by='bedrooms')
# In[7]:


table.boxplot(column=['price'], by='bathrooms')

# In[8]:

table.boxplot(column=['price'], by='floors')


# get_ipython().system('pip install seaborn')


# if you want to show plot by using seaborn you need to install seaborn and uncommented the code below.
# import seaborn as sns
# sns.set_theme(style="ticks", palette="pastel")
# # In[20]:

# # Draw a nested boxplot to show bills by day and time
# sns.boxplot(x="bathrooms", y="price",
#             data=table)
# sns.despine()

# sns.boxplot(x="bedrooms", y="price",
#             data=table)
# sns.despine()

# sns.boxplot(x="floors", y="price",
#             data=table)
# sns.despine()



covar1 = table[['sqft_living','sqft_living15']]
covar1 = covar1.cov()

covar1.plot.scatter('sqft_living','sqft_living15')
# To show scatter plot about sqrt_living against sqrt_living15

covar2 = table[['sqft_lot','sqft_lot15']]
covar2 = covar2.cov()

covar2.plot.scatter(x="sqft_lot",y="sqft_lot15")
plt.show()
# To show scatter plot about sqft_lot against sqft_lot15




