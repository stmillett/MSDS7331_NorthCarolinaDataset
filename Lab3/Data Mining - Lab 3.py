
# coding: utf-8

# In[7]:


import matplotlib as plt
#from mpl_toolkits.basemap import Basemap
import pandas as pd
import numpy as np


# In[50]:


gun_data = pd.read_csv('./Data/gun-violence-data_01-2013_03-2018.csv')


# In[51]:


gun_data.head()


# In[52]:


list_of_keywords=['url','source','notes','address','incident_id']

gun_data = gun_data[gun_data.columns.drop(list(gun_data.filter(regex='|'.join(list_of_keywords))))]
gun_data.head()


# In[53]:


len(gun_data.incident_characteristics)


# In[28]:


gun_data['incident_characteristics'].isnull().sum()


# In[30]:





# In[65]:


gun_data_modified = gun_data.copy()
gun_data_modified.incident_characteristics.dropna(inplace=True)
gun_data_modified.incident_characteristics = gun_data_modified.incident_characteristics.apply(lambda row: row.split('||'))
# for ind,row in enumerate(gun_data_modified.incident_characteristics):
#     if isinstance(row,str):
#         gun_data_modified.loc['incident_characteristics',ind] = row.split('||')
#         #print(gun_data_modified.incident_characteristics.iloc[ind])

incidents = gun_data_modified['incident_characteristics'].tolist()
print(incidents)

