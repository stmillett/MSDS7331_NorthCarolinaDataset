
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import re
#from mpl_toolkits.basemap import Basemap
import pandas as pd
import numpy as np

from collections import Counter

from sklearn.cluster import KMeans, DBSCAN

from sklearn.manifold import TSNE
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


# # Business Understanding (10 points total).
# Describe the purpose of the data set you selected (i.e., why was this data collected in the first place?). How will you measure the effectiveness of a good algorithm? Why does your chosen validation method make sense for this specific dataset and the stakeholders needs? 

# We selected this dataset because we wanted to analyze the sales stats of an online retailer.

# # Data Understanding (20 points total)
# Describe the meaning and type of data (scale, values, etc.) for each
# attribute in the data file. Verify data quality: Are there missing values? Duplicate data?
# Outliers? Are those mistakes? How do you deal with these problems? 
# 
# ### Data Exploration:
# There are 541,909 records in our online marketing data research. This is a transnational data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail. The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers.
# 
# Following are the factor names and descriptions:
# 
# * InvoiceNo: Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation.
# 
# * StockCode: Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product.
# 
# * Description: Product (item) name. Nominal. 
# 
# * Quantity: The quantities of each product (item) per transaction. Numeric.	
# 
# * InvoiceDate: Invice Date and time. Numeric, the day and time when each transaction was generated. 
# 
# * UnitPrice: Unit price. Numeric, Product price per unit in sterling. 
# 
# * CustomerID: Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer. 
# 
# * Country: Country name. Nominal, the name of the country where each customer resides. 
# 

# In[2]:


marketing_data = pd.read_excel('./Data/Online Retail.xlsx')


# In[3]:


marketing_data.head()


# In[4]:


# Data types:
marketing_data.info()


# In[5]:


plt.figure(figsize=(16,10))
plt.xlim(0,1)
plt.suptitle('Percentage of null values per column')
marketing_data.isnull().mean().plot.barh();
plt.show()


# In[7]:


# Find the number of missing values in each column:
print(marketing_data.isnull().sum(axis=0))

na_cols = marketing_data.loc[:,marketing_data.isnull().mean()>.0].columns

for col in na_cols:
    print(col, ' column is ',round(marketing_data[col].isnull().mean(),5)*100,'% null')


# In[7]:


# marketing_data.dropna(subset=['Description'], inplace=True)


# In[8]:


# pred_X = marketing_data[marketing_data['CustomerID'].isna()]
# pred_Y = pred_X['CustomerID']
# pred_X = pred_X.drop('CustomerID',axis='columns')

# X = marketing_data[~marketing_data['CustomerID'].isna()]
# y = X['CustomerID']
# X = X.drop('CustomerID',axis='columns')


# In[9]:


# kfold = KFold(n_splits=10)


# In[10]:


# clf = RandomForestClassifier(n_estimators=10)
# cross_val_score(clf, X, y, cv=kfold)


# By looking at the number of missing values in each attribute, there are 135,080 records that are not belong to any customers, therefore we can remove them from our dataset for furthur analysis..

# In[8]:


#remove the rows with missing in CustomerID column:
marketing_data.dropna(subset=['Description'], inplace=True)
marketing_data['CustomerID'].fillna(99999, inplace=True)


# In[9]:


# Now we have a dataset with no missings
marketing_data.isnull().sum(axis=0)


# In[10]:


# Dimention of the dataset after removing missings:
marketing_data.shape


# By removing missing CustomerID records, we will have a dataset with 406,829 rows.

# In[11]:


# marketing_data[['Quantity', 'UnitPrice']].groupby(['Country']).agg(['mean', 'count'])
per_country=marketing_data.groupby(['Country'])
per_country[['InvoiceNo','StockCode','Description','Quantity','InvoiceDate','UnitPrice','CustomerID']].nunique()


# In[15]:


# for country in marketing_data['Country'].unique():
#     if len(marketing_data[marketing_data['Country']==country]['CustomerID'].unique()) == 1 & ~np.any(np.isnan(marketing_data[marketing_data['Country']==country]['CustomerID'].unique())):
#         print(marketing_data[marketing_data['Country']==country][np.isnan(marketing_data[marketing_data['Country']==country]['CustomerID'])])
#         print(marketing_data[marketing_data['Country']==country]['CustomerID'].unique()[0])


# In[12]:


for ind,el in enumerate(marketing_data['Description']):
    if type(el) != str:
        marketing_data.loc['Description',ind]='Unknown'
    elif re.match('.*[a-zA-Z]+.*',el) is not None:
        pass
    else:
        marketing_data.loc['Description',ind]='Unknown'


# ### Grouping values by transactions by Invoice Number

# In[13]:


marketing_data['TotalPrice'] = marketing_data['Quantity']*marketing_data['UnitPrice']


# In[16]:


marketing_data_grouping = marketing_data.groupby(['InvoiceNo','InvoiceDate','Country','CustomerID'], as_index=False)['UnitPrice','Description','Quantity','TotalPrice'].agg(lambda x: list(x))


##This is to remove any decriptions that may not be strings   
for ind,el in enumerate(marketing_data_grouping['Description']):
    if type(el[0]) != str:
        marketing_data_grouping.drop(ind,inplace=True)

#This will sum the total price for each invoice
marketing_data_grouping['TotalPrice'] = marketing_data_grouping['TotalPrice'].apply(sum)


# ### Classify Transaction type

# In[25]:


marketing_data_grouping['Transaction']=''

for index,row in marketing_data_grouping.iterrows():
     if str(row['InvoiceNo']).startswith("C"):
         marketing_data_grouping.loc[index,'Transaction'] = 'Cancel'
     elif str(row['InvoiceNo']).startswith("A"):
         marketing_data_grouping.loc[index,'Transaction'] = 'Adjust'
     else:
         marketing_data_grouping.loc[index,'Transaction'] = 'Purchase'


# In[26]:


marketing_data_grouping.sample(10)


# Visualize the any important attributes appropriately. Important: Provide an
# interpretation for any charts or graphs.

# # Modeling and Evaluation (50 points total)
# Different tasks will require different evaluation methods. Be as thorough as possible when analyzing
# the data you have chosen and use visualizations of the results to explain the performance and
# expected outcomes whenever possible. Guide the reader through your analysis with plenty of
# discussion of the results.

# ### Option A: Cluster Analysis
# • Perform cluster analysis using several clustering methods
# • How did you determine a suitable number of clusters for each method?
# • Use internal and/or external validation measures to describe and compare the
# clusterings and the clusters (some visual methods would be good).
# • Describe your results. What findings are the most interesting and why? 

# In[6]:


classify = KMeans(init = 'k-means++')


# ### Option B: Association Rule Mining
# • Create frequent itemsets and association rules.
# • Use tables/visualization to discuss the found results.
# • Use several measure for evaluating how interesting different rules are.
# • Describe your results. What findings are the most compelling and why? 

# ### Create lists of different descriptions

# In[30]:


#from apyori import apriori
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# In[37]:


mlb = MultiLabelBinarizer().fit_transform(marketing_data['Description'])
# list_purchase_marketing_data_grouping_descriptions = []
# for el in marketing_data_grouping.loc[marketing_data_grouping['Transaction']=='Purchase','Description']:
#     if len(el)>1:
#         list_purchase_marketing_data_grouping_descriptions.append(el)
    
# print(len(list_purchase_marketing_data_grouping_descriptions))


# In[35]:


purchase_rules = apriori(list_purchase_marketing_data_grouping_descriptions, min_support=0.003,min_confidence=.2,min_lift=0.3,min_length=2 )

purchase_results = list(purchase_rules)


# In[33]:


list_cancel_marketing_data_grouping_descriptions = []
for el in marketing_data_grouping.loc[marketing_data_grouping['Transaction']=='Cancel','Description']:
    if len(el)>1:
        list_cancel_marketing_data_grouping_descriptions.append(el)
    
print(len(list_cancel_marketing_data_grouping_descriptions))


# In[34]:


cancel_rules = apriori(list_cancel_marketing_data_grouping_descriptions, min_support=0.003,min_confidence=.2,min_lift=0.3,min_length=2 )

cancel_results = list(cancel_rules)


# # Deployment (10 points total)
# • Be critical of your performance and tell the reader how you current model might be usable by
# other parties. Did you achieve your goals? If not, can you reign in the utility of your modeling?
# • How useful is your model for interested parties (i.e., the companies or organizations
# that might want to use it)?
# • How would your deploy your model for interested parties?
# • What other data should be collected?
# • How often would the model need to be updated, etc.? 

# # Exceptional Work (10 points total)
# • You have free reign to provide additional analyses or combine analyses 
