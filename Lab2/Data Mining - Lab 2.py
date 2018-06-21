
# coding: utf-8

# # Lab 2
# 
# <b>Class:</b> MSDS 7331 Data Mining
# <br> <b>Dataset:</b> Belk Endowment Educational Attainment Data 
# 
# <h1 style="font-size:150%;"> Teammates </h1>
# Maryam Shahini
# <br> Murtada Shubbar
# <br> Michael Toolin
# <br> Steven Millett

# In[1]:


#Set global variables
#Number of features we will be selecting for feature selection

N_FEATURES_OPTIONS = [2, 25 , 50]


#Alpha and C we will be using for our classifiers

C_OPTIONS = [1e-2, 1e-1, 1e0, 1e1, 1e2]

YEARS = [2014, 2015, 2016, 2017]

#Import data all necessary libraries we will be using in our estimation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import re
import sklearn
import statistics


from umap.umap_ import UMAP
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2, SelectPercentile, RFE

from sklearn.preprocessing import StandardScaler, Binarizer

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC

from sklearn.metrics import accuracy_score, confusion_matrix
from IPython.display import display, HTML

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, KFold

from sklearn.naive_bayes import MultinomialNB


get_ipython().run_line_magic('matplotlib', 'inline')


# # 1.a Data Preparation
# 10 points - Deﬁne and prepare your class variables. Use proper variable 
# representations (int, ﬂoat, one-hot, etc.). Use pre-processing methods (as needed) for
# dimensionality reduction, scaling, etc. Remove variables that are not needed/useful for 
# the analysis.

# # 1.b Data Preparation
# 5 points - Describe the final dataset that is used for classification/regression (include a description of any newly formed variables you created).

# In[10]:


# The 2017 Public Schools Machine Learning 
# Date Set is being used throughout this 
# analysis.  The _ML suffix is removed to less 
# name space size
# Load Full Public School Data Frames for each year

school_data = pd.DataFrame()

for year in YEARS:
    temp_year = pd.read_csv('../Data/'+str(year)+'/Machine Learning Datasets/PublicSchools'+str(year)+'_ML.csv', low_memory=False)
    hs_temp_year = pd.read_csv('../Data/'+str(year)+'/Machine Learning Datasets/PublicHighSchools'+str(year)+'_ML.csv', low_memory=False)
    cols_to_use = hs_temp_year.columns.difference(temp_year.columns)
    cols_to_use = np.append(cols_to_use,'unit_code')
    print(cols_to_use[0:20])
    temp_year = pd.merge(temp_year, hs_temp_year[cols_to_use],left_index=True, right_index=True, on='unit_code',how='left' )
    temp_year['Year']=year
    
    temp_year = pd.read_csv('../Data/'+str(year)+'/Machine Learning Datasets/PublicSchools'+str(year)+'_ML.csv', low_memory=False)
    school_data = pd.concat([school_data,temp_year],ignore_index=True, sort=True)


# In[42]:

CRITICAL_NA = .75

imputed_school_data = school_data.apply(lambda col: col.fillna(0) if col.count()/col.shape[0]<CRITICAL_NA else col.fillna(col.median()),axis=0)

#imputed_school_data = school_data.interpolate(method='krogh',axis=0)
#imputed_school_data.head(10)


# In[38]:

print(imputed_school_data.filter(regex=('ap')).columns)


# # 2.a Modeling and Evaluation
# 10 points - Choose and explain your evaluation metrics that you will use (i.e., accuracy, precision, recall, F-measure, or any metric we have discussed). Why are the measure(s) appropriate for analyzing the results of your modeling? Give a detailed explanation backing up any assertions.

# # 2.b Modeling and Evaluation
# 10 points - Choose the method you will use for dividing your data into training and why testing splits (i.e., are you using Stratiﬁed 10-fold cross validation? Why?). Explain why your chosen method is appropriate or use more than one method as appropriate.

# In[43]:


#split data into X and y dataframes

SPG_Grade_col = imputed_school_data.filter(regex=('^SPG\WGrade')).columns
imputed_school_data[SPG_Grade_col] = imputed_school_data[SPG_Grade_col].apply(lambda col: col.astype(int), axis=1)
y = imputed_school_data[SPG_Grade_col].apply(lambda row:'A' if row.any()!=1 else row[0]*'A+NG'+row[1]*'B'+row[2]*'C'+row[3]*'D'+row[4]*'F'+row[5]*'I',axis=1)

#Removed SPG Grade and unit code(which is primary key for school data table)
 
X = imputed_school_data[school_data.columns.drop(list(school_data.filter(regex='^SPG\WGrade|^SPG\WScore|unit_code')))]


# In[ ]:


# split X and y into test and train sets. We still want
# to do this for external Cross Validation

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)


# # 2.c Modeling and Evaluation
# 20 points - Create three different classification/regression models (e.g., random forest, KNN, and SVM). Two modeling techniques must be new (but the third could be SVM or logistic regression). Adjust parameters as appropriate to increase generalization performance using your chosen metric.

# In[4]:


# Here we establish a basic 10 k-fold internal
# Cross Validation seperation that will be used
# for training our model.

k_fold = KFold(n_splits=10,shuffle=True)

#This creates the template for the pipeline
# This creates a basic pipeline where we will 
# test for dementionality reduction, scaling,
# and classification.

pipe = Pipeline([('reduce_dim', NMF()),
                  ('scale', StandardScaler()), 
                  ('clf', LogisticRegression())])


# In[ ]:


# #Don't run this unless you want to retrain the data.

# # Here we are establishing the basic testing criteria
# # for our pipeline. This will run through a number of
# # parameters for our pipeline, including type of dimensionality
# # reduction, number of features to reduce, scaling (yes/no), 
# # classification models, and parameters of the classification model.

# param_grid = [
#     {
#         'reduce_dim': [NMF(), PCA(),TSNE()],
#         'reduce_dim__n_components': N_FEATURES_OPTIONS,
#         'scale':[None,StandardScaler()],
#         'clf':[SVC(),LogisticRegression()],
#         'clf__C': C_OPTIONS
#     },
#     {
#         'reduce_dim': [NMF(), PCA(),TSNE()],
#         'reduce_dim__n_components': N_FEATURES_OPTIONS,
#         'scale':[None,StandardScaler()],
#         'clf':[SGDClassifier(tol=1e-3,max_iter=1000)],
#         'clf__alpha': C_OPTIONS
#     }
# ]


# # This will test the parameter dict against our 
# # pipeline

# grid_search = GridSearchCV(pipe, param_grid=param_grid, cv=k_fold,n_jobs=-1, verbose=1 )


# # Here we are training the model, this is 
# # what takes the most amount of time to run
# grid_search.fit(X_train, y_train)


# #This saves the grid_search variable
# # to an external file so we don't have to 
# # keep running the gridsearch

# from sklearn.externals import joblib
# joblib.dump(grid_search, 'savedBestModel.pkl')


# # 2.d Modeling and Evaluation
# 10 points - Analyze the results using your chosen method of evaluation. Use visualizations of the results to bolster the analysis. Explain any visuals and analyze why they are interesting to someone that might use this model.

# In[5]:


#Run this to load the model from the save file

from sklearn.externals import joblib
grid_search = joblib.load('savedBestModel.pkl')


# Loads all parameters run into a dict 

params = np.array(grid_search.cv_results_['params'])


# Loads all mean test scores into an array

mean_scores = np.array(grid_search.cv_results_['mean_test_score'])


# In[6]:


# Assigns all models to an array

classifier_labels=['SVC','LogisticRegression','SGDClassifier']


# Creates an empty dataframe that is to be
# filled with the mean test accuracy by C global
# variable and the different classifiers

classifier_temp = pd.DataFrame(columns=classifier_labels,index=C_OPTIONS,
                               data=np.linspace(.1,.25,num=len(C_OPTIONS)*len(classifier_labels)).reshape(len(C_OPTIONS),len(classifier_labels)))
classifier_temp.fillna(0,inplace=True)

for i, (param, score) in enumerate(zip(params, mean_scores)):
    C = param['clf__C'] if 'clf__C' in param else param['clf__alpha']
    class_state = str(param['clf']).split('(')[0]
    if classifier_temp.at[C,class_state] < score:
        classifier_temp.at[C,class_state] = score


# Printing a grid of the best accuracies
        
display(classifier_temp.transpose())   


# Print a line plot which shows the best 
# accuracies
 
classifier_temp.plot(logx=True,ylim=(0,1),figsize=(14,10),title='Accuracy by Classifier'); 


# In[7]:


# Assigns all reduction models to an array

reduce_labels=['NMF','PCA','SelectKBest']


# Translates the N Features array
# to an array full of string

temp_N_FEATURES_OPTIONS = [str(r) for r in N_FEATURES_OPTIONS]
temp_N_FEATURES_OPTIONS=temp_N_FEATURES_OPTIONS+['None']


# Creates an empty dataframe that is to be
# filled with the mean test accuracy by N Features
# variable and the different feature reduction models

reduce_temp = pd.DataFrame(columns=reduce_labels,index=temp_N_FEATURES_OPTIONS,
                               data=np.linspace(.1,.25,num=len(temp_N_FEATURES_OPTIONS)*len(reduce_labels)).reshape(+len(temp_N_FEATURES_OPTIONS),len(reduce_labels)))


for i, (param, score) in enumerate(zip(params, mean_scores)):
    trigger=0
    reduce_state = str(param['reduce_dim']).split('(')[0]
    if 'reduce_dim__k' in param:
        N_FEAT = str(param['reduce_dim__k'])
        trigger=1
    elif 'reduce_dim__n_components' in param:
        N_FEAT = str(param['reduce_dim__n_components'])
        trigger=1
    else:
        if reduce_temp.at['None','NMF'] < score:
            reduce_temp.at['None','NMF'] = score
            reduce_temp.at['None','SelectKBest'] = score
    if trigger == 1:
        if reduce_temp.at[N_FEAT,reduce_state] < score:
            reduce_temp.at[N_FEAT,reduce_state] = score

            
# Printing a grid of the best accuracies

display(reduce_temp.transpose())


# Print a bar plot which shows the best 
# accuracies

reduce_temp.plot(kind='bar',ylim=(0,1),figsize=(14,10),title='Accuracy by Feature Selection',rot=0);           


# In[ ]:


print('The Index of the best model is',grid_search.best_index_)
print('The parameters of the best model is')
display(grid_search.best_params_)
print('The accuracy of the best model is',round(grid_search.best_score_*100,4))


# # 2.e Modeling and Evaluation
# 10 points - Discuss the advantages of each model for each classification task, if any. If there are not advantages, explain why. Is any model better than another? Is the difference signiﬁcant with 95% conﬁdence? Use proper statistical comparison methods.

# # 2.f Modeling and Evaluation
# 10 points - Which attributes from your analysis are most important? Use proper methods discussed in class to evaluate the importance of different attributes. Discuss the results and hypothesize about why certain attributes are more important than others for a given classiﬁcation task.

# # Deployment
# 5 points - How useful is yolur model for interested parties (i.e., the companies or organizations that might want to use it for prediction)? How would you measure the model's value if it was used by these parties? How would your deploy your model for interested parties? What other data should be collected? How often would the model need to be updated, etc.?

# # Exceptional Work
# 10 points - You have free reign to provide additional modeling. 
# One idea: grid search parameters in a parallelized fashion and visualize the 
# performances across attributes. Which parameters are most signiﬁcant for making a 
# good model for each classiﬁcation algorithm?
