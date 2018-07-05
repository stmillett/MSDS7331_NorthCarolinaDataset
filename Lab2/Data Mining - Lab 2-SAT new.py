
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

# In[133]:


#Set global variables
#Variables for file and school informaiton

YEARS = ['2014', '2015', '2016', '2017']

#Number of features we will be selecting for feature selection

N_FEATURES_OPTIONS = [25,50,100,"all"]

#Alpha and C we will be using for our classifiers

C_ESTIMATORS = [50, 100, 200, 500]
C_DEPTH = [2, 3, 5]
LEARNING_RATE = [1e-2, 1e-1, 1e0]

#Used for KNN gridsearch
C_NEIGHBORS = [4, 5]

#Used for SVC grid search
C_OPTIONS = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]

#Import data all necessary libraries we will be using in our estimation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import re
import sklearn
import statistics
import random


from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2, SelectPercentile, RFE

from sklearn.preprocessing import StandardScaler, Binarizer

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, auc, roc_curve
from IPython.display import display, HTML

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, TimeSeriesSplit, StratifiedShuffleSplit

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor,AdaBoostClassifier,RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

get_ipython().run_line_magic('matplotlib', 'inline')


# %%html
# <style>
# table {float:left}
# </style>

# # 1.a Data Preparation
# 10 points - Deﬁne and prepare your class variables. Use proper variable 
# representations (int, ﬂoat, one-hot, etc.). Use pre-processing methods (as needed) for
# dimensionality reduction, scaling, etc. Remove variables that are not needed/useful for 
# the analysis.
# 
# # The Belk Endowment Educational Attainment Data Repository for North Carolina Public Schools
# Our data set originates from the North Carolina Public Schools Reports and Statistics. This public site contains large amounts of information covering many aspects of the performance of students and schools across the state of North Carolina. It includes public and charter schools ranging from the elementary level to high schools. http://www.ncpublicschools.org/
# 
# The data used in our lab consists of portions of this data which includes the school years 2014-2017. The data used is the result of combining and cleaning the raw data sets available on the North Carolina website. The machine learning data sets are broken down by school year and then sub-setted by elementary school, middle school and high school information. 
# 
# In this lab our data set consists of all the data available for school years 2015-2017 from the Machine Learning data available.  First step is to combine all the data from previous years and add the variable ‘Year’ to each row, keeping track of which year this data was collected.
# 
# Next the each feature is inspected for NA values.  If more than 75% of the feature contains NA, we replace that field with 0.  If less than 75% is NA, then the median value of the column is used to replace the NA.
# 
# ###Need description of what was done in Altyrex.
# 
# Two binary classifications are performed.  The first classification looks at what
# 
# 
# 
# |<p align="">Variable|<p align="">Type|<p align="">Note|
# |--------|----|----|
# |<p align="">Year|<p align="">Object|<p align="">Tracks year data is from|
# |<p align="">local_crime_greater|<p align="">int64|<p align="">1 if crime in school > LEA average crime, 0 otherwise
# |<p align="">X_crime_reduced|<p align="">Data Frame|<p align="">Used in crime reduced scope model, removes racial information from data|

# # 1.b Data Preparation
# 5 points - Describe the final dataset that is used for classification/regression (include a description of any newly formed variables you created).

# In[67]:


# The 2017 Public Schools Machine Learning 
# Date Set is being used throughout this 
# analysis.  The _ML suffix is removed to less 
# name space size
# Load Full Public School Data Frames for each year

school_data = pd.DataFrame()

for year in YEARS:
    #Load public school master file
    temp_year = pd.read_csv('../Data/'+str(year)+'/Machine Learning Datasets/PublicHighSchools'+str(year)+'_ML.csv', low_memory=False)
        
    #Add year column and concatonating all data together
    temp_year['Year']=year
    
    if(school_data.empty):
        school_data = pd.concat([school_data,temp_year],ignore_index=True)
    else:
        school_data = pd.concat([school_data,temp_year], join = "inner",ignore_index=True)

print(school_data.shape)


# In[68]:


#This is the critical threshold
CRITICAL_NA = .75

#With this we check if the column is less than 75% non-NA, if it is greater than 75% non-NA
#We replace the NA with the median of the column, otherwise we replace the value with 0

imputed_school_data = school_data.apply(lambda col: col.fillna(0) if col.count()/col.shape[0]<CRITICAL_NA else col.fillna(col.median()),axis=0)



# In[70]:



#ed = imputed_school_data.copy()

#ed.to_csv('C:\\Users\\Bahr\\Desktop\\SMU_DS_Summer_2018\\Data_Mining\\Assignments\\Lab TWO\\MSDS7331_NorthCarolinaDataset\\Lab2\\RAW.csv')

#ED is processed using Alteryx 


#reading in new dataset. 
SAT_Filtered = pd.DataFrame()
SAT_Filtered = pd.read_csv('../Data/School Datasets/HighSchoolCleanFinal.csv', low_memory=False)

SAT_Filtered.describe()

#SAT_SCORE_ZERO	Boolean: If SAT equals zero. Used it to test if anyone participated but produced zero scores. 
#SAT_Score_above1000	 Boolean: SAT score above 1000 or not to get into a decent college in NC. 
#SAT_participation_number	 Estimate of number of students that took the SAT
#EST_Student_College_NO	 Estimate of the number of students who didn't attempt the SAT
#Student_Num_College_Ready_SAT	 Estimate of the number of students that can apply to college without SAT score hinderance from that school. 


# In[5]:


#SAT_Filtered.head()


# # Get list of highly correlated features

# In[6]:


# corr_matrix = imputed_school_data.corr().abs()

# # Select upper triangle of correlation matrix
# upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# # Find index of feature columns with correlation greater than 0.95
# to_drop = [column for column in upper.columns if any(upper[column] > 0.99)]

# print(to_drop)


# # 2.a Modeling and Evaluation
# Using the right evaluation metric for classification system is crucial. Otherwise, it may results in thinking that the model is performing well but in reality, it doesn’t.
# 
# There are two tasks in this section of “NC Educational Data” project:
# 
# The first task is to predict a binary classification target, either if the average SAT score of each school is good enough to gets the student to the North Carolina Universities or not. The SAT is a standardized test widely used for college admissions in the United States. For this purpose we have a cut off 1200 out of 1600. The second task is to predict if the crime per 100 students at each school level is higher than the LEA level or not. After considering all evaluation metrics for classification systems, we ended up using ROC Curve. Area under ROC Curve (or AUC for short) is a performance metric for binary classification problems.
# 
# In fact, a ROC curve can be used to select a threshold for a classifier which maximizes the true positives, while minimizing the false positives.
# 
# We usually use ROC when both classes detection are important. Here, our models should be able to decrease both false positive rate (which is identifying the schools with enough good average SAT score for getting admission in different universities) and also decreasing the false negative rate (which is detecting schools with not good average SAT scores).
# 
# The same for the second task, it is important to decrease both false positive and false negative rates.
# 
# The AUC represents a model’s ability to discriminate between positive and negative classes. An area of 1.0 represents a model that made all predictions perfectly. An area of 0.5 represents a model as good as random. Most classifiers have AUCs that fall somewhere between these two values. Therefore, the overall model performances can be compared by considering the AUC.

# # 2.b Modeling and Evaluation
# 10 points - Choose the method you will use for dividing your data into training and why testing splits (i.e., are you using Stratiﬁed 10-fold cross validation? Why?). Explain why your chosen method is appropriate or use more than one method as appropriate.

# # Task 1: Crime - Classification Model

# In[71]:


# split X and y into test and train sets. We still want
# to do this for external Cross Validation

crime_imputed_school_data = imputed_school_data

crime_imputed_school_data['local_crime_greater'] = crime_imputed_school_data.apply(lambda each_row: 1 if (each_row['crime_per_c_num']-each_row['lea_crime_per_c_num'])<0 else 0,axis=1)

#split data into X and y dataframes

y_crime = crime_imputed_school_data['local_crime_greater']

#Removed SPG Grade and unit code(which is primary key for school data table)
 
X_crime = imputed_school_data[school_data.columns.drop(list(school_data.filter(regex='crime|lea|LEA|^st\_')))]

X_crime_train, X_crime_test, y_crime_train, y_crime_test = train_test_split(X_crime, y_crime, test_size=.2)


# # Task 2: SAT Score - Classification Model

# In[72]:


# To split X and y into test and train sets.

y_SAT = SAT_Filtered['SAT_Score_above1000']

#Removed SAT_SCore_above1000 and unit code(which is primary key for school data table)
 
X_SAT = SAT_Filtered[SAT_Filtered.columns.drop(list(SAT_Filtered.filter(regex='SAT_Score|SAT_score|SAT_SCORE|sat_avg|unit_code|lea|LEA|^st\_')))]

X_SAT_train, X_SAT_test, y_SAT_train, y_SAT_test = train_test_split(X_SAT, y_SAT, test_size=.2)


# In[73]:


#y_SAT.head()
#X_SAT.shape
#SAT_Filtered.shape
#X_SAT.SAT_Score_above1000


# # 2.c Modeling and Evaluation
# 20 points - Create three different classification/regression models (e.g., random forest, KNN, and SVM). Two modeling techniques must be new (but the third could be SVM or logistic regression). Adjust parameters as appropriate to increase generalization performance using your chosen metric.

# In[81]:


k_fold = KFold(n_splits=10,shuffle=True)

#This creates the template for the pipeline
# This creates a basic pipeline where we will 
# test for dementionality reduction, scaling,
# and classification.


pipe = Pipeline([ ('reduce_dim',SelectKBest(chi2)),
                  ('scale', StandardScaler()), 
                  ('clf', GradientBoostingClassifier())])


# # SAT Model #1 : Random Forest Classifier

# In[116]:


param_grid = [
    {
         'reduce_dim__k': N_FEATURES_OPTIONS,
         'clf': [RandomForestClassifier()],
         'clf__n_estimators': C_ESTIMATORS, 
         'clf__max_depth': C_DEPTH,
     }
]

grid_search = GridSearchCV(pipe, param_grid=param_grid, cv=k_fold,n_jobs=-1, verbose=1 )

SAT_RandomForest_model = grid_search.fit(X_SAT_train, y_SAT_train)

y_SAT_score1 = grid_search.predict(X_SAT_test)

print(roc_auc_score(y_SAT_test, y_SAT_score1))


# In[117]:


# Plot ROC Curve


# In[118]:


fpr, tpr, _ = roc_curve(y_SAT_test, y_SAT_score1 )
roc_auc = auc(fpr, tpr)
lw=1
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# # SAT Model #2 : KNN Classifier

# In[120]:


############################
param_grid = [
    {
         'reduce_dim__k': N_FEATURES_OPTIONS,
         'clf': [KNeighborsClassifier()],
         'clf__n_neighbors': C_NEIGHBORS, 
     }
]

grid_search = GridSearchCV(pipe, param_grid=param_grid,cv=k_fold,n_jobs=-1, verbose=1 )

SAT_KNearest_model = grid_search.fit(X_SAT_train, y_SAT_train)

y_SAT_score2 = grid_search.predict(X_SAT_test)

print(roc_auc_score(y_SAT_test, y_SAT_score2))


# In[121]:


# Plot ROC Curve
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_SAT_test, y_SAT_score2 )
roc_auc = auc(fpr, tpr)
lw=1

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# # SAT Model #3 :  SVC Classifier

# In[122]:


##########################
param_grid = [
    {
         'reduce_dim__k': N_FEATURES_OPTIONS,
         'clf': [SVC()],
         'clf__C': C_OPTIONS, 
     }
]

grid_search = GridSearchCV(pipe, param_grid=param_grid, cv=k_fold,n_jobs=-1, verbose=1 )

SAT_SVC_model = grid_search.fit(X_SAT_train, y_SAT_train)

y_SAT_score3 = grid_search.predict(X_SAT_test)

print(roc_auc_score(y_SAT_test, y_SAT_score3))


# In[123]:


# Plot ROC Curve
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_SAT_test, y_SAT_score3 )
roc_auc = auc(fpr, tpr)
lw=1

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# # Crime Models

# # Crime Model #1: Gradient Boosting

# In[111]:


k_fold = StratifiedShuffleSplit(n_splits=10)


# In[147]:


param_grid = [
   {
        'reduce_dim__k': N_FEATURES_OPTIONS,
        'clf':[GradientBoostingClassifier()],
        'clf__n_estimators': C_ESTIMATORS, 
        'clf__max_depth': C_DEPTH,
        'clf__learning_rate':LEARNING_RATE
    }
]


# # This will test the parameter dict against our 
# # pipeline

grid_search = GridSearchCV(pipe, param_grid=param_grid, cv=k_fold,n_jobs=-1, verbose=1 )


# # Here we are training the model, this is 
# # what takes the most amount of time to run
crime_GradientBoost_model = grid_search.fit(X_crime_train, y_crime_train)


y_crime_GradientBoost_score = grid_search.predict(X_crime_test)

roc_auc_score(y_crime_test, y_crime_GradientBoost_score)


# In[148]:


pipe.set_params(**crime_GradientBoost_model.best_params_)
pipe.fit(X_crime_train, y_crime_train)


# In[149]:


coef = pipe.steps[2][1].feature_importances_

mask = pipe.steps[0][1].get_support()
new_features=[]
feature_names=list(X_crime_train.columns.values)
for bool, feature in zip(mask, feature_names):
    if bool:
        new_features.append(feature)

#Creates a new dataframe with the coefficients and the 
predicted_data = pd.DataFrame(data=coef,index=new_features,columns=['Influence'])
print("The top 10 features that influence SPG Grade are the following")



display(predicted_data.sort_values(by='Influence', ascending=False)[0:10])


# # Crime Model #2: Ada Boost

# In[87]:


param_grid = [
   {
        'reduce_dim__k': N_FEATURES_OPTIONS,
        'clf': [AdaBoostClassifier()],
        'clf__n_estimators': C_ESTIMATORS,
        'clf__learning_rate':LEARNING_RATE
    }
]


# # This will test the parameter dict against our 
# # pipeline

grid_search = GridSearchCV(pipe, param_grid=param_grid, cv=k_fold,n_jobs=-1, verbose=1 )


# # Here we are training the model, this is 
# # what takes the most amount of time to run
crime_ADABoost_model = grid_search.fit(X_crime_train, y_crime_train)

y_crime_ADABoost_score = grid_search.predict(X_crime_test)

print(roc_auc_score(y_crime_test, y_crime_ADABoost_score))


# In[88]:


print(crime_ADABoost_model.best_params_)


# # Crime Model #3: Random Forest Classifier

# In[92]:


param_grid = [
   {
        'reduce_dim__k': N_FEATURES_OPTIONS,
        'clf': [RandomForestClassifier()],
        'clf__n_estimators': C_ESTIMATORS, 
        'clf__max_depth': C_DEPTH,
    }
]


# # This will test the parameter dict against our 
# # pipeline

grid_search = GridSearchCV(pipe, param_grid=param_grid, cv=k_fold,n_jobs=-1, verbose=1 )


# # Here we are training the model, this is 
# # what takes the most amount of time to run
crime_RandomForest_model = grid_search.fit(X_crime_train, y_crime_train)

y_crime_RandomForest_score = grid_search.predict(X_crime_test)

print(roc_auc_score(y_crime_test, y_crime_RandomForest_score))


# In[93]:


print(crime_RandomForest_model.best_params_)
print(crime_RandomForest_model.multimetric_)


# # Crime Model #4: KNN

# In[129]:



param_grid = [
    {
         'reduce_dim__k': N_FEATURES_OPTIONS,
         'clf': [KNeighborsClassifier()],
         'clf__n_neighbors': C_NEIGHBORS, 
     }
]


# # This will test the parameter dict against our 
# # pipeline

grid_search = GridSearchCV(pipe, param_grid=param_grid, cv=k_fold,n_jobs=-1, verbose=1 )


# # Here we are training the model, this is 
# # what takes the most amount of time to run
crime_KNearest_model = grid_search.fit(X_crime_train, y_crime_train)

y_crime_score = grid_search.predict(X_crime_test)

print(roc_auc_score(y_crime_test, y_crime_score))


# In[126]:


print(crime_KNearest_model.best_params_)


# # Crime Model #5: Bagging Method

# In[150]:


param_grid = [
    {
         'reduce_dim__k': N_FEATURES_OPTIONS,
         'clf': [BaggingClassifier(DecisionTreeClassifier())],
         'clf__n_estimators': C_ESTIMATORS 
     }
]


# # This will test the parameter dict against our 
# # pipeline

grid_search = GridSearchCV(pipe, param_grid=param_grid, cv=k_fold,n_jobs=-1, verbose=1 )


# # Here we are training the model, this is 
# # what takes the most amount of time to run
crime_Bagging_model = grid_search.fit(X_crime_train, y_crime_train)

y_crime_Bagging_score = grid_search.predict(X_crime_test)

print(roc_auc_score(y_crime_test, y_crime_Bagging_score))


# In[152]:


pipe.set_params(**crime_Bagging_model.best_params_)
pipe.fit(X_crime_train, y_crime_train)


# In[163]:


clf = pipe.steps[2][1]

coef = np.mean([
    tree.feature_importances_ for tree in clf.estimators_
], axis=0)

mask = pipe.steps[0][1].get_support()
new_features=[]
feature_names=list(X_crime_train.columns.values)
for bool, feature in zip(mask, feature_names):
    if bool:
        new_features.append(feature)

#Creates a new dataframe with the coefficients and the 
predicted_data = pd.DataFrame(data=coef,index=new_features,columns=['Influence'])
print("The top 10 features that influence SPG Grade are the following")



display(predicted_data.sort_values(by='Influence', ascending=False)[0:10])


# # Reduced scope model

# In[179]:


X_crime_reduced = X_crime[X_crime.columns.drop(list(X_crime.filter(regex='[Ww]hite|[Mm]ale|[Pp]acific[Ii]sland|[Aa]sian|[Hh]ispanic|[Rr]ace|[Bb]lack|[Mm]inority|[Tw]wo[Oo]r[Mm]ore|[Ii]ndian|[Ww]hite')))]
X_crime_reduced_train, X_crime_reduced_test, y_crime_reduced_train, y_crime_reduced_test = train_test_split(X_crime_reduced, y_crime, test_size=.2)


# In[173]:


# print(X_crime_reduced.shape)


# In[174]:


#  param_grid = [
#     {
#          'reduce_dim__k': N_FEATURES_OPTIONS,
#          'clf__n_estimators': C_ESTIMATORS, 
#          'clf__max_depth': C_DEPTH,
#      }
# ]


# # # This will test the parameter dict against our 
# # # pipeline

# grid_search = GridSearchCV(pipe, param_grid=param_grid, cv=k_fold,n_jobs=-1, verbose=1 )


# # # Here we are training the model, this is 
# # # what takes the most amount of time to run
# crime_reduced_GradientBoost_model = grid_search.fit(X_crime_reduced_train, y_crime_reduced_train)


# y_crime_reduced_score = grid_search.predict(X_crime_reduced_test)

# print(roc_auc_score(y_crime_reduced_test, y_crime_reduced_score))


# Based on the contributing features to the model we wanted to remove possible politically biased items that could create unfavorable models with regard to making policies to improve crime outcomes for the high schools in the schools specified in these North Carolina models. We wanted to remove any indicators that could possibly indicate the race of and makeup of the student body so as to not disadvantage any group of students with any recommendations for policy changes.

# # Reduced method using Gradient Boost method

# In[185]:


pipe.set_params(**crime_GradientBoost_model.best_params_)
pipe.fit(X_crime_reduced_train, y_crime_reduced_train)

y_crime_reduced_score = pipe.predict(X_crime_reduced_test)

print(roc_auc_score(y_crime_reduced_test, y_crime_reduced_score))


# In[186]:


coef = pipe.steps[2][1].feature_importances_

mask = pipe.steps[0][1].get_support()
new_features=[]
feature_names=list(X_crime_reduced_train.columns.values)
for bool, feature in zip(mask, feature_names):
    if bool:
        new_features.append(feature)

#Creates a new dataframe with the coefficients and the 
predicted_data = pd.DataFrame(data=coef,index=new_features,columns=['Influence'])
print("The top 10 features that influence SPG Grade are the following")



display(predicted_data.sort_values(by='Influence', ascending=False)[0:10])


# # Reduced method using Bagging method

# In[187]:


pipe.set_params(**crime_Bagging_model.best_params_)
pipe.fit(X_crime_reduced_train, y_crime_reduced_train)

y_crime_bagging_reduced_score = pipe.predict(X_crime_reduced_test)

print(roc_auc_score(y_crime_reduced_test, y_crime_bagging_reduced_score))


# In[188]:


clf = pipe.steps[2][1]

coef = np.mean([
    tree.feature_importances_ for tree in clf.estimators_
], axis=0)

mask = pipe.steps[0][1].get_support()
new_features=[]
feature_names=list(X_crime_reduced_train.columns.values)
for bool, feature in zip(mask, feature_names):
    if bool:
        new_features.append(feature)

#Creates a new dataframe with the coefficients and the 
predicted_data = pd.DataFrame(data=coef,index=new_features,columns=['Influence'])
print("The top 10 features that influence SPG Grade are the following")



display(predicted_data.sort_values(by='Influence', ascending=False)[0:10])


# # 2.d Modeling and Evaluation
# 10 points - Analyze the results using your chosen method of evaluation. Use visualizations of the results to bolster the analysis. Explain any visuals and analyze why they are interesting to someone that might use this model.

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
