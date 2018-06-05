
# coding: utf-8

# # Lab 1
# 
# <b>Class:</b> MSDS 7331 Data Mining
# <br> <b>Dataset:</b> Belk Endowment Educational Attainment Data 
# 
# <h1 style="font-size:150%;"> Teammates </h1>
# Maryam Shahini
# <br> Murtada Shubbar
# <br> Michael Toolin
# <br> Steven Millett

# In[68]:


import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

# In[69]:


#
# The 2017 Public Schools Machine Learning Date Set is being used throughout this analysis.  The _ML suffix is removed to less name space size
#
# Load Full Public School Data Frames for each year

school_data = pd.read_csv('./Data/2017/machine Learning Datasets/PublicSchools2017_ML.csv', low_memory=False)


# # Business Understanding 
# 
# The North Carolina General Assembly passed legislation in 2014-2014 requiring the assignment of School Performance Grades (SPG) for public and charter Schools [1].  This data set is collected in response to this legislation.  A school's SPG is calculated using 80% of the school’s achievement score and 20% of the school’s growth score.  The achievement score is calculated through a variety of student testing and the growth score is calculated using the EVASS School Accountability Growth Composite Index [2]. Schools are assigned a letter grade where A: 100-85 points, B: 84-70 points, C: 69-55 points, D: 54-40 points and F: less than 40 points.  Schools that receive grades of D or F are required by to inform parents of the school district.  In 2016, the North Carolina General Assembly passed legislation creating the Achievement School District(ASD). This school district is run by a private organization and are run as charter schools [3].
# 
# This data set contains 334 features describing 2,443 schools.  The data includes testing results used to derive the SPG described above.  It also contains school financial data, demographic information, attendance, and student behavior data measured by metrics such as suspension and expulsions. We can look into all these different types of information to see if any correlation with school performances exists, both good and bad.  Do poorly performing schools line up with any specific demographics?  Are there school financial situations that help attribute to a school’s performance? Finding correlations of this data with SPG and being able to use that information in a predictive analysis algorithm may help educators identify schools before the performance metrics deteriorate, allowing them to intervene. The end result of all the testing and analysis is providing all students a fair and equal opportunity at a quality education.
# 
# We are examining this data set from the point of view of trying find correlations with SPG Score for each LEA.  SPG Score is a continuous variable, but there is also a group of categorical variables describing SPG Score.  We are choosing to examine the data from continuous variable point of view, although at times we do use the categorical group for certain visualizations. This choice leads us to a use regression model, where we can validate that model with 'k-fold' cross validation for accuracy.
# 
# [1] source: http://schools.cms.k12.nc.us/jhgunnES/Documents/School%20Performance%20Grade%20PP%20January%2014,%202015%20(1).pptx<br>
# [2] (EVASS Growth information available at http://www.ncpublicschools.org/effectiveness-model/evaas/selection/)<br>
# [3] source: https://www.ncforum.org/committee-on-low-performing-schools/
# 
# ###citation: Drew J., The Belk Endowment Educational Attainment Data Repository for North Carolina Public Schools, (2018), GitHub repository, https://github.com/jakemdrew/EducationDataNC
# 

# # Data Meaning Type 
# 

# The comprehensive description of all 334 attributes can be found in the data-dictionary.pdf associated with the NC Report Card database provided by Dr. Drew. We were interested in 60 variables moving forward in the course. We visualize several attributes of interest in this report. The most interesting relationships will be between funding, race, and achievement scores. 

# <img src="files/data_meaning.jpg"> 



# In[44]:



SPG_Grade_col = school_data.filter(regex=('^SPG\WGrade')).columns
y = school_data[SPG_Grade_col].apply(lambda row:'A' if row.any()!=1 else 
                                 row[0]*'A+NG'+row[1]*'B'+row[2]*'C'+row[3]*'D'+row[4]*'F'+row[5]*'I',axis=1)

X = school_data[school_data.columns.drop(list(school_data.filter(regex='^SPG\WGrade|unit_code')))]




# In[47]:


#split X and y into test and train sets.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)


# In[48]:


#applied a scaling procedure to scale the size of variables in the x dataframe

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_scale = sc.fit_transform(X_train)
X_test_scale = sc.transform(X_test)


# In[49]:


#run PCA on 62 components of the dataset, which explains 85% of the variance.
# =============================================================================
# 
from sklearn.decomposition import PCA
pca = PCA(n_components=60)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
d = {'ratio':pca.explained_variance_ratio_,'total':pca.explained_variance_ratio_.cumsum()}
# 
pca = PCA(n_components=60)
X_train_scale = pca.fit_transform(X_train_scale)
X_test_scale = pca.transform(X_test_scale)
d_scale = {'ratio':pca.explained_variance_ratio_,'total':pca.explained_variance_ratio_.cumsum()}
# 
# 
# 
# =============================================================================

# In[51]:

#Run a Kfolds cross validation model on the data set and predicted y from the set

from sklearn.linear_model import LogisticRegressionCV
from sklearn.cross_validation import KFold
fold = KFold(len(y_train), n_folds=10, shuffle=True)
classifier = LogisticRegressionCV(Cs=list(np.power(10.0, np.arange(-10, 10)))
        ,penalty='l2'
        ,scoring='roc_auc'
        ,cv=fold
        ,max_iter=4000
        ,fit_intercept=True
       ,solver='newton-cg')

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

#print(classifier.scores_[1].max())

classifier.fit(X_train_scale, y_train)
y_pred_scale = classifier.predict(X_test_scale)

#print(classifier.scores_[1].max())


# In[52]:


#Created a confusion matrix



cm = confusion_matrix(y_test, y_pred)

print(cm)
print(accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred_scale)

print(cm)
print(accuracy_score(y_test, y_pred_scale))


# In[53]:

from sklearn import svm

classifier = svm.SVC()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

classifier.fit(X_train_scale, y_train)
y_pred_scale = classifier.predict(X_test_scale)

# In[53]:
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
print(accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred_scale)

print(cm)
print(accuracy_score(y_test, y_pred_scale))
# In[53]:

# Import `RandomForestClassifier`
from sklearn.ensemble import RandomForestClassifier



# Build the model
classifier = RandomForestClassifier()

# Fit the model
classifier.fit(X_train, y_train)

# Print the results
y_pred = classifier.predict(X_test)

classifier.fit(X_train_scale, y_train)
y_pred_scale = classifier.predict(X_test_scale)



# ## Tableau: AP Scores Vs Teacher Salaries 
# 
# Here we have an dual axis line graph demonstrating the relationship between AP score of 3 points or higher with the percent expenditure on teacher salaries. We can see some visual corrolation between rising scores and an increase in percentage. 

# ##Incase you don't have a Tableau account, I added a screenshot
# <img src="files/Tableau_pic.jpg"> 
# 
# In[ ]:

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
print(accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred_scale)

print(cm)
print(accuracy_score(y_test, y_pred_scale))
# In[ ]:
import xgboost
classifier = XGBClassifier()
kfold = StratifiedKFold(n_splits=10,shuffle=True,random_state=7)
results = cross_val_score(classifier,X_train_scale, y_train,cv=kfold )
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
# Fit the model
classifier.fit(X_train, y_train)

# Print the results
y_pred = classifier.predict(X_test)

classifier.fit(X_train_scale, y_train)
y_pred_scale = classifier.predict(X_test_scale)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
print(accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred_scale)

print(cm)
print(accuracy_score(y_test, y_pred_scale))


# In[ ]:
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

SPG_Grade_col = school_data.filter(regex=('^SPG\WGrade')).columns
y = school_data[SPG_Grade_col].apply(lambda row:'A' if row.any()!=1 else 
                                 row[0]*'A+NG'+row[1]*'B'+row[2]*'C'+row[3]*'D'+row[4]*'F'+row[5]*'I',axis=1)

X = school_data[school_data.columns.drop(list(school_data.filter(regex='^SPG\WGrade|unit_code')))]


encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_y)

# In[47]:


#split X and y into test and train sets.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size=.2)


# In[48]:
seed = 7
np.random.seed(seed)

#applied a scaling procedure to scale the size of variables in the x dataframe

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_scale = sc.fit_transform(X_train)
X_test_scale = sc.transform(X_test)

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
 	
from keras.layers import Dense, Dropout, Activation, Flatten
def baseline_model():
# Initialising the ANN
    model = Sequential()
    
    # Adding the input layer and the first hidden layer
    model.add(Dense(units = 150, kernel_initializer = 'uniform', activation = 'relu', input_dim = 327))
    
    # Adding the second hidden layer
    model.add(Dense(units = 75, kernel_initializer = 'uniform', activation = 'relu'))
    
        # Adding the second hidden layer
    model.add(Dense(units = 37, kernel_initializer = 'uniform', activation = 'relu'))
    
    model.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))
    
    # Adding the output layer
    model.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'softmax'))
    
    # Compiling the ANN
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

results = cross_val_score(estimator, X_train_scale, y_train, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
