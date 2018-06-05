# In[1]:

#split data into X and y dataframes

SPG_Grade_col = school_data.filter(regex=('^SPG\WGrade')).columns
y = school_data[SPG_Grade_col].apply(lambda row:'A' if row.any()!=1 else 
                                 row[0]*'A+NG'+row[1]*'B'+row[2]*'C'+row[3]*'D'+row[4]*'F'+row[5]*'I',axis=1)

X = school_data[school_data.columns.drop(list(school_data.filter(regex='^SPG\WGrade|unit_code')))]

# In[1]:

#split X and y into test and train sets.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

# In[1]:

#applied a scaling procedure to scale the size of variables in the x dataframe

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_scale = sc.fit_transform(X_train)
X_test_scale = sc.transform(X_test)


# In[1]:

#Initializing the Logistic Regression function to a classifier variable
classifier = LogisticRegression()

#Fitting the Logistic Regression Classifier to the original dataset and then predicting.
classifier.fit(X_train, y_train)
y_pred_Log_Reg = classifier.predict(X_test)

#Fitting the Logistic Regression Classifier to the scaled dataset and then predicting.
classifier.fit(X_train_scale, y_train)
y_pred_Log_Reg_scale = classifier.predict(X_test_scale)

# =============================================================================
# Below we have the confusion matrix and accuracy score for SVM without scaling. 
# Based on the accuracy score it is evident that Logistic Regression handles non-standardized variables relatively well.
# 
# =============================================================================

# In[52]:


#Created a confusion matrix



cm_Log_Reg = confusion_matrix(y_test, y_pred_Log_Reg)

print(cm_Log_Reg)
print(accuracy_score(y_test, y_pred_Log_Reg))

cm_Log_Reg_scale = confusion_matrix(y_test, y_pred_Log_Reg_scale)

print(cm_Log_Reg_scale)
print(accuracy_score(y_test, y_pred_Log_Reg_scale))


# In[53]:

#Initializing the SVM function to a classifier variable
classifier = svm.SVC()

#Fitting the SVM Classifier to the original dataset and then predicting.
classifier.fit(X_train, y_train)
y_pred_svm = classifier.predict(X_test)

#Fitting the SVM Classifier to the scaled dataset and then predicting.
classifier.fit(X_train_scale, y_train)
y_pred_svm_scale = classifier.predict(X_test_scale)

# In[53]:
cm_svm = confusion_matrix(y_test, y_pred_svm)

print(cm_svm)
print(accuracy_score(y_test, y_pred_svm))

#Assigning a confusion matrix with the scaled dataset.
cm_svm_scale = confusion_matrix(y_test, y_pred_svm_scale)

print(cm_svm_scale)
print(accuracy_score(y_test, y_pred_svm_scale))
