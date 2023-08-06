#!/usr/bin/env python
# coding: utf-8

# # Loan Status Prediction using SVM

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# Data Collection and Processing
# 

# In[2]:


loan= pd.read_csv(r"C:\Users\user\Downloads\archive (6)\train_u6lujuX_CVtuZ9i (1).csv")


# In[3]:


loan.head()


# In[4]:


loan.shape


# In[5]:


loan.describe()


# In[6]:


loan.info()


# In[7]:


## finding no. of null values 
loan.isnull().sum()


# In[8]:


## dropping missing values 
loan=loan.dropna()


# In[9]:


##label encoding - we'll take Y as 1 and N as 0
loan.replace({'Loan_Status':{'Y':1,'N':0}},inplace=True)


# In[10]:


loan.head()


# In[11]:


#dependent column values
loan['Dependents'].value_counts()


# In[12]:


loan.info()


# In[13]:


# replacing the value of 3+ to 4
loan['Dependents']=loan['Dependents'].astype(str)
loan['Dependents']=loan['Dependents'].replace("3+","4")


# In[14]:


loan['Dependents'].value_counts()


# In[30]:


# data visualisation
# education and loan status
sns.countplot(x='Education',hue='Loan_Status',palette="Greens",data=loan)


# In[31]:


# marriage and loan status
sns.countplot(x='Married',hue='Loan_Status',palette="Greens",data=loan)


# In[33]:


sns.barplot(x='Loan_Status',y='ApplicantIncome',palette="Oranges",data = loan)


# In[32]:


sns.countplot(x='Self_Employed',hue='Loan_Status',palette="Oranges",data=loan)


# In[26]:


sns.countplot(x='Gender',hue='Loan_Status',palette="Oranges",data=loan)


# In[34]:


sns.countplot(x='Credit_History',hue='Loan_Status',palette="Purples_r",data=loan)


# In[25]:


sns.countplot(x='Property_Area',hue='Loan_Status',palette="Purples_r",data=loan)


# In[24]:


sns.countplot(x='Dependents',hue='Loan_Status',palette="Purples_r",data=loan)


# In[35]:


# convert categorical data to numerical values
loan.replace({'Married':{'Yes':1,'No':0},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},
              'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)


# In[36]:


loan.head()


# In[37]:


loan.to_csv('Cleaned loan.csv')


# In[38]:


#separating the data and label
x= loan.drop(columns=['Loan_ID','Loan_Status'],axis=1)
y= loan['Loan_Status']


# In[39]:


x


# In[40]:


y


# In[41]:


#Train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,stratify=y,random_state=2)


# In[42]:


print(x.shape,x_test.shape,x_train.shape)


# In[43]:


##training the model: Support Vector Machine method


# In[44]:


classifier=svm.SVC(kernel='linear')


# In[45]:


# training the support vector machine model


# In[46]:


classifier.fit(x_train,y_train)


# # model evaluation

# In[47]:


# accuracy score on training data
x_train_prediction= classifier.predict(x_train)


# In[48]:


training_data_accuracy=accuracy_score(x_train_prediction,y_train)


# In[49]:


training_data_accuracy


# In[50]:


# accuracy score on test data
x_test_prediction= classifier.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)


# In[51]:


test_data_accuracy


# # making a predictive model

# In[52]:


Input_data = (1,1,2,1,0,3200,700.0,70.0,360.0,1.0,2)

Input_data_as_numpy_array = np.asarray(Input_data)

# reshaping the data as we are pridicting for one instance

input_reshaping = Input_data_as_numpy_array.reshape(1,-1)

x = classifier.predict(input_reshaping)

print(x)
if(x==1):
    print("Yes,Loan will be granted")
if(x==0):
    print("No, Loan will not be granted")

