#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 11:28:50 2019

@author: abhishek
"""
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#importing dataset
from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
#creating dataframe
df_cancer=pd.DataFrame(np.c_[cancer['data'],cancer['target']],columns=np.append(cancer['feature_names'],['target']))
#visualising data
#for pairwise
sns.pairplot(df_cancer,hue='target',vars=['mean radius','mean texture','mean perimeter','mean area','mean smoothness'])
#to get the count 
sns.countplot(df_cancer['target'],label="count")
#scatter plot against
sns.scatterplot(x='mean smoothness',y='mean radius',hue='target',data=df_cancer)
#check correlation
plt.figure(figsize=(20,10))
sns.heatmap(df_cancer.corr(),annot=True)
#model training
X=df_cancer.drop(['target'],axis=1)
y=df_cancer['target']
#splitting the dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test=sc_x.transform(X_test)
#svm
from sklearn.svm import SVC
svm_class=SVC()
svm_class.fit(X_train,y_train)
#predicting
y_pred=svm_class.predict(X_test)
#confusion matrix and evaluation
from sklearn.metrics import confusion_matrix,classification_report
cm=confusion_matrix(y_test,y_pred)
print(cm)
sns.heatmap(cm)
print(classification_report(y_test,y_pred))
#improving the model
from sklearn.model_selection import GridSearchCV
parameters={'C':[0.1,0.5,1,10,100],'gamma':[1,0.1,0.01,0.001,0.0001]}
grid=GridSearchCV(estimator=svm_class,param_grid=parameters,scoring='accuracy')
grid.fit(X_train,y_train)
grid.best_estimator_
grid.best_params_
grid_pred=grid.predict(X_test)
cm=confusion_matrix(y_test,grid_pred)
print(cm)