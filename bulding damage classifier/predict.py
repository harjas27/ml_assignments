# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 10:07:13 2018

@author: dell pc
"""

import pandas as pd

#%%
df1=pd.read_csv('C:\\Users\\dell pc\\Desktop\\Dataset\\Building_Ownership_Use.csv')
df2=pd.read_csv('C:\\Users\\dell pc\\Desktop\\Dataset\\Building_Structure.csv')

#%%
df=pd.read_csv('C:\\Users\\dell pc\\Desktop\\Dataset\\test.csv')
#%%
print(list(df))
#%%
df1.drop(['district_id', 'vdcmun_id'],axis=1,inplace=True)
df2.drop(['district_id', 'vdcmun_id'],axis=1,inplace=True)
#%%
df = df.join(df1.set_index('building_id'), on='building_id')
#%%
df.drop('ward_id',axis=1,inplace=True)
df = df.join(df2.set_index('building_id'), on='building_id')
#%%
df_=df
#%%
import numpy as np
#%%
for i in list(df):
    print(i,df[i].dtype)
    
#%%    
lis=['area_assesed','legal_ownership_status','land_surface_condition','foundation_type','roof_type','ground_floor_type','other_floor_type','position','plan_configuration','condition_post_eq']
for i in lis:
    df=pd.concat([df,pd.get_dummies(df[i],prefix=i)],axis=1)
    df.drop(i,axis=1,inplace=True)
    print(df.shape)
def convert(x):
    c=x[-1]
    return int(c)
#df['damage_grade']=df['damage_grade'].apply(lambda x: convert(x))
#%%
df.to_csv('C:\\Users\\dell pc\\Desktop\\Dataset\\test.csv',index=False)

#%%
from sklearn import preprocessing,cross_validation,svm,tree
from sklearn.linear_model import LogisticRegression
#%%
df=pd.read_csv('C:\\Users\\dell pc\\Desktop\\Dataset\\train.csv')
#df=df.set_index('building_id')
#%%
df.dropna(axis=0,inplace=True)
#%%
x = df.drop('damage_grade',axis=1)
y = df['damage_grade']
min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)
#%%
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)
#%%
clf = LogisticRegression()
clf.fit(x_train,y_train)
#%%
accuracy = clf.score(x_test,y_test)
print(accuracy)
#%%
import pickle
file='C:\\Users\\dell pc\\Desktop\\Dataset\\lr.pkl'
clf=LogisticRegression()
with open(file,'rb') as f:
    clf=pickle.load(f)
#%%    
df=df.fillna(0)
x = df.drop('building_id',axis=1)
min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)
#%%
my_prediction = clf.predict(x)
#%%
my_solution = pd.DataFrame(my_prediction, df['building_id'], columns = ['damage_grade'])
#%%
def convert_back(x):
    return 'Grade '+str(x)
my_solution['damage_grade']=my_solution['damage_grade'].apply(lambda x: convert_back(x))
#%%
my_solution.astype(str).to_csv('C:\\Users\\dell pc\\Desktop\\Dataset\\my_solution_lr.csv', index_label = ['building_id'])
#%%
smp =pd.read_csv('C:\\Users\\dell pc\\Desktop\\Dataset\\sample_submission.csv')
#%%
