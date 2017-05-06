# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 23:41:26 2017

@author: DELL1
"""
"""
Created on Fri Mar 24 03:22:12 2017

@author: DELL1
"""

import numpy as np
import pandas as pd
from sklearn import linear_model,cross_validation,neighbors,svm
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import make_scorer, accuracy_score
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.metrics import mean_squared_error
#from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pickle

#read data
dataframe = pd.read_csv('C:\\Users\\DELL1\\Desktop\\pro.csv')
dataframe.set_index('Year',inplace=True)
print(dataframe.head())

#X=dataframe[:,['Gdp','Deflator','Rural','Credit_PSB','Tot_Reserves','FPI','Agri_Land','Ex_Imp','ODA','CO2','cap_PS']].values
#Y=dataframe[:,['INFLATION']].values

X=dataframe.iloc[:,1]
Y=dataframe.iloc[:,0]

# step 3: get features (x) and scale the features
# get x and convert it to numpy array
x = dataframe.ix[:,1:11].values
scaler = MinMaxScaler(feature_range=(0, 1))
x_std = scaler.fit_transform(x)

# step 4: get class labels y and then encode it into number 
# get class label data
y = dataframe.ix[:,0].values
# encode the class label
'''class_labels = np.unique(y)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
'''
y= dataframe.ix[:,0].values
scaler = MinMaxScaler(feature_range=(0, 1))
y_std = scaler.fit_transform(y)

x_std=np.asmatrix(x_std)
y_std=np.asmatrix(y_std).T


x_std=np.delete(x_std,(16),0)
x_std=np.delete(x_std,(14),0)
y_std=np.delete(y_std,(14),0)
y_std=np.delete(y_std,(16),0)


print (x_std,y_std)


pca = decomposition.PCA(n_components=3)
pca.fit(x_std)
X = pca.transform(x_std)
print ("PCA",X)

'''
body_reg = linear_model.LinearRegression()
body_reg.fit(x_std,y_std)

#visualize results
plt.scatter(x_std,y_std,color='blue')
plt.plot(x_std,body_reg.predict(y_std),color='black')
plt.show()
'''

# step 5: split the data into training set and test set
test_percentage = 0.40
x_train, x_test, y_train, y_test =cross_validation.train_test_split(x_std, y_std, test_size = test_percentage, random_state = 243)

clf=neighbors.KNeighborsRegressor()

k_range = list(range(1, 32))
print(k_range)
param_grid = dict(n_neighbors=k_range)
print(param_grid)

grid = GridSearchCV(clf, param_grid, cv=10, scoring='neg_mean_squared_error')
grid.fit(x_std, y_std)

grid.grid_scores_
print(grid.grid_scores_[0].parameters)
print(grid.grid_scores_[0].cv_validation_scores)
print(grid.grid_scores_[0].mean_validation_score)

grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
print(grid_mean_scores)

plt.plot(k_range, grid_mean_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')

print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)

# Set the clf to the best combination of parameters
clf = grid.best_estimator_

# Fit the best algorithm to the data.
clf.fit(x_train,y_train)

print(x_test)
predictions = clf.predict(x_test)
print('prediction',predictions)
print('ytest')
print(y_test)
mse =mean_squared_error(predictions, y_test)
print(mse)
'''
plt.plot(predictions)
plt.plot(y_test)
plt.xlabel('Year')
plt.ylabel('value')
'''
with open('logmodel.pkl', 'wb') as fid:
    pickle.dump(clf, fid,2)    

with open('ytest', 'wb') as fid:
    pickle.dump(y_test, fid,2)  
    

with open('pred', 'wb') as fid:
    pickle.dump(predictions, fid,2)       
'''   
combined = np.vstack((predictions, y_test)).T
print(combined)

combi =list(zip((predictions,y_test)))
print(combi)

'''
'''
def f(i):
    i=i*(36.23)-7.63
f = np.vectorize(f)
preditions=np.apply_along_axis(f,-1,predictions)
print(predictions)
'''
'''
scaler = MinMaxScaler(feature_range=(0, 1))
y_test1 = scaler.inverse_transform(y_test)

predictions1=scaler.inverse_transform(predictions)

print(predictions1)
print(y_test1)

'''

print("========================================")

