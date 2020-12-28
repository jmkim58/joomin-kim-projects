# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC,SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv('census-income.data.csv',header=None)
test_data = pd.read_csv('census-income.test.csv',header=None)
data = data.fillna(data.mean())

y_train = data[14]
X_train = data.drop([14],axis=1)

y_test = test_data[14]
X_test = test_data.drop([14],axis=1)

le_1 = preprocessing.LabelEncoder()
le_3 = preprocessing.LabelEncoder()
le_5 = preprocessing.LabelEncoder()
le_6 = preprocessing.LabelEncoder()
le_7 = preprocessing.LabelEncoder()
le_8 = preprocessing.LabelEncoder()
le_9 = preprocessing.LabelEncoder()
le_13 = preprocessing.LabelEncoder()
le_y = preprocessing.LabelEncoder()

le_1 = le_1.fit(X_train[1])
le_3 = le_3.fit(X_train[3])
le_5 = le_5.fit(X_train[5])
le_6 = le_6.fit(X_train[6])
le_7 = le_7.fit(X_train[7])
le_8 = le_8.fit(X_train[8])
le_9 = le_9.fit(X_train[9])
le_13 = le_13.fit(X_train[13])
le_y = le_y.fit(y_train)

X_train[1] = le_1.transform(X_train[1])
X_train[3] = le_3.transform(X_train[3])
X_train[5] = le_5.transform(X_train[5])
X_train[6] = le_6.transform(X_train[6])
X_train[7] = le_7.transform(X_train[7])
X_train[8] = le_8.transform(X_train[8])
X_train[9] = le_9.transform(X_train[9])
X_train[13] = le_13.transform(X_train[13])
y_train = le_y.fit_transform(y_train)

min_max_scaler = preprocessing.MinMaxScaler(feature_range =(0, 1)) 
X_train = min_max_scaler.fit_transform(X_train) 

X_test[1] = le_1.transform(X_test[1])
X_test[3] = le_3.transform(X_test[3])
X_test[5] = le_5.transform(X_test[5])
X_test[6] = le_6.transform(X_test[6])
X_test[7] = le_7.transform(X_test[7])
X_test[8] = le_8.transform(X_test[8])
X_test[9] = le_9.transform(X_test[9])
X_test[13] = le_13.transform(X_test[13])
y_test = le_y.fit_transform(y_test)

X_test = min_max_scaler.fit_transform(X_test) 


# multinomial naive bayes 
clf = MultinomialNB()
clf.fit(X_train,y_train)
y_pred_training = clf.predict(X_train)
print("training Multinomial Naive Bayes:  ",accuracy_score(y_train, y_pred_training))
y_pred = clf.predict(X_test)
print("testing Multinomial Naive Bayes: ",accuracy_score(y_test, y_pred))

# knn
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_training_knn = knn.predict(X_train)
print("training KNN:  ",accuracy_score(y_train, y_pred_training_knn))
pred_knn = knn.predict(X_test)
print("testing KNN: ",accuracy_score(y_test, pred_knn))

# decision tree
clf_tree = DecisionTreeClassifier(random_state=2)
clf_tree = clf_tree.fit(X_train,y_train)
y_pred_training_tree = clf_tree.predict(X_train)
print("training Decision Tree:  ",accuracy_score(y_train, y_pred_training_tree))
tree_pred = clf_tree.predict(X_test)
print("testing Decision Tree: ",accuracy_score(y_test, tree_pred))

# svm
SVC_model = SVC()
SVC_model.fit(X_train, y_train)
y_pred_training_svc = SVC_model.predict(X_train)
print("training Support Vector Classifier:  ",accuracy_score(y_train, y_pred_training_svc))
svc_pred = SVC_model.predict(X_test)
print("testing Support Vector Classifier: ",accuracy_score(y_test, svc_pred))

# random forest
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0) 
regressor.fit(X_train, y_train) 
forest_pred_train = regressor.predict(X_train) # test the output by changing values 
forest_pred_train = [round(num) for num in forest_pred_train]
print("training Random Forest: ",accuracy_score(y_train, forest_pred_train))
forest_pred = regressor.predict(X_test) # test the output by changing values 
forest_pred = [round(num) for num in forest_pred]
print("testing Random Forest: ",accuracy_score(y_test, forest_pred))

# extra trees classifier
extratree_regressor = ExtraTreesClassifier(n_estimators = 100, random_state = 0) 
extratree_regressor.fit(X_train, y_train) 
extratree_pred_train = extratree_regressor.predict(X_train) # tests output by changing values 
print("training Extra Tree Classifier: ",accuracy_score(y_train, extratree_pred_train))
extratree_pred = extratree_regressor.predict(X_test) # tests output by changing values 
print("testing Extra Tree Classifier: ",accuracy_score(y_test, extratree_pred))

# final accuracy
i=0
pred_ensemble_train = []
while i<len(X_train):
    t = y_pred_training_knn[i]+y_pred_training_svc[i]+y_pred_training_tree[i]+forest_pred_train[i]+extratree_pred_train[i]
    t = round(t/5)
    pred_ensemble_train.append(t)
    i+=1
print("training ensemble: ",accuracy_score(y_train, pred_ensemble_train))
i=0
pred_ensemble = []
while i<len(X_test):
    t = pred_knn[i]+svc_pred[i]+tree_pred[i]+forest_pred[i]+extratree_pred[i]
    t = round(t/5)
    pred_ensemble.append(t)
    i+=1
print("testing ensemble: ",accuracy_score(y_test, pred_ensemble))

