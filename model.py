import pandas as pd
import numpy as np
import csv

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_curve, roc_auc_score
# two ways hyperparameters turning
from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import RandomizedSearchCV

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

"""
Uncomment tune parameters code to run the GridSearchCV for parameter tuning
Some of them may run a little bit longer to get the optimal parameters
"""



"""
Data preprocessing
"""
# read in data set
train_original = pd.read_csv("income2022f/train_final.csv")
test_original = pd.read_csv("income2022f/test_final.csv")

# get a copy of datasets
train = train_original.copy()
test = test_original.copy()

# read in another format
train_np = np.genfromtxt("income2022f/train_final.csv", delimiter = ',')

# get features 
features = list(train.columns)
# get each column type
types = train.dtypes

# handle categorical data
# ordinal encode
num_cols = []
cate_cols = []
for col_name, dtype in types.items():
    if dtype == 'object':
        cate_cols.append(col_name)
# =============================================================================
#         # unique value of a non-int type column
#         unique_list = train[col_name].unique()
#         
#         # dictionary to map
#         dic = {}
#         for i, name in enumerate(unique_list):
#             dic[name] = i
#         
#         #if col_name == 'native.couuntry':
#             #dic['Holand-Netherlands'] = 41
#         #print(dic)
#         # takes a dictionary with information on how to convert the values
#         train[col_name] = train[col_name].map(dic)
#         test[col_name] = test[col_name].map(dic)
# =============================================================================
    else:
        num_cols.append(col_name)

# check if there is null value
train_explore = train.isnull().any()
test_explore = test.isnull().any()

# by exploratory data analysis, there is missing value in native.country column
test['native.country'] = test['native.country'].fillna("Canada")     
y = train['income>50K']
train2 = train.drop('income>50K', axis=1, inplace=False)

# handle numerical data
scaler = StandardScaler()
num_cols = num_cols[0:6]
train2[num_cols] = scaler.fit_transform(train_original[num_cols])
test[num_cols]= scaler.fit_transform(test_original[num_cols])


# handle categorical data
encoder = OneHotEncoder(sparse=False)
combo = pd.concat([train2, test], axis=0)
combo.reset_index(inplace=True, drop=True)
combo.drop('ID' ,axis=1, inplace=True)

df_encoded = pd.DataFrame(encoder.fit_transform(combo[cate_cols]))
df_encoded.columns = encoder.get_feature_names(cate_cols)
# Replace Categotical Data with Encoded Data
combo = combo.drop(cate_cols ,axis=1)
combo = pd.concat([df_encoded, combo], axis=1)


train2 = combo.iloc[0:25000,:]
test = combo.iloc[25000: 48842,:]

# seperate the feature columns from the target column
X = train2



"""
Models:
"""


"""
Decision Tree
"""
# tune parameters
# para_dict = {'criterion': ['gini', 'entropy'],
#              'max_depth': [6,8,10,12]}

# grid = GridSearchCV(DecisionTreeClassifier(), 
#                     param_grid = para_dict,
#                     cv = 10)

# grid.fit(X, y)
# params, score = grid.best_params_, grid.best_score_



# generate decision tree
dtree = DecisionTreeClassifier(criterion='gini', max_depth=10)
dtree = dtree.fit(train2, y)

predict = dtree.predict(test)   

predict_prob = dtree.predict_proba(test)[:,1]
predict_prob_train = dtree.predict_proba(train2)[:,1]
print("Decision Tree:")
print("ROC_AUC score:", roc_auc_score(train_np[1:,-1], predict_prob_train))


"""
Random Forest
"""
# tune parameters, this may run a little bit longer
# param_dict = { 
#     'n_estimators': [200, 500],
#     'max_features': ['auto', 'sqrt'],# log2
#     #'max_depth' : [6,8,10,12],
#     'criterion' :['gini', 'entropy']
# }

# grid = GridSearchCV(RandomForestClassifier(), 
#                     param_grid = param_dict,
#                     cv = 10)

# grid.fit(X, y)
# params, score = grid.best_params_, grid.best_score_

# generate random forest
randomforest = RandomForestClassifier(criterion='entropy', max_features='sqrt', n_estimators=500)
randomforest.fit(train2, y)

predict = randomforest.predict(test)
predict_prob = randomforest.predict_proba(test)[:,1]
predict_prob_train = randomforest.predict_proba(train2)[:,1]
print("Random Forest:")
print("ROC_AUC score:", roc_auc_score(train_np[1:,-1], predict_prob_train))



"""
Perceptron
didn't have predict_proba method, so not consider this one'
"""

# tune parameters
# param_dict = {'alpha': [0.001, 0.005, 0.01], 'fit_intercept': [True], 'max_iter': [500]}
# grid = GridSearchCV(Perceptron(), 
#                     param_grid = param_dict,
#                     cv = 10)

# grid.fit(X, y)
# params, score = grid.best_params_, grid.best_score_

perceptron = Perceptron(alpha = 0.001, fit_intercept=True, max_iter=500)
perceptron.fit(train2, y)
predict_prob = perceptron.predict(test)
# = perceptron.predict_proba(train2)[:,1]
#print("ROC_AUC score:", roc_auc_score(train_np[1:,-1], predict_prob_train))



"""
Gradient Boosting
"""
# tune parameters
# param_dict ={'n_estimators':[100,200],'learning_rate':[0.05,0.1,0.5],'max_depth':[5,10]}
# grid=GridSearchCV(GradientBoostingClassifier(),param_dict,cv=10)

# grid.fit(X, y)
# params, score = grid.best_params_, grid.best_score_

# generate gradient boosting
gradientb = GradientBoostingClassifier(n_estimators=200,learning_rate=0.05,max_depth=5)
gradientb = gradientb.fit(train2, y)

predict = gradientb.predict(test)   

predict_prob = gradientb.predict_proba(test)[:,1]
predict_prob_train = gradientb.predict_proba(train2)[:,1]
print("Gradient Boosting:")
print("ROC_AUC score:", roc_auc_score(train_np[1:,-1], predict_prob_train))


"""
Logistic Regression
"""
# tune parameters
# param_dict ={'max_iter': [500,1000]}
# grid=GridSearchCV(LogisticRegression(),param_dict,cv=10)

# grid.fit(X, y)
# params, score = grid.best_params_, grid.best_score_

# generate logistic regression
lr = LogisticRegression(max_iter=500)
lr = lr.fit(train2, y)

predict = lr.predict(test)   

predict_prob = lr.predict_proba(test)[:,1]
predict_prob_train = lr.predict_proba(train2)[:,1]
print("Logistic Regression:")
print("ROC_AUC score:", roc_auc_score(train_np[1:,-1], predict_prob_train))





"""
Write results to files
"""
# write to file
with open('submit.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['ID', 'Prediction'])
    for i in range(len(predict_prob)):
        writer.writerow([str(i+1), str(predict_prob[i])])



