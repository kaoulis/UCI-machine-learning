import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', 500)

#Load data
data_array = np.genfromtxt("agaricus-lepiota.data", dtype='str', delimiter=",")
df = pd.DataFrame(data=data_array)

#############################
# 1. Pre-processing #########
#############################

# Balanced?
print((df[0][df[0] == 'e']).count())
print((df[0][df[0] == 'p']).count())

# Missing values
df = df.replace('?', np.nan)
print(df.isnull().sum())

# Fill in the missing values with the most common value
df[11].fillna(df[11].mode()[0], inplace=True)

################################
# Encoding #####################
################################
df[0] = LabelEncoder().fit_transform(df[0])
df = pd.get_dummies(df)
# fix some varibles that should not be converted to dummies.
df.drop(['4_f', '8_b', '10_e', '16_p'], axis=1, inplace=True)

################################
# Feature selection-extraction #
################################
# seperate target class from the other features
labels = df[0].values
df.drop(0, axis=1, inplace=True)
features = df.values
# Data split
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=7)

# fitting random forest for feature importance
trainedforest = RandomForestClassifier(n_estimators=100, random_state=7).fit(x_train, y_train)
sel = SelectFromModel(trainedforest)
sel.fit(x_train, y_train)

# plot feature importances
plt.figure(figsize=(5, 4))
feat_importances = pd.Series(trainedforest.feature_importances_, index=df.columns.values)
feat_importances.nlargest(10).plot(kind='barh')
plt.xlabel('Importance')
plt.ylabel('Features')
# plt.savefig("plt_featureImportance.png")
plt.show()

# Complete feature selection
selected_features = df[['5_n', '5_f', '8_n']]
features = selected_features.to_numpy()

###################################################
# Classifiers with cross validation ###############
###################################################

#Logistic regression
lg_scores = {'Accuracy': [], 'Precision': [], 'Recall': []}
lg_AUC = []
lg = LogisticRegression()
# 10-Fold cross validation
cv = KFold(n_splits=10, shuffle=True, random_state=7)
start = time.process_time()
for train_index, test_index in cv.split(features):
    X_train, X_test, y_train, y_test = features[train_index], features[test_index], labels[train_index], labels[
        test_index]
    lg.fit(X_train, y_train)
    y_pred_lg = lg.predict(X_test)
    lg_scores['Accuracy'].append(metrics.accuracy_score(y_test, y_pred_lg))
    lg_scores['Precision'].append(metrics.precision_score(y_test, y_pred_lg))
    lg_scores['Recall'].append(metrics.recall_score(y_test, y_pred_lg))
    lg_AUC.append(metrics.roc_auc_score(y_test, y_pred_lg))

print('Time: ', time.process_time() - start)
print('Accuracy: ', np.mean(lg_scores['Accuracy']))
print('Precision: ', np.mean(lg_scores['Precision']))
print('Recall: ', np.mean(lg_scores['Recall']))
print('F1-Score: ', 2 * np.mean(lg_scores['Precision']) * np.mean(lg_scores['Recall']) /
      (np.mean(lg_scores['Precision']) + np.mean(lg_scores['Recall'])))
print('AUC:', np.mean(lg_AUC))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

# Decision Tree
dc_scores = {'Accuracy': [], 'Precision': [], 'Recall': []}
dc_AUC = []
dc = DecisionTreeClassifier(random_state=7)
cv = KFold(n_splits=10, shuffle=True, random_state=7)
start = time.process_time()
for train_index, test_index in cv.split(features):
    X_train, X_test, y_train, y_test = features[train_index], features[test_index], labels[train_index], labels[
        test_index]
    dc.fit(X_train, y_train)
    y_pred_dc = dc.predict(X_test)
    dc_scores['Accuracy'].append(metrics.accuracy_score(y_test, y_pred_dc))
    dc_scores['Precision'].append(metrics.precision_score(y_test, y_pred_dc))
    dc_scores['Recall'].append(metrics.recall_score(y_test, y_pred_dc))
    dc_AUC.append(metrics.roc_auc_score(y_test, y_pred_dc))

print('Time: ', time.process_time() - start)
print('Accuracy: ', np.mean(dc_scores['Accuracy']))
print('Precision: ', np.mean(dc_scores['Precision']))
print('Recall: ', np.mean(dc_scores['Recall']))
print('F1-Score: ', 2 * np.mean(dc_scores['Precision']) * np.mean(dc_scores['Recall']) /
      (np.mean(dc_scores['Precision']) + np.mean(dc_scores['Recall'])))
print('AUC:', np.mean(lg_AUC))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


# Random Forest
rf_scores = {'Accuracy': [], 'Precision': [], 'Recall': []}
rf_AUC = []
rf = RandomForestClassifier(random_state=7)
cv = KFold(n_splits=10, shuffle=True, random_state=7)
start = time.process_time()
for train_index, test_index in cv.split(features):
    X_train, X_test, y_train, y_test = features[train_index], features[test_index], labels[train_index], labels[
        test_index]
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rf_scores['Accuracy'].append(metrics.accuracy_score(y_test, y_pred_rf))
    rf_scores['Precision'].append(metrics.precision_score(y_test, y_pred_rf))
    rf_scores['Recall'].append(metrics.recall_score(y_test, y_pred_rf))
    rf_AUC.append(metrics.roc_auc_score(y_test, y_pred_rf))

print('Time: ', time.process_time() - start)
print('Accuracy: ', np.mean(rf_scores['Accuracy']))
print('Precision: ', np.mean(rf_scores['Precision']))
print('Recall: ', np.mean(rf_scores['Recall']))
print('F1-Score: ', 2 * np.mean(rf_scores['Precision']) * np.mean(rf_scores['Recall']) /
      (np.mean(rf_scores['Precision']) + np.mean(rf_scores['Recall'])))
print('AUC:', np.mean(lg_AUC))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
