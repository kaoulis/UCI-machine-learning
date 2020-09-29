import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Loading data
data_array = np.genfromtxt("abalone19.dat", dtype='str', delimiter=",", skip_header=13)
df = pd.DataFrame(data=data_array)
df[[1, 2, 3, 4, 5, 6, 7]] = df[[1, 2, 3, 4, 5, 6, 7]].apply(pd.to_numeric)

#############################
# Pre-processing #########
#############################
# Encoding
df.loc[df[8] == ' positive', 8] = 1
df.loc[df[8] == 'negative', 8] = 0
df = pd.get_dummies(df)
# seperate categorical from continuous for scaling
num_df = df.iloc[:, 0:7]
cat_df = df.iloc[:, 8:11]
labels = df[8]
num_df = pd.DataFrame(data=StandardScaler().fit_transform(num_df))
# After transformation merge again
scaled_df = num_df.merge(cat_df, left_index=True, right_index=True)

# Imbalanced data
print((df[8][df[8] == 1]).count())
print((df[8][df[8] == 0]).count())

features = scaled_df
# basic data split
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=7)

# logistic regression to check the metrics of imbanced data
lg = LogisticRegression().fit(x_train, y_train)
y_pred_lr = lg.predict(x_test)
lg_cm = metrics.confusion_matrix(y_test, y_pred_lr)
lg_acc = metrics.accuracy_score(y_test, y_pred_lr)
lg_pre = metrics.precision_score(y_test, y_pred_lr)
lg_rec = metrics.recall_score(y_test, y_pred_lr)
lg_f1 = metrics.f1_score(y_test, y_pred_lr)

print("Confusion Matrix: ", lg_cm)
print("Accuracy: ", lg_acc)
print("Precision: ", lg_pre)
print("Recall: ", lg_rec)
print("F1-score: ", lg_f1)

###################
# Oversampling ####
###################
# merge training data back
x = pd.concat([x_train, y_train], axis=1)
print(list(x.columns.values))

# separate minority and majority classes
negative = x.loc[x[8] == 0]
positive = x.loc[x[8] == 1]

# oversample minority
over_positive = resample(positive, replace=True, n_samples=len(negative), random_state=7)

# combine back minority and majority classes. Now the have equal number of instances.
oversample = pd.concat([negative, over_positive])

x_train_over = oversample.drop(8, axis=1)
y_train_over = oversample[8]

# check oversampling with logistic regression
lg = LogisticRegression().fit(x_train_over, y_train_over)
y_pred_lr = lg.predict(x_test)
lg_cm = metrics.confusion_matrix(y_test, y_pred_lr)
lg_acc = metrics.accuracy_score(y_test, y_pred_lr)
lg_pre = metrics.precision_score(y_test, y_pred_lr)
lg_rec = metrics.recall_score(y_test, y_pred_lr)
lg_f1 = metrics.f1_score(y_test, y_pred_lr)
metrics.plot_confusion_matrix(lg, x_test, y_test)
plt.show()

print("Confusion Matrix: ", lg_cm)
print("Accuracy: ", lg_acc)
print("Precision: ", lg_pre)
print("Recall: ", lg_rec)
print("F1-score: ", lg_f1)

###################
# SMOTE ###########
###################
sm = SMOTE(random_state=7)
x_train, y_train = sm.fit_sample(x_train, y_train)

smote = LogisticRegression().fit(x_train, y_train)
y_pred_lr = smote.predict(x_test)
sm_cm = metrics.confusion_matrix(y_test, y_pred_lr)
sm_acc = metrics.accuracy_score(y_test, y_pred_lr)
sm_pre = metrics.precision_score(y_test, y_pred_lr)
sm_rec = metrics.recall_score(y_test, y_pred_lr)
sm_f1 = metrics.f1_score(y_test, y_pred_lr)
print("Confusion Matrix: ", sm_cm)
print("Accuracy: ", sm_acc)
print("Precision: ", sm_pre)
print("Recall: ", sm_rec)
print("F1-score: ", sm_f1)

###########################################
# Classification and Cross Validation #####
###########################################
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
features = features.to_numpy()
labels = labels.to_numpy()

#Logistic regression
lg_scores = {'Accuracy': [], 'Precision': [], 'Recall': []}
lg_AUC = []
lg = LogisticRegression()
cv = KFold(n_splits=10,shuffle=True, random_state=7)
start = time.process_time()
for train_index, test_index in cv.split(features):
    x_train, x_test, y_train, y_test = features[train_index], features[test_index], labels[train_index], labels[test_index]
    x_train, y_train = sm.fit_sample(x_train, y_train)
    lg.fit(x_train, y_train)
    y_pred_lg = lg.predict(x_test)
    lg_scores['Accuracy'].append(metrics.accuracy_score(y_test, y_pred_lg))
    lg_scores['Precision'].append(metrics.precision_score(y_test, y_pred_lg))
    lg_scores['Recall'].append(metrics.recall_score(y_test, y_pred_lg))
    lg_AUC.append(metrics.roc_auc_score(y_test, y_pred_lg))

print('Time: ', time.process_time() - start)
print('Accuracy: ', np.mean(lg_scores['Accuracy']))
print('Precision: ', np.mean(lg_scores['Precision']))
print('Recall: ', np.mean(lg_scores['Recall']))
print('F1-Score: ', 2*np.mean(lg_scores['Precision'])*np.mean(lg_scores['Recall'])/
      (np.mean(lg_scores['Precision']) + np.mean(lg_scores['Recall'])))
print('AUC:', np.mean(lg_AUC))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


#Decision Tree
dc_scores = {'Accuracy': [], 'Precision': [], 'Recall': []}
dc_AUC = []
dc = DecisionTreeClassifier(random_state=7)
cv = KFold(n_splits=10,shuffle=True, random_state=7)
start = time.process_time()
for train_index, test_index in cv.split(features):
    x_train, x_test, y_train, y_test = features[train_index], features[test_index], labels[train_index], labels[
        test_index]
    x_train, y_train = sm.fit_sample(x_train, y_train)
    dc.fit(x_train, y_train)
    y_pred_dc = dc.predict(x_test)
    dc_scores['Accuracy'].append(metrics.accuracy_score(y_test, y_pred_dc))
    dc_scores['Precision'].append(metrics.precision_score(y_test, y_pred_dc))
    dc_scores['Recall'].append(metrics.recall_score(y_test, y_pred_dc))
    dc_AUC.append(metrics.roc_auc_score(y_test, y_pred_dc))

print('Time: ', time.process_time() - start)
print('Accuracy: ', np.mean(dc_scores['Accuracy']))
print('Precision: ', np.mean(dc_scores['Precision']))
print('Recall: ', np.mean(dc_scores['Recall']))
print('F1-Score: ', 2*np.mean(dc_scores['Precision'])*np.mean(dc_scores['Recall'])/
      (np.mean(dc_scores['Precision']) + np.mean(dc_scores['Recall'])))
print('AUC:', np.mean(lg_AUC))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


#Random forest
rf_scores = {'Accuracy': [], 'Precision': [], 'Recall': []}
rf_AUC = []
rf = RandomForestClassifier(random_state=7)
cv = KFold(n_splits=10, shuffle=True, random_state=7)
start = time.process_time()
for train_index, test_index in cv.split(features):
    x_train, x_test, y_train, y_test = features[train_index], features[test_index], labels[train_index], labels[
        test_index]
    x_train, y_train = sm.fit_sample(x_train, y_train)
    rf.fit(x_train, y_train)
    y_pred_rf = rf.predict(x_test)
    rf_scores['Accuracy'].append(metrics.accuracy_score(y_test, y_pred_rf))
    rf_scores['Precision'].append(metrics.precision_score(y_test, y_pred_rf))
    rf_scores['Recall'].append(metrics.recall_score(y_test, y_pred_rf))
    rf_AUC.append(metrics.roc_auc_score(y_test, y_pred_rf))

print('Time: ', time.process_time() - start)
print('Accuracy: ', np.mean(rf_scores['Accuracy']))
print('Precision: ', np.mean(rf_scores['Precision']))
print('Recall: ', np.mean(rf_scores['Recall']))
print('F1-Score: ', 2*np.mean(rf_scores['Precision'])*np.mean(rf_scores['Recall'])/
      (np.mean(rf_scores['Precision']) + np.mean(rf_scores['Recall'])))
print('AUC:', np.mean(lg_AUC))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')




