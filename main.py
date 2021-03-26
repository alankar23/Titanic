#!/usr/bin/env python
# coding: utf-8
 
# %%
import pandas as pd
import seaborn as sns
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from scipy import stats
import pandas_profiling as pp
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# %%


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
#y_Test = pd.read_csv('gender_submission.csv') this is sample submission csv


# %%


train.head()


# %%


test.head()


# %%


train.info()


# %%

test.info()


# %%

train.describe()


# %%

train.shape ,test.shape


# %%


#del train['Ticket']


# %%

train.isnull().sum()


# %%
test.isnull().sum()


# From above, the test data has Age column missing 177 values, Embarked column missing 2 values and Cabin columnn missing 687 values, also the test data has Age column missing 86 values, fare cabin missing 1 value and Cabin column missing 327 values.

# %%


train.Sex.value_counts()


# %%

train.groupby('Sex').Survived.value_counts()


# 2D Scatter Plot

# %%

pp.ProfileReport(train)


# %%

train.groupby(['Survived','Sex'])['Survived'].count()


# From above we can say that women are more likely to survive then men.

# %%


fig, (ax1, ax2) = plt.subplots(1, 2 , figsize =(15,5))
men = train['Survived'][train['Sex']=='male'].value_counts()
female = train['Survived'][train['Sex'] =='female'].value_counts()
label = ['Dead' , 'Survived']
ax1.pie(men, explode=[0,0.2] ,labels =label ,autopct ='%1.1f%%',  shadow= True)
ax1.set_title('Male')
ax2.pie(female , explode=[0,0.2], labels =label[::-1], autopct='%1.1f%%',  shadow=True)
ax2.set_title('Female')
plt.show()


# From the pie chart, it's confermed more females survived then men.

# %%


sns.barplot(x=train.Sex , y=train.Age , hue =train.Survived)

# %%


def bar_chart(feature):
    sns.barplot(x=train['Sex'] , y=train[feature], hue=train['Survived'])


# %%

bar_chart('Pclass')


# %%

sns.barplot(x=train['Sex'] , y=train['Fare'], hue=train['Survived'])


# %%

sns.barplot(x=train['Sex'] , y=train['SibSp'], hue=train['Survived'])


# %%

def bar_graph(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar' ,figsize=(12,6))

# %%

bar_graph('Pclass')

# %%

bar_graph('SibSp')


# %%


bar_graph('Parch')


# %%


bar_graph('Embarked')


# %%


corr_matx = train.corr()


# %%


corr_matx['Survived'].sort_values(ascending = False)


# New Features

# %%


train['title'] = train.Name.str.extract('\, ([A-Z][^ ]*\.)',expand=False) 
test['title'] = test.Name.str.extract('\, ([A-Z][^ ]*\.)',expand=False) 


# %%


train['title'].value_counts()


# %%


train_dict = dict(train['title'].value_counts())
test_dict = dict(test['title'].value_counts())


# %%


title1 = train['title'].tolist()
title2 = test['title'].tolist()
name_list = []
name_list2 = []

for x in title1:
    name= train_dict.get(x)
    if name is None or name <= 20:
        name_list.append('other')
    else:
        name_list.append(x)

for x in title2:
    name = train_dict.get(x)
    if name is None or name <= 20:
        name_list2.append('other')
    else:
        name_list2.append(x)


# %%

train['titles'] = name_list
test['titles'] = name_list2

# %%


train.replace({'Sex':{'male':0, 'female':1 }} ,inplace = True)
test.replace({'Sex':{'male':0 ,'female':1}}, inplace =True)


# %%


train.isnull().sum()


# %%

test.isnull().sum()


# %%

# fill missing age with median age for each title (Mr, Mrs, Miss, Others)
train["Age"].fillna(train.groupby("titles")["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("titles")["Age"].transform("median"), inplace=True)


# %%


train[train['Age'].isnull()]

# %%

test[test['Age'].isnull()]


# Filling missing values in Embarked Column

# %%


fig, (ax1, ax2 ,ax3) = plt.subplots(1,3, figsize= (15,5))
class_1 = train['Embarked'][train['Pclass'] == 1].value_counts()
class_2 = train['Embarked'][train['Pclass'] == 2].value_counts()
class_3 = train['Embarked'][train['Pclass'] == 3].value_counts()
t = train['Embarked'].value_counts()
label = t.index.values
ax1.set_title('1st Class')
ax1.pie(class_1 ,labels = label , autopct='%1.1f%%')
ax2.set_title('2nd Class')
ax2.pie(class_2 ,labels= label ,autopct='%1.1f%%')
ax3.set_title('3rd Class')
ax3.pie(class_3 ,labels = label, autopct='%1.1f%%')
plt.show()


# We can say that from every class most of the people embarked from S port
# %%

train['Embarked'].fillna('S', inplace =True)
test['Embarked'].fillna('S', inplace =True)


# %%

train['FamilySize'] = train['SibSp']+train['Parch']
test['FamilySize'] = test['SibSp']+test['Parch']


# %%


bins = [0,15,30,45,60,100]
labels = [1,2,3,4,5]
Ages_1 = pd.cut(train['Age'],bins , labels = labels)
Ages_2 = pd.cut(test['Age'], bins, labels = labels)


# %%


one_hot_4 = pd.get_dummies(Ages_1)
one_hot_5 = pd.get_dummies(Ages_2)


# %%

train = train.join(one_hot_4)
test = test.join(one_hot_5)


# %%


test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)

# %%

train.columns


# %%
test.columns


# One Hot encoding

# %%


from sklearn.preprocessing import OneHotEncoder


# %%


x1 = train[['Pclass','Sex', 'Embarked', 'titles']]
x2 = test[['Pclass','Sex', 'Embarked', 'titles']]


# %%


one_hot = OneHotEncoder(sparse= False)


# %%

feat = one_hot.fit(x1)
x_1 = feat.transform(x1)
x_2 = feat.transform(x2)


# %%


feat.get_feature_names()


# %%


column1 =  train[['FamilySize', 1, 2, 3, 4, 5]].values
column2 =  test[['FamilySize', 1, 2, 3, 4, 5]].values


# %%


column2.shape


# %%


final = np.column_stack((x_1,column1))
Test = np.column_stack((x_2,column2))
target = train.pop('Survived')


# %%


print(final.shape, Test.shape)


# Mode Testing
# %%


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from prettytable import PrettyTable
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


#  Test Train Split
# %%


xTrain , xTest , yTrain , yTest = train_test_split(final, target , test_size = 0.3)


# %%

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


# ## 1.1 Navie Bayes
# %%


clf = GaussianNB()
scoring = 'accuracy'
score = cross_val_score(clf, xTrain, yTrain, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

# %%

alpha = {}
indx = 1
a = 0
MSE= []
a = 10**-2
while a < 10**2:
    NB = MultinomialNB(alpha = a , class_prior=None , fit_prior=True)
    scores = cross_val_score(NB, xTrain, yTrain, cv =k_fold , scoring='accuracy')
    error = 1 - scores.mean()
    MSE.append(error)
    alpha[error] = a
    a+=0.05
    indx  += 1

opt_alpha = alpha.get(min(MSE))
print('The optimal value of  alpha is {}' .format(round(opt_alpha,3)))


# %%

opt_alpha = 0.21

# %%
# Naive Bayes with Optimal alpha
multiNB = MultinomialNB(alpha =opt_alpha, class_prior=None , fit_prior=True)
multiNB.fit(xTrain, yTrain)
pred = multiNB.predict(xTest)

# %%


acc = accuracy_score(yTest, pred) * 100
prec = precision_score(yTest, pred)  *100
f1 = f1_score(yTest, pred) *100
conf = confusion_matrix(yTest,pred)


# %%

print('The accuracy is {} , Precision is {} and F1 score is {}' .format(acc, prec, f1) , '\n confusion matrix is\n {}' .format(conf)  )

# %%

table_b = PrettyTable()
table_b.add_column('Optimal Alpha' , [round(opt_alpha,5)])
table_b.add_column('Accuracy', [round(acc,5)])
table_b.add_column('Precision', [round(prec,5)])
table_b.add_column('f1 score', [round(f1,5)])
print(table_b)


# %%


Labels = ['Dead' , 'Survived']
plt.title('Confusion matrix of the classifier')
sns.heatmap(conf, annot=True, cmap='RdBu', fmt=".3f", xticklabels=Labels, yticklabels=Labels)
plt.xlabel('Predicted Class')
plt.ylabel('Original Class')
plt.show()

# %%


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": pred
    })


# %%


submission['Survived'][5]


# %%

submission['Survived'].value_counts()

# %%

yTest.value_counts()


# %%


sub = pd.read_csv('submission.csv')


# %%

submission.to_csv('submission.csv', index = False)


# Logistic Reggression
# %%

c_dist =  {'C':stats.uniform(10**-3, 10**2)}
clf = RandomizedSearchCV(LogisticRegression(penalty='l1', class_weight = 'balanced'), c_dist, n_iter=1000, scoring= 'accuracy', cv =k_fold, return_train_score = True)


# %%

alpha = {}
indx = 1
a = 0
MSE= []
a = 10**-2
while a < 10**2:
    clf = LogisticRegression(C=a, penalty='l1', class_weight = 'balanced', solver = 'liblinear' , max_iter =500)
    scores = cross_val_score(clf, xTrain, yTrain, cv =k_fold , scoring='accuracy')
    error = 1 - scores.mean()
    MSE.append(error)
    alpha[error] = a
    a+=0.05
    indx  += 1

opt_alpha = alpha.get(min(MSE))
print('The optimal value of  alpha is {}' .format(round(opt_alpha,3)))

# %%

clf.fit(xTrain ,yTrain)
print(clf.best_estimator_)
print(clf.score(xTrain, yTrain))


# %%


clf = LogisticRegression(C=opt_alpha, penalty = 'l1', class_weight = 'balanced', solver = 'liblinear' , max_iter =500)
clf.fit(xTrain, yTrain)
pred= clf.predict(xTest)


# %%

results = clf.cv_results_ 
result_ = pd.DataFrame.from_dict(results)
result_.head()
# %%


result_ = result_.sort_values('param_C')
test_score = list(result_.mean_test_score)
train_score = list(result_.mean_train_score)
param = list(result_.param_C)

# %%


plt.title('Train AUC vs Test AUC')
plt.xlabel('Alpha Values')
plt.ylabel('AUC score')
plt.plot(param,test_score)
plt.plot(param, train_score)
plt.legend(['Test AUC', 'Train AUC'])
plt.show()


# %%

opt_alpha=2.017451013883791 #3.0489778895550868
log_r =LogisticRegression(C = opt_alpha, penalty='l1', class_weight = 'balanced')
log_r.fit(xTrain, yTrain)
pred= log_r.predict(xTest)


# %%


accMulti_bow= round(accuracy_score(yTest, pred) * 100, 3)
precMulti_bow = round(precision_score(yTest, pred)  *100 ,3)
f1Multi_bow = round(f1_score(yTest, pred) *100,3)
confMulti_bow = confusion_matrix(yTest,pred)


# %%

table_b = PrettyTable()
table_b.add_column('Optimal Alpha' , [round(opt_alpha,5)])
table_b.add_column('Accuracy', [accMulti_bow])
table_b.add_column('Precision', [precMulti_bow])
table_b.add_column('f1 score', [f1Multi_bow])
print(table_b)


# %%
print('Confusion Matrix for Bag of Words with L1 regularizer')
Labels = ['Negative' , 'Positive'] 
plt.figure(figsize=(5,4))
sns.heatmap(confMulti_bow, annot=True, cmap='RdBu', fmt=".3f", xticklabels=Labels, yticklabels=Labels)
plt.xlabel('Predicted Class')
plt.ylabel('Original Class')
plt.show()


# %%

import warnings
warnings.filterwarnings("ignore")


# %%


a = np.arange(10**-3,10,0.05)


# ## SVM

# %%


from sklearn.svm import SVC
svc = SVC( class_weight = 'balanced')


# %%


c = np.arange(10**-3,10,0.05)
gamma = np.arange(10**-3,1,0.01)
param_grid ={'C': c, 'gamma': gamma, 'kernel': ['rbf', 'poly','linear']}


# %%



clf = RandomizedSearchCV(svc, param_grid , scoring= 'accuracy', cv =k_fold, return_train_score = True,n_iter = 100, n_jobs=-1 )
clf.fit(xTrain ,yTrain)
print(clf.best_estimator_)
print(clf.score(xTrain, yTrain))


# %%
clf = SVC(C=1.301, cache_size=200, class_weight='balanced', coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.38099999999999995,
    kernel='rbf', max_iter=-1, probability=False, random_state=None,
    shrinking=True, tol=0.001, verbose=False)
clf.fit(xTrain, yTrain)
pred= clf.predict(xTest)

# %%

accMulti_bow= round(accuracy_score(yTest, pred) * 100, 3)
precMulti_bow = round(precision_score(yTest, pred)  *100 ,3)
f1Multi_bow = round(f1_score(yTest, pred) *100,3)
confMulti_bow = confusion_matrix(yTest,pred)


# %%

table_b = PrettyTable()
table_b.add_column('Optimal Alpha' , [round(1.301)])
table_b.add_column('Accuracy', [accMulti_bow])
table_b.add_column('Precision', [precMulti_bow])
table_b.add_column('f1 score', [f1Multi_bow])
print(table_b)


# %%

print('Confusion Matrix for Bag of Words with L1 regularizer')
Labels = ['Negative' , 'Positive'] 
plt.figure(figsize=(5,4))
sns.heatmap(confMulti_bow, annot=True, cmap='RdBu', fmt=".3f", xticklabels=Labels, yticklabels=Labels)
plt.xlabel('Predicted Class')
plt.ylabel('Original Class')
plt.show()


# ## Decision Tree

# %%


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(class_weight = 'balanced')

# %%

# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in range(10,1000,10)]

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree

# Create the random grid
grid = {       'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               }

# %%

clf = RandomizedSearchCV(dt, param_distributions=grid, n_iter=1000, scoring= 'accuracy', cv =k_fold, return_train_score = True)


# %%


clf.fit(xTrain ,yTrain)
print(clf.best_estimator_)
print(clf.score(xTrain, yTrain))

# %%


results = clf.cv_results_ 
result_ = pd.DataFrame.from_dict(results)
result_.head()

# %%


result_ = result_.sort_values('param_max_depth')
test_score = list(result_.mean_test_score)
train_score = list(result_.mean_train_score)
param = list(result_.param_max_depth)

# %%

plt.title('Train Accuracy vs Test Accuracy')
plt.xlabel('Alpha Values')
plt.ylabel('AUC score')
plt.plot(param,test_score)
plt.plot(param, train_score)
plt.legend(['Test AUC', 'Train AUC'])
plt.show()

# %%

table_b = PrettyTable()
#table_b.add_column('Optimal Alpha' , [round(opt_C_1,5)])
table_b.add_column('Accuracy', [accMulti_bow])
table_b.add_column('Precision', [precMulti_bow])
table_b.add_column('f1 score', [f1Multi_bow])
print(table_b)


# %%


dt =DecisionTreeClassifier(class_weight='balanced', criterion='gini', max_depth=670,
                       max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=4, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=None, splitter='best')
dt.fit(xTrain, yTrain)
pred= dt.predict(xTest)


# %%


accMulti_bow= round(accuracy_score(yTest, pred) * 100, 3)
precMulti_bow = round(precision_score(yTest, pred)  *100 ,3)
f1Multi_bow = round(f1_score(yTest, pred) *100,3)
confMulti_bow = confusion_matrix(yTest,pred)


# %%


table_b = PrettyTable()
table_b.add_column('Accuracy', [accMulti_bow])
table_b.add_column('Precision', [precMulti_bow])
table_b.add_column('f1 score', [f1Multi_bow])
print(table_b)


# %%


print('Confusion Matrix for Bag of Words with L1 regularizer')
Labels = ['Negative' , 'Positive'] 
plt.figure(figsize=(5,4))
sns.heatmap(confMulti_bow, annot=True, cmap='RdBu', fmt=".3f", xticklabels=Labels, yticklabels=Labels)
plt.xlabel('Predicted Class')
plt.ylabel('Original Class')
plt.show()


# ## Random Forest

# %%

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(class_weight='balanced')

# %%


param_grid = {'n_estimators':[x for x in range(10,1000)],'criterion':['gini', 'entropy'],'min_samples_split':[2,5,10],'min_samples_leaf':[2,5,10] }


# %%

clf = RandomizedSearchCV(rf, param_grid, n_iter=100, scoring= 'accuracy', cv =k_fold, return_train_score = True)
clf.fit(xTrain,yTrain)
print(clf.best_estimator_)
print(clf.score(xTrain,yTrain))

# %%


rf = RandomForestClassifier(bootstrap=True, class_weight='balanced',
                       criterion='entropy', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, min_impurity_decrease=0.0,
                       min_impurity_split=None, min_samples_leaf=2,
                       min_samples_split=10, min_weight_fraction_leaf=0.0,
                       n_estimators=595, n_jobs=None, oob_score=False,
                       random_state=None, verbose=0, warm_start=False)


# %%


rf.fit(xTrain,yTrain)
pred = rf.predict(xTest)

# %%


accuracy = round(accuracy_score(yTest,pred)*100,3)
precision = round(precision_score(yTest,pred)*100,3)
F1_score = round(f1_score(yTest,pred)*100,3)
roc_auc = round(roc_auc_score(yTest,pred)*100,3)
conf = confusion_matrix(yTest,pred)

# %%


table = PrettyTable()
table.add_column('Number of trees',['595'])
table.add_column('Roc Auc score',[roc_auc])
table.add_column('Accuracy',[accuracy])
table.add_column('Precison',[precision])
table.add_column('F1 Score',[F1_score])
print(table)

# %%

Labels = ['Dead' , 'Survived']
plt.title('Confusion matrix of the classifier')
sns.heatmap(conf, annot=True, cmap='RdBu', fmt=".3f", xticklabels=Labels, yticklabels=Labels)
plt.xlabel('Predicted Class')
plt.ylabel('Original Class')
plt.show()




