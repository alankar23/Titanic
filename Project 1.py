#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
#y_Test = pd.read_csv('gender_submission.csv') this is sample submission csv


# In[3]:


train.head()


# In[4]:


test.head()


# In[5]:


train.info()


# In[6]:


test.info()


# In[7]:


train.describe()


# In[8]:


train.shape ,test.shape


# In[9]:


#del train['Ticket']


# In[10]:


train.isnull().sum()


# In[11]:


test.isnull().sum()


# From above, the test data has Age column missing 177 values, Embarked column missing 2 values and Cabin columnn missing 687 values, also the test data has Age column missing 86 values, fare cabin missing 1 value and Cabin column missing 327 values.

# In[12]:


train.Sex.value_counts()


# In[13]:


train.groupby('Sex').Survived.value_counts()


# # 2D Scatter Plot

# In[14]:


pp.ProfileReport(train)


# In[15]:


train.groupby(['Survived','Sex'])['Survived'].count()


# From above we can say that women are more likely to survive then men.

# In[16]:


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

# In[17]:


sns.barplot(x=train.Sex , y=train.Age , hue =train.Survived)


# In[18]:


def bar_chart(feature):
    sns.barplot(x=train['Sex'] , y=train[feature], hue=train['Survived'])


# In[19]:


bar_chart('Pclass')


# In[20]:


sns.barplot(x=train['Sex'] , y=train['Fare'], hue=train['Survived'])


# In[21]:


sns.barplot(x=train['Sex'] , y=train['SibSp'], hue=train['Survived'])


# In[22]:


def bar_graph(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar' ,figsize=(12,6))


# In[23]:


bar_graph('Pclass')


# In[24]:


bar_graph('SibSp')


# In[25]:


bar_graph('Parch')


# In[26]:


bar_graph('Embarked')


# In[27]:


corr_matx = train.corr()


# In[28]:


corr_matx['Survived'].sort_values(ascending = False)


# ## New Features

# In[29]:


train['title'] = train.Name.str.extract('\, ([A-Z][^ ]*\.)',expand=False) 
test['title'] = test.Name.str.extract('\, ([A-Z][^ ]*\.)',expand=False) 


# In[30]:


train['title'].value_counts()


# In[31]:


train_dict = dict(train['title'].value_counts())
test_dict = dict(test['title'].value_counts())


# In[32]:


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


# In[33]:


train['titles'] = name_list
test['titles'] = name_list2


# In[34]:


train.replace({'Sex':{'male':0, 'female':1 }} ,inplace = True)
test.replace({'Sex':{'male':0 ,'female':1}}, inplace =True)


# In[35]:


train.isnull().sum()


# In[36]:


test.isnull().sum()


# In[37]:


# fill missing age with median age for each title (Mr, Mrs, Miss, Others)
train["Age"].fillna(train.groupby("titles")["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("titles")["Age"].transform("median"), inplace=True)


# In[38]:


train[train['Age'].isnull()]


# In[39]:


test[test['Age'].isnull()]


# # Filling missing values in Embarked Column

# In[40]:


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

# In[41]:


train['Embarked'].fillna('S', inplace =True)
test['Embarked'].fillna('S', inplace =True)


# In[42]:


train['FamilySize'] = train['SibSp']+train['Parch']
test['FamilySize'] = test['SibSp']+test['Parch']


# In[43]:


bins = [0,15,30,45,60,100]
labels = [1,2,3,4,5]
Ages_1 = pd.cut(train['Age'],bins , labels = labels)
Ages_2 = pd.cut(test['Age'], bins, labels = labels)


# In[44]:


one_hot_4 = pd.get_dummies(Ages_1)
one_hot_5 = pd.get_dummies(Ages_2)


# In[45]:


train = train.join(one_hot_4)
test = test.join(one_hot_5)


# In[46]:


test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)


# In[47]:


train.columns


# In[48]:


test.columns


# # One Hot encoding

# In[49]:


from sklearn.preprocessing import OneHotEncoder


# In[50]:


x1 = train[['Pclass','Sex', 'Embarked', 'titles']]
x2 = test[['Pclass','Sex', 'Embarked', 'titles']]


# In[51]:


one_hot = OneHotEncoder(sparse= False)


# In[52]:


feat = one_hot.fit(x1)
x_1 = feat.transform(x1)
x_2 = feat.transform(x2)


# In[53]:


feat.get_feature_names()


# In[54]:


column1 =  train[['FamilySize', 1, 2, 3, 4, 5]].values
column2 =  test[['FamilySize', 1, 2, 3, 4, 5]].values


# In[55]:


column2.shape


# In[56]:


final = np.column_stack((x_1,column1))
Test = np.column_stack((x_2,column2))
target = train.pop('Survived')


# In[57]:


print(final.shape, Test.shape)


# ## Mode Testing

# In[58]:


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


# ## Test Train Split

# In[59]:


xTrain , xTest , yTrain , yTest = train_test_split(final, target , test_size = 0.3)


# In[60]:


k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


# ## 1.1 Navie Bayes

# In[88]:


clf = GaussianNB()
scoring = 'accuracy'
score = cross_val_score(clf, xTrain, yTrain, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[68]:


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


# In[30]:


opt_alpha = 0.21


# In[69]:


# Naive Bayes with Optimal alpha
multiNB = MultinomialNB(alpha =opt_alpha, class_prior=None , fit_prior=True)
multiNB.fit(xTrain, yTrain)
pred = multiNB.predict(xTest)


# In[70]:


acc = accuracy_score(yTest, pred) * 100
prec = precision_score(yTest, pred)  *100
f1 = f1_score(yTest, pred) *100
conf = confusion_matrix(yTest,pred)


# In[71]:


print('The accuracy is {} , Precision is {} and F1 score is {}' .format(acc, prec, f1) , '\n confusion matrix is\n {}' .format(conf)  )


# In[72]:


table_b = PrettyTable()
table_b.add_column('Optimal Alpha' , [round(opt_alpha,5)])
table_b.add_column('Accuracy', [round(acc,5)])
table_b.add_column('Precision', [round(prec,5)])
table_b.add_column('f1 score', [round(f1,5)])
print(table_b)


# In[73]:


Labels = ['Dead' , 'Survived']
plt.title('Confusion matrix of the classifier')
sns.heatmap(conf, annot=True, cmap='RdBu', fmt=".3f", xticklabels=Labels, yticklabels=Labels)
plt.xlabel('Predicted Class')
plt.ylabel('Original Class')
plt.show()


# In[34]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": pred
    })


# In[43]:


submission['Survived'][5]


# In[36]:


submission['Survived'].value_counts()


# In[38]:


yTest.value_counts()


# In[55]:


sub = pd.read_csv('submission.csv')


# In[99]:


submission.to_csv('submission.csv', index = False)


# ## Logistic Reggression

# In[53]:


c_dist =  {'C':stats.uniform(10**-3, 10**2)}
clf = RandomizedSearchCV(LogisticRegression(penalty='l1', class_weight = 'balanced'), c_dist, n_iter=1000, scoring= 'accuracy', cv =k_fold, return_train_score = True)


# In[76]:


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


# In[77]:


clf.fit(xTrain ,yTrain)
print(clf.best_estimator_)
print(clf.score(xTrain, yTrain))


# In[79]:


clf = LogisticRegression(C=opt_alpha, penalty = 'l1', class_weight = 'balanced', solver = 'liblinear' , max_iter =500)
clf.fit(xTrain, yTrain)
pred= clf.predict(xTest)


# In[55]:


results = clf.cv_results_ 
result_ = pd.DataFrame.from_dict(results)
result_.head()


# In[80]:


result_ = result_.sort_values('param_C')
test_score = list(result_.mean_test_score)
train_score = list(result_.mean_train_score)
param = list(result_.param_C)


# In[57]:


plt.title('Train AUC vs Test AUC')
plt.xlabel('Alpha Values')
plt.ylabel('AUC score')
plt.plot(param,test_score)
plt.plot(param, train_score)
plt.legend(['Test AUC', 'Train AUC'])
plt.show()


# In[61]:


opt_alpha=2.017451013883791 #3.0489778895550868
log_r =LogisticRegression(C = opt_alpha, penalty='l1', class_weight = 'balanced')
log_r.fit(xTrain, yTrain)
pred= log_r.predict(xTest)


# In[81]:


accMulti_bow= round(accuracy_score(yTest, pred) * 100, 3)
precMulti_bow = round(precision_score(yTest, pred)  *100 ,3)
f1Multi_bow = round(f1_score(yTest, pred) *100,3)
confMulti_bow = confusion_matrix(yTest,pred)


# In[83]:


table_b = PrettyTable()
table_b.add_column('Optimal Alpha' , [round(opt_alpha,5)])
table_b.add_column('Accuracy', [accMulti_bow])
table_b.add_column('Precision', [precMulti_bow])
table_b.add_column('f1 score', [f1Multi_bow])
print(table_b)


# In[84]:


print('Confusion Matrix for Bag of Words with L1 regularizer')
Labels = ['Negative' , 'Positive'] 
plt.figure(figsize=(5,4))
sns.heatmap(confMulti_bow, annot=True, cmap='RdBu', fmt=".3f", xticklabels=Labels, yticklabels=Labels)
plt.xlabel('Predicted Class')
plt.ylabel('Original Class')
plt.show()


# In[52]:


import warnings
warnings.filterwarnings("ignore")


# In[114]:


a = np.arange(10**-3,10,0.05)


# ## SVM

# In[87]:


from sklearn.svm import SVC
svc = SVC( class_weight = 'balanced')


# In[85]:


c = np.arange(10**-3,10,0.05)
gamma = np.arange(10**-3,1,0.01)
param_grid ={'C': c, 'gamma': gamma, 'kernel': ['rbf', 'poly','linear']}


# In[88]:



clf = RandomizedSearchCV(svc, param_grid , scoring= 'accuracy', cv =k_fold, return_train_score = True,n_iter = 100, n_jobs=-1 )
clf.fit(xTrain ,yTrain)
print(clf.best_estimator_)
print(clf.score(xTrain, yTrain))


# In[89]:


clf = SVC(C=1.301, cache_size=200, class_weight='balanced', coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.38099999999999995,
    kernel='rbf', max_iter=-1, probability=False, random_state=None,
    shrinking=True, tol=0.001, verbose=False)
clf.fit(xTrain, yTrain)
pred= clf.predict(xTest)


# In[90]:


accMulti_bow= round(accuracy_score(yTest, pred) * 100, 3)
precMulti_bow = round(precision_score(yTest, pred)  *100 ,3)
f1Multi_bow = round(f1_score(yTest, pred) *100,3)
confMulti_bow = confusion_matrix(yTest,pred)


# In[91]:


table_b = PrettyTable()
table_b.add_column('Optimal Alpha' , [round(1.301)])
table_b.add_column('Accuracy', [accMulti_bow])
table_b.add_column('Precision', [precMulti_bow])
table_b.add_column('f1 score', [f1Multi_bow])
print(table_b)


# In[92]:


print('Confusion Matrix for Bag of Words with L1 regularizer')
Labels = ['Negative' , 'Positive'] 
plt.figure(figsize=(5,4))
sns.heatmap(confMulti_bow, annot=True, cmap='RdBu', fmt=".3f", xticklabels=Labels, yticklabels=Labels)
plt.xlabel('Predicted Class')
plt.ylabel('Original Class')
plt.show()


# ## Decision Tree

# In[93]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(class_weight = 'balanced')


# In[94]:


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


# In[95]:


clf = RandomizedSearchCV(dt, param_distributions=grid, n_iter=1000, scoring= 'accuracy', cv =k_fold, return_train_score = True)


# In[96]:


clf.fit(xTrain ,yTrain)
print(clf.best_estimator_)
print(clf.score(xTrain, yTrain))


# In[97]:


results = clf.cv_results_ 
result_ = pd.DataFrame.from_dict(results)
result_.head()


# In[98]:


result_ = result_.sort_values('param_max_depth')
test_score = list(result_.mean_test_score)
train_score = list(result_.mean_train_score)
param = list(result_.param_max_depth)


# In[102]:


plt.title('Train Accuracy vs Test Accuracy')
plt.xlabel('Alpha Values')
plt.ylabel('AUC score')
plt.plot(param,test_score)
plt.plot(param, train_score)
plt.legend(['Test AUC', 'Train AUC'])
plt.show()


# In[103]:


table_b = PrettyTable()
#table_b.add_column('Optimal Alpha' , [round(opt_C_1,5)])
table_b.add_column('Accuracy', [accMulti_bow])
table_b.add_column('Precision', [precMulti_bow])
table_b.add_column('f1 score', [f1Multi_bow])
print(table_b)


# In[104]:


dt =DecisionTreeClassifier(class_weight='balanced', criterion='gini', max_depth=670,
                       max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=4, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=None, splitter='best')
dt.fit(xTrain, yTrain)
pred= dt.predict(xTest)


# In[105]:


accMulti_bow= round(accuracy_score(yTest, pred) * 100, 3)
precMulti_bow = round(precision_score(yTest, pred)  *100 ,3)
f1Multi_bow = round(f1_score(yTest, pred) *100,3)
confMulti_bow = confusion_matrix(yTest,pred)


# In[106]:


table_b = PrettyTable()
table_b.add_column('Accuracy', [accMulti_bow])
table_b.add_column('Precision', [precMulti_bow])
table_b.add_column('f1 score', [f1Multi_bow])
print(table_b)


# In[107]:


print('Confusion Matrix for Bag of Words with L1 regularizer')
Labels = ['Negative' , 'Positive'] 
plt.figure(figsize=(5,4))
sns.heatmap(confMulti_bow, annot=True, cmap='RdBu', fmt=".3f", xticklabels=Labels, yticklabels=Labels)
plt.xlabel('Predicted Class')
plt.ylabel('Original Class')
plt.show()


# ## Random Forest

# In[40]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(class_weight='balanced')


# In[39]:


param_grid = {'n_estimators':[x for x in range(10,1000)],'criterion':['gini', 'entropy'],'min_samples_split':[2,5,10],'min_samples_leaf':[2,5,10] }


# In[42]:


clf = RandomizedSearchCV(rf, param_grid, n_iter=100, scoring= 'accuracy', cv =k_fold, return_train_score = True)
clf.fit(xTrain,yTrain)
print(clf.best_estimator_)
print(clf.score(xTrain,yTrain))


# In[43]:


rf = RandomForestClassifier(bootstrap=True, class_weight='balanced',
                       criterion='entropy', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, min_impurity_decrease=0.0,
                       min_impurity_split=None, min_samples_leaf=2,
                       min_samples_split=10, min_weight_fraction_leaf=0.0,
                       n_estimators=595, n_jobs=None, oob_score=False,
                       random_state=None, verbose=0, warm_start=False)


# In[44]:


rf.fit(xTrain,yTrain)
pred = rf.predict(xTest)


# In[47]:


accuracy = round(accuracy_score(yTest,pred)*100,3)
precision = round(precision_score(yTest,pred)*100,3)
F1_score = round(f1_score(yTest,pred)*100,3)
roc_auc = round(roc_auc_score(yTest,pred)*100,3)
conf = confusion_matrix(yTest,pred)


# In[49]:


table = PrettyTable()
table.add_column('Number of trees',['595'])
table.add_column('Roc Auc score',[roc_auc])
table.add_column('Accuracy',[accuracy])
table.add_column('Precison',[precision])
table.add_column('F1 Score',[F1_score])
print(table)


# In[50]:


Labels = ['Dead' , 'Survived']
plt.title('Confusion matrix of the classifier')
sns.heatmap(conf, annot=True, cmap='RdBu', fmt=".3f", xticklabels=Labels, yticklabels=Labels)
plt.xlabel('Predicted Class')
plt.ylabel('Original Class')
plt.show()


# # XGBoost

# In[61]:


#from xgboost import XGBClassifier as xgb
#from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier 


# In[62]:


model = GradientBoostingClassifier()
model.fit(xTrain, yTrain)


# In[ ]:


model

