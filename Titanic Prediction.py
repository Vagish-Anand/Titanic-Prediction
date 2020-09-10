#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# In[2]:


os.getcwd()


# In[3]:


#Reading test & train data 
train=pd.read_csv("Desktop/titanic/train.csv")
test=pd.read_csv("Desktop/titanic/test.csv")


# In[4]:


train.head()


# In[5]:


train.info()


# In[6]:


survived = 'survived'
not_survived = 'not survived'
fig, axes = plt.subplots(nrows= 1, ncols = 2, figsize = (15,8))
women = train[train['Sex']== 'female']
men = train[train['Sex']== 'male']

ax = sns.distplot(women[women['Survived']==1].Age.dropna(),bins=18, label = survived, ax=axes[0],kde=False)
ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax= axes[0], kde= False)
ax.legend()
ax.set_title("Female")

ax = sns.distplot(men[men['Survived']==1].Age.dropna(),bins=18, label = survived, ax=axes[1],kde=False)
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax= axes[1], kde= False)
ax.legend()
ax.set_title("Male")


# In[7]:


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize= (13,5))
sns.boxplot(x="Age", data=train, orient='v', ax=ax1)
ax1.set_xlabel('People Age', fontsize=15)
ax1.set_ylabel('Age', fontsize=15)
ax1.set_title('Age Distribution', fontsize=15)
ax1.tick_params(labelsize=15)

sns.distplot(train['Age'], ax=ax2)
sns.despine(ax=ax2)
ax2.set_xlabel('Age',fontsize=15)
ax2.set_ylabel('occurance',fontsize=15)
ax2.set_title('Age x Occurance', fontsize=15)
ax2.tick_params(labelsize=15)

plt.subplots_adjust(wspace=0.5)
plt.tight_layout()


# In[8]:


FacetGrid= sns.FacetGrid(train, row='Embarked', height=4.5, aspect=1.6)
FacetGrid.map(sns.pointplot, 'Pclass','Survived', 'Sex', palette=None, hue_order=None)
FacetGrid.add_legend()


# In[9]:


sns.barplot(x='Pclass',y='Survived', data=train )


# In[10]:


grid=sns.FacetGrid(train,col='Survived', row= 'Pclass', size=2.2, aspect = 1.6)
grid.map(plt.hist,'Age', bins= 20)
grid.add_legend()


# In[11]:


data=[train,test]
for dataset in data:
    dataset['relatives']=dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives']>0, 'not_alone']=0
    dataset.loc[dataset['relatives']==0, 'not_alone']=1
    dataset['not_alone']=dataset['not_alone'].astype(int)
train['not_alone'].value_counts()


# In[12]:


axes=sns.factorplot('relatives', 'Survived', data= train, aspect= 3)


# In[13]:


train=train.drop(['PassengerId'], axis=1)
test=test.drop(['PassengerId'], axis=1)


# In[14]:


train.head()


# In[15]:


test.head()


# In[16]:


def Check_Duplicate(df):
    duplicate_row=df.duplicated().sum()
    null_values=df.isnull().sum()
    Total_null_values=sum(null_values)
    if (duplicate_row>0):
        print("Please remove duplicate Rows=", duplicate_row)
    elif(Total_null_values>0):
            print("Please deal with Missing Values=", null_values)
    else:
        print(duplicate_row, "duplicated with nullvalues", Total_null_values, "in the dataframe")


# In[17]:


Check_Duplicate(train)


# In[18]:


Check_Duplicate(test)


# In[19]:


def deal_with_outlier(column,df_name,col_name):
    Q1=column.quantile(q=0.25)
    Q2=column.quantile(q=0.50)
    Q3=column.quantile(q=0.75)
    Q4=column.quantile(q=1.00)
    print("1^quantile:", Q1)
    print("2^quantile:", Q2)
    print("3^quantile:", Q3)
    print("4^quantile:", Q4)
    
    IQR= Q3-Q1
    Lower=Q1-1.5*IQR
    Upper=Q3+1.5*IQR
    print("Lower Bound=",Lower)
    print("Upper Bound=", Upper)
    out=column.quantile(q=0.75)+1.5*(column.quantile(q=0.75)-column.quantile(q = 0.25))
    print ('above', out, "are outliers")
    
    #% of outliers in upper
    print("number of outliers in upper:", df_name[column>Upper][col_name].count())
    print("number of clients:",len(df_name))
    print("Outliers are :", round(df_name[column>Upper][col_name].count()*100/len(df_name),2), '%')
    
    #% of outliers on lower
    print("number of outliers in lower:", df_name[column>Lower][col_name].count())
    print("number of clients:",len(df_name))
    print("Outliers are :", round(df_name[column>Lower][col_name].count()*100/len(df_name),2), '%')

        
    #Flooring
    df_name.loc[column<(Q1-1.5*IQR), col_name]=column.quantile(0.05)
    #Capping
    df_name.loc[column>(Q3+1.5*IQR), col_name]=column.quantile(0.95)
    
    Boxplot=df_name.boxplot(column=[col_name])
    
    
    #After Deal 
    
    #% of outlier in upper
    print("numner of outliers in upper after deal:", df_name[column>Upper][col_name].count())
    print("number of clients:",len(df_name))
    print("Outliers after deal are :", round(df_name[column>Upper][col_name].count()*100/len(df_name),2), '%')

    #% of outlier in lower
    print("number of outliers in lower after deal:", df_name[column>Lower][col_name].count())
    print("number of clients:",len(df_name))
    print("Outliers after deal are :", round(df_name[column>Lower][col_name].count()*100/len(df_name),2), '%')
    
    
    return Boxplot


# In[20]:


train.boxplot(column=['Age'])
plt.title("We have seen outliers in AGE Variable")


# In[21]:


deal_with_outlier(train['Age'],train,'Age')


# In[22]:


print('MEAN:', round(train['Age'].mean(), 1))
print('STD :', round(train['Age'].std(), 1))
print('Median',round(train['Age'].median(),1))
# I thing the best way to give a precisly insight abou dispersion is using the CV (coefficient variation) (STD/MEAN)*100
#    cv < 15%, low dispersion
#    cv > 30%, high dispersion
print('CV  :',round(train['Age'].std()*100/train['Age'].mean(), 1), ', High middle dispersion')


# In[23]:


test.boxplot(column=['Age'])
plt.title("We have seen outliers in AGE variable(test dataset)")


# In[24]:


deal_with_outlier(test['Age'],test,'Age')


# In[25]:


train['Age']=train['Age'].fillna(train['Age'].mean())
test['Age']=test['Age'].fillna(test['Age'].mean())


# In[26]:


Check_Duplicate(train)


# In[27]:


Check_Duplicate(test)


# In[28]:


train["Cabin"].mode()


# In[29]:


train=train.drop(["Cabin"],axis=1)
test=test.drop(["Cabin"], axis=1)


# In[30]:


Check_Duplicate(train)


# In[31]:


train=train.dropna()


# In[32]:


train=train.reset_index(drop=True)


# In[33]:


Check_Duplicate(train)


# In[34]:


Check_Duplicate(test)


# In[35]:


train.boxplot(column=['Fare'])
plt.title("We have seen outliers in FARE variable")


# In[36]:


deal_with_outlier(train['Fare'],train,'Fare')


# In[37]:


deal_with_outlier(test['Fare'],test,'Fare')


# In[38]:


train['Fare']=train['Fare'].fillna(train['Fare'].mean())
test['Fare']=test['Fare'].fillna(test['Fare'].mean())


# In[39]:


Check_Duplicate(train)


# In[40]:


Check_Duplicate(test)


# In[41]:


train.info()


# In[42]:


train.head()


# In[43]:


test.info()


# In[44]:


test.head()


# In[45]:


data=[train,test]

for dataset in data:
    dataset['Fare']=dataset['Fare'].astype(int)


# In[46]:


train.info()


# In[47]:


test.info()


# In[48]:


data = [train, test]
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in data:
    # extract titles
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    # replace titles with a more common title or as Rare
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    # convert titles into numbers
    dataset['Title'] = dataset['Title'].map(titles)
    # filling NaN with 0, to get safe
    dataset['Title'] = dataset['Title'].fillna(0)
train= train.drop(['Name'], axis=1)
test= test.drop(['Name'], axis=1)


# In[49]:


genders={"male":0,"female":1}

data=[train,test]

for dataset in data:
    dataset['Sex']=dataset['Sex'].map(genders)


# In[50]:


train['Ticket'].describe()


# In[51]:


train=train.drop(['Ticket'], axis=1)
test=test.drop(['Ticket'], axis=1)


# In[52]:


ports={"S":0, "C":1, "Q":2}
data=[train,test]

for dataset in data:
    dataset['Embarked']=dataset['Embarked'].map(ports)


# In[53]:


data=[train,test]
for dataset in data:
    dataset['Age']=dataset['Age'].astype(int)
    dataset.loc[dataset['Age']<=11,'Age']=0
    dataset.loc[(dataset['Age']>11) & (dataset['Age']<=18), 'Age']=1
    dataset.loc[(dataset['Age']>18) & (dataset['Age']<=22), 'Age']=2
    dataset.loc[(dataset['Age']>22) & (dataset['Age']<=27), 'Age']=3
    dataset.loc[(dataset['Age']>27) & (dataset['Age']<=33), 'Age']=4
    dataset.loc[(dataset['Age']>33) & (dataset['Age']<=40), 'Age']=5
    dataset.loc[(dataset['Age']>40) & (dataset['Age']<=66), 'Age']=6
    dataset.loc[dataset['Age']>66,'Age']=6


# In[54]:


train.head()


# In[55]:


test.head()


# In[56]:


y=train.Survived


# In[57]:


y.head()


# In[58]:


X=train.drop(['Survived'], axis=1)


# In[59]:


X.head()


# In[60]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.33, random_state=42)


# In[61]:


print('Train Dataset', X_train.shape, y_train.shape)
print('Test Dataset', X_test.shape, y_test.shape)


# In[73]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from sklearn.model_selection import KFold


# In[74]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train_f= sc_X.fit_transform(X_train)
X_test_f=sc_X.transform(X_test)
K_fold=KFold(n_splits=10, shuffle=True, random_state=0)


# In[75]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
logpred= logmodel.predict(X_test)


cnf_matrix= confusion_matrix(y_test,logpred)
print('Confusion matrix on Train Dataset:')
print(confusion_matrix(y_test,logpred))
print(round(accuracy_score(y_test,logpred),2)*100)


# In[76]:


LOGCV= (cross_val_score(logmodel, X_train, y_train, cv=K_fold, n_jobs=1, scoring='accuracy').mean())
print("stratified cross validation accuracy is", LOGCV)


# In[77]:


from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        
    print(cm)
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, logpred))
print(round(accuracy_score(y_test, logpred),2)*100)


# In[78]:


plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['YES=1','NO=0'],normalize= False,  title='Confusion matrix')


# In[83]:


logpred = logmodel.predict(test)


# In[84]:


get_ipython().system('pip install pydotplus')


# In[86]:


import numpy as np, pandas as pd, matplotlib.pyplot as plt, pydotplus
from sklearn import tree, metrics, model_selection, preprocessing
from IPython.display import Image, display
from sklearn.tree import export_graphviz


# In[88]:


dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
dtree.fit(X_train,y_train)


# In[89]:


y_pred_test = dtree.predict(X_test)
y_pred_test


# In[90]:


accuracy=metrics.accuracy_score(y_test,y_pred_test)
print('Accuracy : {:.2f}'.format(accuracy))


# In[91]:


y_pred = dtree.predict(test)
y_pred


# In[92]:


cv = KFold(n_splits=12, shuffle=True, random_state=0)


# In[93]:


fold_accuracy = []

# titanic_train["Sex"] = encoded_sex

for train_fold, valid_fold in cv.split(X):
    train = X.loc[train_fold] # Extract train data with cv indices
    valid = X.loc[valid_fold] # Extract valid data with cv indices
    
    train_y = y.loc[train_fold]
    valid_y = y.loc[valid_fold]
    
    model = dtree.fit(X = train, 
                           y = train_y)
    valid_acc = model.score(X = valid, 
                            y = valid_y)
    fold_accuracy.append(valid_acc)    

print("Accuracy per fold: ", fold_accuracy, "\n")
print("Average accuracy: ", sum(fold_accuracy)/len(fold_accuracy))


# In[94]:


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
# from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.model_selection import train_test_split, cross_val_score


# In[96]:


from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, BaggingRegressor, RandomForestRegressor

# we can achieve the above two tasks using the following codes
# Bagging: using all features
rfc1 = RandomForestClassifier(max_features=8, random_state=1)
rfc1.fit(X_train, y_train)
pred1 = rfc1.predict(X_test)
print(roc_auc_score(y_test, pred1))

# play around with the setting for max_features
rfc2 = RandomForestClassifier(max_features=6, random_state=1)
rfc2.fit(X_train, y_train)
pred2 = rfc2.predict(X_test)
print(roc_auc_score(y_test, pred2))

pred_x = rfc1.predict(X_test)
print(roc_auc_score(y_test, pred_x))


# In[97]:


y_pred_final = dtree.predict(test)


# In[98]:


submission=pd.read_csv('Desktop/titanic/test.csv')


# In[99]:


submission['Survived']=y_pred_final


# In[100]:


submission= submission.drop(['Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'], axis=1)


# In[101]:


submission.head()


# In[102]:


submission.to_csv('submission',index=False)

