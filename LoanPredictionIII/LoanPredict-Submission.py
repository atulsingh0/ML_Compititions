
# coding: utf-8

# In[1]:

# import
import pandas as pd
import numpy as np
from collections import defaultdict

from sklearn.preprocessing import Imputer, LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score


# In[2]:

# reading the data
data = pd.read_csv("train.csv")
data[:4]


# In[3]:

#let's remove all the rows which is having nulls
# train = train[~train.isnull()]
data = data.dropna()


# In[4]:

enc = dict()

# Encoding the variable
for x in ['Gender', 'Married', 'Education', 'Dependents', 'Self_Employed', 'Property_Area']:
    enc[x]=LabelEncoder()
    enc[x].fit(data[x])
    print(enc[x].classes_)
    data[x]=enc[x].transform(data[x])
#print(enc)

#fit = data.apply(lambda x: d[x.name].fit_transform(x))

# Inverse the encoded
#fit.apply(lambda x: d[x.name].inverse_transform(x))

# Using the dictionary to label future data
#data.apply(lambda x: d[x.name].transform(x))


# In[5]:

data_id = data.iloc[:, 0]


# In[6]:

# scaling the data
sc = StandardScaler()
data.iloc[:,1:-1] = sc.fit_transform(data.iloc[:,1:-1])


# In[7]:

# Splitting the data into X and y
X = data.iloc[:, 1:-1]
y = data.iloc[:, -1]

print(X[:2])
print(y[:2])


# In[8]:

# Split the data into 25% test and 75% training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# ** Implementing RandomForestClassifier **

# In[9]:

# Create a random forest classifier
clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)

# Train the classifier
clf.fit(X_train, y_train)


# In[10]:

# Print the name and gini importance of each feature
for feature in zip(X_train.columns, clf.feature_importances_):
    print(feature)


# In[11]:

# Create a selector object that will use the random forest classifier to identify
# features that have an importance of more than 0.01
sfm = SelectFromModel(clf, threshold=0.02)


# In[12]:

sfm.fit(X_train, y_train)

# Print the names of the most important features
for feature_list_index in sfm.get_support(indices=True):
    print(X_train.columns[feature_list_index])


# In[13]:

# Transform the data to create a new dataset containing only the most important features
# Note: We have to apply the transform to both the training X and test X data.
X_important_train = sfm.transform(X_train)
X_important_test = sfm.transform(X_test)


# In[14]:

# Create a new random forest classifier for the most important features
clf_important = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)

# Train the new classifier on the new dataset containing the most important features
clf_important.fit(X_important_train, y_train)


# In[15]:

# Apply The Full Featured Classifier To The Test Data
y_pred = clf.predict(X_test)

# View The Accuracy Of Our Full Feature Model
accuracy_score(y_test, y_pred)


# In[16]:

# Apply The Full Featured Classifier To The Test Data
y_important_pred = clf_important.predict(X_important_test)

# View The Accuracy Of Our Limited Feature  Model
accuracy_score(y_test, y_important_pred)


# As we can see there is no improvement after selecting some features from the given dataset  
# ** Implementing Logistic Regression **

# In[17]:

clf2 = LogisticRegression()

clf2.fit(X_train, y_train)


# In[18]:

y2_pred = clf2.predict(X_test)

accuracy_score(y_test, y2_pred)


# ** Implementing LogisticRegression with KFold **

# In[19]:

kf = KFold(n_splits=5)
scores = cross_val_score(X=X_train, y=y_train, cv=kf, estimator=clf2, n_jobs=1)
print(scores)
print(np.mean(scores))


# ** Implementing DecisionTreeClassifier **

# In[20]:

clf3 = DecisionTreeClassifier()

clf3.fit(X_train, y_train)

y3_pred = clf3.predict(X_test)

accuracy_score(y_test, y3_pred)


# In[21]:

clf4 = BernoulliNB()

clf4.fit(X_train, y_train)

y4_pred = clf3.predict(X_test)

accuracy_score(y_test, y4_pred)


# In[22]:

kf = KFold(n_splits=5)
scores = cross_val_score(X=X_train, y=y_train, cv=kf, estimator=clf3, n_jobs=1)
print(scores)
print(np.mean(scores))


# In[23]:

test_data = pd.read_csv('test.csv')
test_data[:4]


# In[24]:

test_data = test_data.dropna()


# In[25]:

enc = dict()

# Encoding the variable
for x in ['Gender', 'Married', 'Education', 'Dependents', 'Self_Employed', 'Property_Area']:
    enc[x]=LabelEncoder()
    enc[x].fit(test_data[x])
    print(enc[x].classes_)
    test_data[x]=enc[x].transform(test_data[x])


# In[26]:

test_data_id = test_data.iloc[:, 0]

# scaling the data
sc = StandardScaler()
test_data.iloc[:,1:-1] = sc.fit_transform(test_data.iloc[:,1:-1])

# Splitting the data into X and y
X_test = test_data.iloc[:, 1:]


# In[27]:

y_test_pred = clf2.predict(X_test)


# In[30]:

#y_test_pred = ['Y' if x==1 else 'N' for x in list(y_test_pred)]

result = pd.DataFrame(list(zip(test_data_id, y_test_pred)))

pd.DataFrame.to_csv(result, 'result.csv', index=None, header = ['Loan_ID','Loan_Status'])


# In[29]:

list(zip(test_data_id, y_test_pred))


# In[ ]:



