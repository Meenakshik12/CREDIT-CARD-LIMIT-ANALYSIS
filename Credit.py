#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[7]:


A=pd.read_csv("/Users/ashish/OneDrive/Desktop/Data Science/DS-1/credit.csv")


# In[8]:


A.head()


# In[9]:


A.info()


# In[10]:


A.describe()


# In[12]:


A.shape


# In[13]:


list(A.columns)


# # Exploratory Data Analysis(EDA)
# # Univariate Analysis

# In[14]:


cat=[]
con=[]
for i in A.columns:
    if(A[i].dtypes=="object"):
        cat.append(i)
    else:
        con.append(i)


# In[15]:


import matplotlib.pyplot as plt
import seaborn as sb


# In[23]:


for i in A.columns:
    if(A[i].dtypes == 'object'):
        sb.boxplot(A[i],A.Limit)#Boxplot between categorical and continuous columns
        plt.xlabel(i)
        plt.ylabel('Limit')
        plt.show()
    else:
        plt.scatter(A[i],A.Limit)# Scatter plot between continuous and continuous columns
        plt.xlabel('Limit')
        plt.ylabel(i)
        plt.show()


# In[24]:


cat


# In[25]:


con


# In[26]:


import seaborn as sb
import matplotlib.pyplot as plt


# In[27]:


for i in con:
    sb.distplot(A[i])
    plt.show()


# In[28]:


plt.figure(figsize=(40,19))
plt.subplot(3,3,1)
sb.distplot(A.Income)
plt.subplot(3,3,2)
sb.distplot(A.Limit)
plt.subplot(3,3,3)
sb.distplot(A.Rating)
plt.subplot(3,3,4)
sb.distplot(A.Age)


# In[29]:


A["Income"].hist()


# In[30]:


sb.countplot(A.Gender)


# In[31]:


sb.countplot(A.Ethnicity)


# In[32]:


A["Ethnicity"].value_counts()


# In[33]:


A["Ethnicity"].value_counts().plot(kind="barh")


# In[34]:


A["Ethnicity"].value_counts().plot(kind="bar")


# In[35]:


A["Ethnicity"].value_counts().plot(kind="pie")


# # Bivariate Analysis
# Continuous VS Continuous Variable

# In[36]:


plt.scatter(A.Income,A.Rating,c="red")
plt.xticks(range(0,200,20))
plt.yticks(range(0,1000,100))
plt.xlabel("Income")
plt.ylabel("Rating")
plt.title("Income vs Rating scatterplot")


# # Categorical VS Continuous Variable

# In[37]:


sb.boxplot(A.Ethnicity,A.Income)


# # Categorical VS Categorical Variables

# In[38]:


sb.countplot(A.Ethnicity,hue=A.Gender)


# In[39]:


pd.crosstab(A.Ethnicity,A.Gender)


# # Multivariate Analysis

# In[41]:


sb.pairplot(A)


# # DECISION TREE REGRESSOR BASED ON MAX DEPTH

# In[42]:


X=A.drop(labels=['Unnamed: 0','ID','Limit'],axis=1)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
X=X.apply(le.fit_transform)
Y=A[['Limit']]

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.3)

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor(random_state=20)
tp={'max_depth':[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]}
cv=GridSearchCV(dt,tp,scoring='neg_mean_absolute_error',cv=4)
cvmodel=cv.fit(xtrain,ytrain)
md=cvmodel.best_params_['max_depth']
dtr1=DecisionTreeRegressor(random_state=20,max_depth=md)
model=dtr1.fit(xtrain,ytrain)
pred=model.predict(xtest)
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(ytest,pred))


# # DECISION TREE REGRESSOR BASED ON MINIMUM SAMPLES SPLIT

# In[43]:


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor(random_state=20)
tp={'min_samples_split':[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]}
cv=GridSearchCV(dt,tp,scoring='neg_mean_absolute_error',cv=4)
cvmodel=cv.fit(xtrain,ytrain)
md=cvmodel.best_params_['min_samples_split']
dtr1=DecisionTreeRegressor(random_state=20,min_samples_split=md)
model=dtr1.fit(xtrain,ytrain)
pred=model.predict(xtest)
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(ytest,pred))


# # RANDOM FOREST REGRESSOR 

# In[44]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.3)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
rfr=RandomForestRegressor(random_state=30,criterion="mse")
tp={'n_estimators':range(2,50,1),'max_depth':range(2,10,1)}
cv=GridSearchCV(rfr,tp,scoring='neg_mean_squared_error',cv=4)
cvmodel=cv.fit(xtrain,ytrain)
cvmodel.best_params_['max_depth']
rfr1=RandomForestRegressor(random_state=20,max_depth=md)
model=rfr1.fit(xtrain,ytrain)
pred=model.predict(xtest)

from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(ytest,pred))


# In[ ]:




