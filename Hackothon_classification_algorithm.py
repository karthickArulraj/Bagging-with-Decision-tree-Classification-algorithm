#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import the Libraries

#For numerical libraries
import numpy as np

#To handle data in the form of rows and columns
import pandas as pd

#importing seaborn for statistical plots
import seaborn as sns

#importing ploting libraries
import matplotlib.pyplot as plt

#styling figures
plt.rc('font',size=14)
sns.set(style='white')
sns.set(style='whitegrid',color_codes=True)

#To enable plotting graphs in Jupyter notebook
get_ipython().run_line_magic('matplotlib', 'inline')

#importing the Encoding library
from sklearn.preprocessing import LabelEncoder

#Import SMOTE library for handling imbalance class
from imblearn.over_sampling import SMOTE

#Import Decision Tree Classifier machine learning Library
from sklearn.tree import DecisionTreeClassifier

# Import Logistic Regression machine learning library
from sklearn.linear_model import LogisticRegression 

#Import Naive Bayes' machine learning Library
from sklearn.naive_bayes import GaussianNB

#Import Sklearn package's data splitting function which is based on random function
from sklearn.model_selection import train_test_split

#Import the metrics
from sklearn import metrics

#Import the Voting classifier for Ensemble
from sklearn.ensemble import VotingClassifier


# #  Load the dataset

# In[2]:


#reading the CSV file into pandas dataframe
bank_data=pd.read_csv(r"C:\Users\karulrax\OneDrive - Intel Corporation\Documents\karthick\Learning\Hackothon\28may2021\train.csv")


# In[3]:


#Check top few records of the dataset
bank_data.head()


# In[4]:


#To show the detailed summary 
bank_data.info()


# In[5]:


#fill na with others
bank_data.fillna("Others", inplace=True)


# In[6]:


#change object datatypes to category except id 
bank_data['Gender']=bank_data.Gender.astype('category')
bank_data['Region_Code']=bank_data.Region_Code.astype('category')
bank_data['Occupation']=bank_data.Occupation.astype('category')
bank_data['Channel_Code']=bank_data.Channel_Code.astype('category')
bank_data['Credit_Product']=bank_data.Credit_Product.astype('category')
bank_data['Is_Active']=bank_data.Is_Active.astype('category')
bank_data.dtypes


# In[7]:


#To get the shape 
bank_data.shape


# # Exploratory data quality report

# ## Univariate analysis

# In[8]:


#To get the number of columns
bank_data.columns


# In[9]:


#Analyze the distribution of the dataset
bank_data.describe().T


# ## Description of independent attribute

# ### Age

# In[10]:


print('Minimum age: ', bank_data['Age'].min())
print('Maximum age: ',bank_data['Age'].max())
print('Mean value: ', bank_data['Age'].mean())
print('Median value: ',bank_data['Age'].median())
print('Standard deviation: ', bank_data['Age'].std())
print('Null values: ',bank_data['Age'].isnull().any())


# In[11]:


Q1=bank_data['Age'].quantile(q=0.25)
Q3=bank_data['Age'].quantile(q=0.75)
print('1st Quartile (Q1) is: ', Q1)
print('3st Quartile (Q3) is: ', Q3)


# In[12]:


# IQR=Q3-Q1
#lower 1.5*IQR whisker i.e Q1-1.5*IQR
#upper 1.5*IQR whisker i.e Q3+1.5*IQR
L_outliers=Q1-1.5*(Q3-Q1)
U_outliers=Q3+1.5*(Q3-Q1)
print('Lower outliers in Ages: ', L_outliers)
print('Upper outliers in Ages: ', U_outliers)


# In[13]:


print('Number of outliers in Age upper : ', bank_data[bank_data['Age']>90]['Age'].count())
print('Number of outliers in Age lower : ', bank_data[bank_data['Age']<0]['Age'].count())
print('% of Outlier in Age upper: ',round(bank_data[bank_data['Age']>90]['Age'].count()*100/len(bank_data)), '%')
print('% of Outlier in Age lower: ',round(bank_data[bank_data['Age']<0]['Age'].count()*100/len(bank_data)), '%')


# In[14]:


fig, (ax1,ax2,ax3)=plt.subplots(1,3,figsize=(13,5))

#boxplot
sns.boxplot(x='Age',data=bank_data,orient='v',ax=ax1)
ax1.set_xlabel('Client age', fontsize=15)
ax1.set_ylabel('Age', fontsize=15)
ax1.set_title('Distribution of age', fontsize=15)
ax1.tick_params(labelsize=15)

#distplot
sns.distplot(bank_data['Age'],ax=ax2)
ax2.set_xlabel('Age', fontsize=15)
ax2.set_ylabel('Occurrence', fontsize=15)
ax2.set_title('Age vs Occurrence', fontsize=15)
ax2.tick_params(labelsize=15)

#histogram
ax3.hist(bank_data['Age'])
ax3.set_xlabel('Age', fontsize=15)
ax3.set_ylabel('Occurrence', fontsize=15)
ax3.set_title('Age vs Occurrence', fontsize=15)
ax3.tick_params(labelsize=15)

plt.subplots_adjust(wspace=0.5)
plt.tight_layout()


# ### Occupation

# In[15]:


print('Jobs: \n', bank_data['Occupation'].unique())


# In[16]:


bank_data.groupby('Occupation').size()


# In[17]:


fig, ax=plt.subplots()
fig.set_size_inches(20,8)

#countplot
sns.countplot(bank_data['Occupation'],data=bank_data)
ax1.set_xlabel('Job', fontsize=18)
ax1.set_ylabel('Count', fontsize=18)
ax1.set_title('Job vs Count', fontsize=18)
ax1.tick_params(labelsize=20)


# ### Gender

# In[18]:


print('gender: \n', bank_data['Gender'].unique())


# In[19]:


bank_data.groupby('Gender').size()


# In[20]:


fig, ax=plt.subplots()
fig.set_size_inches(10,5)

#countplot
sns.countplot(bank_data['Gender'],data=bank_data)
ax.set_xlabel('Gender', fontsize=15)
ax.set_ylabel('Count', fontsize=15)
ax.set_title('Gender vs Count', fontsize=15)
ax.tick_params(labelsize=15)


# ### Credit_Product

# In[21]:


print('Credit_Product: \n', bank_data['Credit_Product'].unique())


# In[22]:


bank_data.groupby('Credit_Product').size()


# In[23]:


fig, ax=plt.subplots()
fig.set_size_inches(10,5)

#countplot
sns.countplot(bank_data['Credit_Product'],data=bank_data)
ax.set_xlabel('Credit_Product', fontsize=15)
ax.set_ylabel('Count', fontsize=15)
ax.set_title('Credit_Product vs Count', fontsize=15)
ax.tick_params(labelsize=15)


# ### Avg_Account_Balance

# In[24]:



print('Minimum balance: ', bank_data['Avg_Account_Balance'].min())
print('Maximum balance: ',bank_data['Avg_Account_Balance'].max())
print('Mean value: ', bank_data['Avg_Account_Balance'].mean())
print('Median value: ',bank_data['Avg_Account_Balance'].median())
print('Standard deviation: ', bank_data['Avg_Account_Balance'].std())
print('Null values: ',bank_data['Avg_Account_Balance'].isnull().any())


# In[25]:


from scipy.stats import zscore

bank_data[['Avg_Account_Balance']].mean()
bank_data[['Avg_Account_Balance']].mean()

bank_data['balance_outliers'] = bank_data['Avg_Account_Balance']
bank_data['balance_outliers']= zscore(bank_data['balance_outliers'])

condition1 = (bank_data['balance_outliers']>3) | (bank_data['balance_outliers']<-3 )
bank_data = bank_data.drop(bank_data[condition1].index, axis = 0, inplace = False)


# In[26]:


bank_data = bank_data.drop('balance_outliers', axis=1)


# In[27]:


print('Minimum balance: ', bank_data['Avg_Account_Balance'].min())
print('Maximum balance: ',bank_data['Avg_Account_Balance'].max())
print('Mean value: ', bank_data['Avg_Account_Balance'].mean())
print('Median value: ',bank_data['Avg_Account_Balance'].median())
print('Standard deviation: ', bank_data['Avg_Account_Balance'].std())
print('Null values: ',bank_data['Avg_Account_Balance'].isnull().any())


# In[28]:


fig, (ax1,ax2,ax3)=plt.subplots(1,3,figsize=(13,5))

#boxplot
sns.boxplot(x='Avg_Account_Balance',data=bank_data,orient='v',ax=ax1)
ax1.set_xlabel('Client balance', fontsize=15)
ax1.set_ylabel('Avg_Account_Balance', fontsize=15)
ax1.set_title('Distribution of balance', fontsize=15)
ax1.tick_params(labelsize=15)

#distplot
sns.distplot(bank_data['Avg_Account_Balance'],ax=ax2)
ax2.set_xlabel('Avg_Account_Balance', fontsize=15)
ax2.set_ylabel('Occurrence', fontsize=15)
ax2.set_title('Balance vs Occurrence', fontsize=15)
ax2.tick_params(labelsize=15)

#histogram
ax3.hist(bank_data['Avg_Account_Balance'])
ax3.set_xlabel('Avg_Account_Balance', fontsize=15)
ax3.set_ylabel('Occurrence', fontsize=15)
ax3.set_title('Balance vs Occurrence', fontsize=15)
ax3.tick_params(labelsize=15)

plt.subplots_adjust(wspace=0.5)
plt.tight_layout()


# ### vintage

# In[29]:


print('Minimum age: ', bank_data['Vintage'].min())
print('Maximum age: ',bank_data['Vintage'].max())
print('Mean value: ', bank_data['Vintage'].mean())
print('Median value: ',bank_data['Vintage'].median())
print('Standard deviation: ', bank_data['Vintage'].std())
print('Null values: ',bank_data['Vintage'].isnull().any())


# In[30]:


Q1=bank_data['Vintage'].quantile(q=0.25)
Q3=bank_data['Vintage'].quantile(q=0.75)
print('1st Quartile (Q1) is: ', Q1)
print('3st Quartile (Q3) is: ', Q3)


# In[31]:


# IQR=Q3-Q1
#lower 1.5*IQR whisker i.e Q1-1.5*IQR
#upper 1.5*IQR whisker i.e Q3+1.5*IQR
L_outliers=Q1-1.5*(Q3-Q1)
U_outliers=Q3+1.5*(Q3-Q1)
print('Lower outliers in Vintage: ', L_outliers)
print('Upper outliers in Vintage: ', U_outliers)


# In[32]:


print('Number of outliers in Vintage upper : ', bank_data[bank_data['Vintage']>152.5]['Vintage'].count())
print('Number of outliers in Vintage lower : ', bank_data[bank_data['Vintage']<-59.6]['Vintage'].count())
print('% of Outlier in Vintage upper: ',round(bank_data[bank_data['Vintage']>152.5]['Vintage'].count()*100/len(bank_data)), '%')
print('% of Outlier in Vintage lower: ',round(bank_data[bank_data['Vintage']<-59.5]['Vintage'].count()*100/len(bank_data)), '%')


# In[33]:


fig, (ax1,ax2,ax3)=plt.subplots(1,3,figsize=(13,5))

#boxplot
sns.boxplot(x='Vintage',data=bank_data,orient='v',ax=ax1)
ax1.set_xlabel('Client Vintage', fontsize=15)
ax1.set_ylabel('Vintage', fontsize=15)
ax1.set_title('Distribution of Vintage', fontsize=15)
ax1.tick_params(labelsize=15)

#distplot
sns.distplot(bank_data['Vintage'],ax=ax2)
ax2.set_xlabel('Vintage', fontsize=15)
ax2.set_ylabel('Occurrence', fontsize=15)
ax2.set_title('Vintage vs Occurrence', fontsize=15)
ax2.tick_params(labelsize=15)

#histogram
ax3.hist(bank_data['Vintage'])
ax3.set_xlabel('Vintage', fontsize=15)
ax3.set_ylabel('Occurrence', fontsize=15)
ax3.set_title('VintageAge vs Occurrence', fontsize=15)
ax3.tick_params(labelsize=15)

plt.subplots_adjust(wspace=0.5)
plt.tight_layout()


# ### Region_Code

# In[34]:


print('Region_Code: \n', bank_data['Region_Code'].unique())


# In[35]:


bank_data.groupby('Region_Code').size()


# In[36]:


fig, ax=plt.subplots()
fig.set_size_inches(20,8)

#countplot
sns.countplot(bank_data['Region_Code'],data=bank_data)
ax1.set_xlabel('Region_Code', fontsize=18)
ax1.set_ylabel('Count', fontsize=18)
ax1.set_title('Region_Code vs Count', fontsize=18)
ax1.tick_params(labelsize=20)


# ### Channel_Code

# In[37]:


print('Channel_Code: \n', bank_data['Channel_Code'].unique())


# In[38]:


bank_data.groupby('Channel_Code').size()


# In[39]:


fig, ax=plt.subplots()
fig.set_size_inches(20,8)

#countplot
sns.countplot(bank_data['Channel_Code'],data=bank_data)
ax1.set_xlabel('Channel_Code', fontsize=18)
ax1.set_ylabel('Count', fontsize=18)
ax1.set_title('Channel_Code vs Count', fontsize=18)
ax1.tick_params(labelsize=20)


# ### Is_Active

# In[40]:


print('Is_Active: \n', bank_data['Is_Active'].unique())


# In[41]:


bank_data.groupby('Is_Active').size()


# In[42]:


fig, ax=plt.subplots()
fig.set_size_inches(10,5)

#countplot
sns.countplot(bank_data['Is_Active'],data=bank_data)
ax.set_xlabel('Is_Active', fontsize=15)
ax.set_ylabel('Count', fontsize=15)
ax.set_title('Is_Active vs Count', fontsize=15)
ax.tick_params(labelsize=15)


# ### Is_Lead

# In[43]:


print('Is_Lead: \n', bank_data['Is_Lead'].unique())


# In[44]:



print(bank_data.groupby('Is_Lead').size())


# In[45]:


fig, ax=plt.subplots()
fig.set_size_inches(10,5)

#countplot
sns.countplot(bank_data['Is_Lead'],data=bank_data)
ax.set_xlabel('Is_Lead', fontsize=15)
ax.set_ylabel('Count', fontsize=15)
ax.set_title('Is_Lead vs Count', fontsize=15)
ax.tick_params(labelsize=15)


# # Encode the categorical variables

# In[46]:


#drop id and region_code
bank_data = bank_data.drop(['ID','Region_Code'], axis=1)


# In[47]:


#Encoding of categorical variables

labelencoder_X=LabelEncoder()

bank_data['Gender']=labelencoder_X.fit_transform(bank_data['Gender'])
bank_data['Occupation']=labelencoder_X.fit_transform(bank_data['Occupation'])
bank_data['Channel_Code']=labelencoder_X.fit_transform(bank_data['Channel_Code'])
bank_data['Credit_Product']=labelencoder_X.fit_transform(bank_data['Credit_Product'])
bank_data['Is_Active']=labelencoder_X.fit_transform(bank_data['Is_Active'])
bank_data.head()


# In[48]:



bank_data.describe().T


# # Multivariate Analysis
# ## Visualization

# In[49]:



# corrlation matrix 
cor=bank_data.corr()
cor


# In[50]:


# correlation plot---heatmap
sns.set(font_scale=1.15)
fig,ax=plt.subplots(figsize=(18,15))
sns.heatmap(cor,vmin=0.8,cmap='cividis', annot=True,linewidths=0.01,center=0,linecolor="white",cbar=False,square=True)
plt.title('Correlation between attributes',fontsize=18)
ax.tick_params(labelsize=18)


# In[51]:


sns.pairplot(bank_data,hue='Is_Lead')


# # One Hot Encoder

# In[52]:


#HOT ENCODER
#convert to categorical data to dummy data
bank_data = pd.get_dummies(bank_data, columns=["Occupation","Credit_Product","Channel_Code"])
bank_data.head()


# # Test Data 

# In[53]:


#reading the CSV file into pandas dataframe for testing
credit_test = pd.read_csv(r"C:\Users\karulrax\OneDrive - Intel Corporation\Documents\karthick\Learning\Hackothon\28may2021\test.csv")
credit_test


# In[54]:


#fill na with others
credit_test.fillna("Others", inplace=True)


# In[55]:


print(credit_test.isnull().values.sum())


# In[56]:


#take a copy of id from dataframe
ids = credit_test["ID"]


# In[57]:


#drop id and region_code
credit_test=credit_test.drop(["ID",'Region_Code'],axis =1 )
credit_test.reset_index(drop=True, inplace=True)


# In[58]:


#Encoding of categorical variables

labelencoder_X=LabelEncoder()

credit_test['Gender']=labelencoder_X.fit_transform(credit_test['Gender'])
credit_test['Occupation']=labelencoder_X.fit_transform(credit_test['Occupation'])
credit_test['Channel_Code']=labelencoder_X.fit_transform(credit_test['Channel_Code'])
credit_test['Credit_Product']=labelencoder_X.fit_transform(credit_test['Credit_Product'])
credit_test['Is_Active']=labelencoder_X.fit_transform(credit_test['Is_Active'])
credit_test.head()


# In[59]:


credit_test= pd.get_dummies(credit_test, columns=["Occupation","Credit_Product","Channel_Code"])
credit_test.head()


# # Model Building: Managing Imbalance the Target column and normalizing the columns

# In[60]:


# create training and test data
bank_new=bank_data
X_train=bank_new.loc[:,bank_new.columns!='Is_Lead']
y_train=bank_new.loc[:,bank_new.columns=='Is_Lead']
X_test = credit_test


# In[61]:


columns=X_train.columns


# ## Normalizing/Scaling the Columns

# In[62]:


#for normalization
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# ## Managing Imbalance classes

# In[63]:


bank_data.groupby('Is_Lead').size()


# In[64]:


#Import the library for handling the imbalance dataset
from imblearn.over_sampling import SMOTE
Ov_sampling=SMOTE(random_state=100)
# now use SMOTE to oversample our train data which have features data_train_X and labels in data_train_y
ov_data_X,ov_data_y=Ov_sampling.fit_sample(X_train,y_train)
ov_data_X=pd.DataFrame(data=ov_data_X,columns=columns)
ov_data_y=pd.DataFrame(ov_data_y,columns=['Is_Lead'])


# In[65]:


print('length of oversampled data is   ',len(ov_data_X))
print('Number of no subscription in oversampled data ' ,len(ov_data_y[ov_data_y['Is_Lead']==0]))
print('Number of subscription ' ,len(ov_data_y[ov_data_y['Is_Lead']==1]))
print('Proportion of no subscription data in oversampled data is ' ,len(ov_data_y[ov_data_y['Is_Lead']==0])/len(ov_data_X))
print('Proportion of subscription data in oversampled data is ' ,len(ov_data_y[ov_data_y['Is_Lead']==1])/len(ov_data_X))


# In[66]:


ov_data_y['Is_Lead'].value_counts()


# # Model Building

# ## Decision Tree Model

# In[67]:


# Regularizing the Decision tree classifier and fitting the model
reg_dt_model = DecisionTreeClassifier(criterion = 'entropy', max_depth = 7,random_state=100,min_samples_leaf=5)
reg_dt_model.fit(ov_data_X,ov_data_y)


# In[68]:


y_predict = reg_dt_model.predict(X_test)


# ## Ensemble Learning - Bagging

# In[69]:



from sklearn.ensemble import BaggingClassifier


# In[70]:


bgcl_1 = BaggingClassifier(base_estimator=reg_dt_model, n_estimators=100,random_state=100, max_samples = 0.6, max_features = 0.8)
bgcl_1 = bgcl_1.fit(ov_data_X,ov_data_y)


# In[71]:


bgcl_1_y_predict_1 = bgcl_1.predict(X_test)
bgcl_1_y_predict_1


# # Export CSV file

# In[72]:


#create output csv file
bgcl_1_df1 = pd.DataFrame({"ID" : ids,"Is_Lead" : bgcl_1_y_predict_1 })
bgcl_1_df1 
bgcl_1_df1 .to_csv(r"C:\Users\karulrax\OneDrive - Intel Corporation\Documents\karthick\Learning\Hackothon\28may2021\bgcl_226677321_df1 _Submissions.csv")


# In[ ]:




