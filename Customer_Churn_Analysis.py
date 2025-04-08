import pandas as pd
import numpy as np
import pylab as plt
import seaborn as sns

sns.set_style("whitegrid")


data = pd.read_csv("Telco-Customer-Churn.csv")
#print(data.head())

#searching missing values
#print(data.isnull().sum())

data.drop('customerID',axis=1,inplace = True)
# Convert data types
pd.set_option('future.no_silent_downcasting', True)
binary_cols = ['Partner','Dependents','PhoneService','PaperlessBilling','Churn']
data[binary_cols] = data[binary_cols].replace({'Yes':0,'No':1})
data['gender'] = data ['gender'].map({'Male':1,'Female':0})
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'],errors ='coerce')

#Handing outliers
numerical_cols = ['tenure','MonthlyCharges','TotalCharges']
plt.figure(figsize=(12,7))
for i,col in enumerate(numerical_cols,1):
    plt.subplot(1,3,i)
    sns.boxplot(y=data[col])
plt.tight_layout()
plt.show()
#No outliers found

#Exploratory data analysis
print(data.describe())

#Churn distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Churn',data=data)
plt.title("Customer Churn Distribution")
plt.show()

#Correlation Matrix
plt.figure(figsize=(6,4))
sns.heatmap(data[numerical_cols + ['Churn']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

#Monthly charges vs. churn
plt.figure(figsize=(10,6))
sns.boxplot(x='Churn',y='MonthlyCharges',data=data)
plt.title("Monthly Charges by Churn Status")
plt.show()

#Tenure vs. Churn
plt.figure(figsize = (10,6))
sns.boxplot(x = 'Churn',y='tenure',data = data)
plt.title("Tenure by Churn Status")
plt.show()

#Categorical Feature Analysis
cat_cols = ['InternetService','Contract','PaymentMethod']
plt.figure(figsize = (15,10))
for i,col in enumerate(cat_cols,1):
    plt.subplot(2,2,i)
    sns.countplot(x=col,hue='Churn',data=data)
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
