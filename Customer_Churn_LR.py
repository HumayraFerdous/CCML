import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.core.interchange.dataframe_protocol import Column
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score


data = pd.read_csv("Telco-Customer-Churn.csv")
#print(data.head())
#print(data.dtypes)
#print(data.shape)
#print(data.isnull().sum())
data.drop('customerID',axis=1,inplace=True)
data['Churn']= data['Churn'].map({'No':0,'Yes':1}).astype(int)
data['TotalCharges']=pd.to_numeric(data['TotalCharges'],errors = 'coerce')
data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())
"""print(data.isnull().sum())
print(data['Churn'].value_counts())
sns.countplot(x='Churn',data=data)
plt.title("Churn Distribution")
plt.show()

sns.histplot(data['tenure'], bins=30, kde=True)
plt.title('Tenure Distribution')
plt.show()

sns.countplot(x='gender', hue='Churn', data=data)
plt.title('Churn by Gender')
plt.show()

corr = data[['tenure','MonthlyCharges', 'TotalCharges', 'Churn']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()"""
X = data.drop('Churn',axis=1)
y = data['Churn']

numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = X.columns.difference(numeric_features)

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop = 'first')

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])
pipeline.fit(X_train,y_train)
y_pred = pipeline.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC-AUC Curve
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_score(y_test, y_pred_proba):.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid()
plt.show()