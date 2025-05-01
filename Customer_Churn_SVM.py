import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

data = pd.read_csv("Telco-Customer-Churn.csv")
data.drop('customerID',axis=1,inplace=True)
data['TotalCharges']=pd.to_numeric(data['TotalCharges'],errors = 'coerce')
data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())

cat_cols= [col for col in data.columns if data[col].dtype == 'object']
num_cols= [col for col in data.columns if data[col].dtype != 'object']
print(cat_cols)
print(num_cols)

le =  LabelEncoder()
sc = StandardScaler()
for i in cat_cols:
    data[i] = le.fit_transform(data[i])
for i in num_cols:
    data[[i]] = sc.fit_transform(data[[i]])
print(data['Churn'].value_counts())
X = data.drop('Churn',axis=1)
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC(kernel='linear',C = 1.0, class_weight={0:1,1:3})
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test,y_pred))
print("ROC-AUC: ", roc_auc_score(y_test, y_pred))

conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn','Churn'],
            yticklabels=['No Churn','Churn'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

feature_importance = np.abs(model.coef_[0])
feature_names = data.drop(columns = ['Churn']).columns
important_features = pd.Series(feature_importance,index = feature_names)
important_features = important_features.sort_values(ascending=False)

print(important_features.head(5))

