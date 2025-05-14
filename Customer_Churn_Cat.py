import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
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

X = data.drop('Churn', axis=1)
y = data['Churn']

cat_features = X.select_dtypes(include=['object']).columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    cat_features=cat_features,
    eval_metric='Accuracy',
    verbose=100,
    random_state=42
)


model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)


y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))