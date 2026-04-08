import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

class ChurnModeler:
    def __init__(self, data_path='customer_churn.csv'):
        self.df = pd.read_csv(data_path)
        self.le = LabelEncoder()
        self.sc = StandardScaler()
        self.best_model = None

    def preprocess_data(self):
        # Basic feature engineering
        df = self.df.drop('CustomerID', axis=1)
        for col in ['Gender', 'Contract']:
            df[col] = self.le.fit_transform(df[col])
        
        X = df.drop('Churn', axis=1)
        y = df['Churn']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = self.sc.fit_transform(X_train)
        X_test = self.sc.transform(X_test)
        
        return X_train, X_test, y_train, y_test

    def train_and_compare(self):
        X_train, X_test, y_train, y_test = self.preprocess_data()
        
        models = {
            'Logistic Regression': LogisticRegression(),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        }
        
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            results[name] = acc
            print(f"{name} Accuracy: {acc:.4f}")
            
        self.best_model = models['Random Forest']
        joblib.dump(self.best_model, 'churn_model.joblib')
        joblib.dump(self.sc, 'scaler.joblib')
        joblib.dump(self.le, 'label_encoder.joblib')
        
        return results

if __name__ == "__main__":
    modeler = ChurnModeler()
    modeler.train_and_compare()