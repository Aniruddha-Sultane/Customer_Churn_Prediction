import pandas as pd
import numpy as np

def generate_sample_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'CustomerID': range(1000, 1000 + num_samples),
        'Gender': np.random.choice(['Male', 'Female'], num_samples),
        'Age': np.random.randint(18, 80, num_samples),
        'Tenure': np.random.randint(0, 10, num_samples),
        'MonthlyCharges': np.random.uniform(20, 150, num_samples),
        'TotalCharges': np.random.uniform(100, 5000, num_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], num_samples),
        'Churn': np.random.choice([0, 1], num_samples, p=[0.8, 0.2])
    }
    df = pd.DataFrame(data)
    df.to_csv('customer_churn.csv', index=False)
    return df

if __name__ == "__main__":
    generate_sample_data()
    print("Sample dataset 'customer_churn.csv' generated.")