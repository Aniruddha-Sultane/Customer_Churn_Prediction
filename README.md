# Customer_Churn_Prediction

A robust predictive maintenance and customer retention system designed to identify high-risk customers likely to churn. This project compares multiple classification algorithms and utilizes feature engineering to optimize prediction accuracy.

🚀 Key Features
Model Comparison: Built and evaluated multiple classification models, including Logistic Regression, Decision Tree, and Random Forest.
Top-Tier Performance: Achieved a high 87% accuracy using the Random Forest model, specifically tuned for identifying churn patterns.
Feature Engineering: Implemented custom feature engineering for categorical variables like Contract type and Gender.
Hyperparameter Tuning: Utilized standard techniques to optimize model depth and estimator counts.
Real-time Dashboard: Built an interactive Streamlit interface for quick customer churn assessments.
🛠 Tech Stack
Model Engine: Scikit-Learn (Random Forest, Logistic Regression, Decision Trees)
Frontend: Streamlit
Data Manipulation: Pandas, Numpy
Visualization: Matplotlib, Seaborn
Persistence: Joblib (for model saving and loading)
📦 Getting Started
1. Clone the repository
git clone https://github.com/your-username/ChurnPrediction.git
cd ChurnPrediction
2. Install dependencies
pip install -r requirements.txt
3. Generate Data & Train Models
python generate_data.py
python model_trainer.py
4. Run the application
streamlit run app.py
📊 Performance Comparison
Random Forest: 87% Accuracy (Primary Model)
Logistic Regression: 82% Accuracy
Decision Tree: 79% Accuracy
📄 License
MIT License
