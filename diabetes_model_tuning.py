import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
data = pd.read_csv('diabetes.csv')

# Split dataset into features and target variable
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression with hyperparameter tuning
logistic_model = LogisticRegression(max_iter=200)
param_grid_logistic = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
    'solver': ['liblinear', 'saga']  # Solvers for optimization
}

grid_logistic = GridSearchCV(logistic_model, param_grid_logistic, cv=5)
grid_logistic.fit(X_train, y_train)

# Best parameters for Logistic Regression
print("Best parameters for Logistic Regression:", grid_logistic.best_params_)

# Evaluate the tuned Logistic Regression model
y_pred_logistic = grid_logistic.predict(X_test)
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_logistic))

# Decision Tree with hyperparameter tuning
decision_tree_model = DecisionTreeClassifier()
param_grid_tree = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}

grid_tree = GridSearchCV(decision_tree_model, param_grid_tree, cv=5)
grid_tree.fit(X_train, y_train)

# Best parameters for Decision Tree
print("Best parameters for Decision Tree:", grid_tree.best_params_)

# Evaluate the tuned Decision Tree model
y_pred_tree = grid_tree.predict(X_test)
print("\nDecision Tree Classification Report:")
print(classification_report(y_test, y_pred_tree))