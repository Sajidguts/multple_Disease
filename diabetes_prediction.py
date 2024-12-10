import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('diabetes.csv')

# Split the dataset into features and target variable
X = data.drop('Outcome', axis=1)  # Features
y = data['Outcome']                 # Target variable

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
logistic_model = LogisticRegression(max_iter=200)
logistic_model.fit(X_train, y_train)  # Train the model

# Decision Tree Model
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train, y_train)  # Train the model

# Predict on the test set
y_pred_logistic = logistic_model.predict(X_test)
y_pred_tree = decision_tree_model.predict(X_test)

# Evaluate accuracy for both models
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
accuracy_tree = accuracy_score(y_test, y_pred_tree)

# Print the accuracy results
print(f'Logistic Regression Accuracy: {accuracy_logistic:.2f}')
print(f'Decision Tree Accuracy: {accuracy_tree:.2f}')