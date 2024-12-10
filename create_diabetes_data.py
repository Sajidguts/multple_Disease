import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
num_samples = 768

# Generate synthetic data
data = {
    'Pregnancies': np.random.randint(0, 17, size=num_samples),
    'Glucose': np.random.randint(0, 200, size=num_samples),
    'BloodPressure': np.random.randint(0, 122, size=num_samples),
    'SkinThickness': np.random.randint(0, 100, size=num_samples),
    'Insulin': np.random.randint(0, 846, size=num_samples),
    'BMI': np.round(np.random.uniform(18.5, 50.0, size=num_samples), 1),
    'DiabetesPedigreeFunction': np.round(np.random.uniform(0.0, 2.5, size=num_samples), 3),
    'Age': np.random.randint(21, 81, size=num_samples),
    'Outcome': np.random.randint(0, 2, size=num_samples)  # Binary outcome
}

# Create DataFrame
diabetes_df = pd.DataFrame(data)

# Save to CSV
diabetes_df.to_csv('diabetes.csv', index=False)

print("diabetes.csv has been created successfully.")