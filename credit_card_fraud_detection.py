# Credit Card Fraud Detection using Machine Learning

# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
data = pd.read_csv("creditcard.csv")

# Show Dataset Information
print("First 5 Rows of Dataset:\n")
print(data.head())

# Check Fraud and Normal Transactions
print("\nTransaction Counts:\n")
print(data['Class'].value_counts())

# Separate Features and Target
X = data.drop('Class', axis=1)
y = data['Class']

# Scale Amount and Time Columns
scaler = StandardScaler()

X['Amount'] = scaler.fit_transform(X[['Amount']])
X['Time'] = scaler.fit_transform(X[['Time']])

# Split Dataset into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Logistic Regression Model
model = LogisticRegression(max_iter=1000)

# Train Model
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Model Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy: ", accuracy)

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# Predict Single Transaction Example
sample_data = X_test.iloc[0:1]

prediction = model.predict(sample_data)

if prediction[0] == 0:
    print("\nTransaction is Normal")
else:
    print("\nTransaction is Fraud")
