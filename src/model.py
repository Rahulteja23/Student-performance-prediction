import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load dataset
data = pd.read_csv("...student performance prediction/dataset/student_data.csv")

# Features and target
X = data[['study_hours', 'attendance']]
y = data['marks']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
error = mean_absolute_error(y_test, predictions)
print("Model trained successfully")
print("Mean Absolute Error:", error)
