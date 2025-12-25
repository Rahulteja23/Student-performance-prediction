# Student Performance Prediction using Machine Learning

## Problem Statement
Predict student academic performance based on study hours and attendance using Machine Learning techniques.

## Dataset
The dataset contains the following features:
- Study hours per day
- Attendance percentage
- Final exam marks

## Approach
1. Loaded the dataset using Pandas.
2. Performed train-test split to evaluate model performance.
3. Used Linear Regression to model the relationship between features and marks.
4. Evaluated the model using Mean Absolute Error.

## Technologies Used
-> Python
-> Pandas
-> scikit-learn

## Visualization
- Actual vs Predicted Marks plot to evaluate model performance
- Study Hours vs Marks plot to understand feature relationships

## Results
The model was able to predict student marks with reasonable accuracy on a small dataset, demonstrating the effectiveness of basic regression techniques.

## How to Run
1. Install dependencies:
   pip install -r requirements.txt
2. Run the model:
   python src/model.py
