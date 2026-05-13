# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm
1. Collect and Prepare Data:Gather employee data such as experience, age, education, and skills. Clean the dataset and split it into training and testing sets.
2.Train the Decision Tree Regressor Model:Import the Decision Tree Regressor from Scikit-learn and train the model using the training dataset.
3.Predict Employee Salary:Use the trained model to predict employee salaries for the test data or new employee details.
4.Evaluate the Model Performance:Compare predicted salaries with actual salaries using evaluation metrics such as Mean Squared Error (MSE) or R² score to measure accuracy.
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: MENAKA M S
RegisterNumber: 212225040232 
*/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
data = {
    "Experience": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Age": [22, 24, 26, 28, 30, 32, 34, 36, 38, 40],
    "Salary": [25000, 30000, 35000, 45000, 50000,
               60000, 65000, 75000, 85000, 95000]
}
df = pd.DataFrame(data)
print("Dataset:\n")
print(df)
X = df[["Experience", "Age"]]
y = df["Salary"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = DecisionTreeRegressor(
    criterion='squared_error',
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nModel Evaluation:")
print("Mean Squared Error:")
print(mean_squared_error(y_test, y_pred))
print("\nMean Absolute Error:")
print(mean_absolute_error(y_test, y_pred))

print("\nR2 Score:")
print(r2_score(y_test, y_pred))
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})
print("\nFeature Importance:")
print(importance)
plt.figure(figsize=(14,8))
plot_tree(
    model,
    feature_names=X.columns,
    filled=True
)
plt.title("Decision Tree Regressor for Salary Prediction")
plt.show()
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual Salary vs Predicted Salary")
plt.grid(True)
plt.show()
employee_data = [[5, 30]]
predicted_salary = model.predict(employee_data)
print("\nPredicted Salary for Employee:")
print(f"Predicted Salary = {predicted_salary[0]:.2f}")
```
## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee](sam.png)
<img width="842" height="366" alt="image" src="https://github.com/user-attachments/assets/5e9426ca-0328-421c-808a-9cb29f5a620b" />
<img width="827" height="472" alt="image" src="https://github.com/user-attachments/assets/da0cfb0d-8fb4-4097-a3b0-c2aeef7aa19a" />
<img width="657" height="423" alt="image" src="https://github.com/user-attachments/assets/7a64edae-5af2-4272-a085-31d880d163ad" />
<img width="190" height="35" alt="image" src="https://github.com/user-attachments/assets/1cad31ef-6873-4761-81c1-ed7b505c099f" />
## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
