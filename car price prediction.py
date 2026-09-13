Car Price Prediction - CodeAlpha Task
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Sample Dataset
data = {
    'Present_Price': [5.59, 9.54, 9.85, 4.15, 6.75, 9.03, 11.2, 5.15],
    'Kms_Driven': [27000, 43000, 6900, 5200, 42450, 27000, 26000, 43000],
    'Fuel_Type': [0, 1, 0, 0, 1, 0, 1, 0],
    'Seller_Type': [0, 0, 0, 0, 0, 1, 1, 0],
    'Selling_Price': [3.35, 4.50, 7.10, 3.20, 4.10, 6.50, 8.20, 3.00]
}
df = pd.DataFrame(data)

print("Dataset:")
print(df.head())

X = df[['Present_Price','Kms_Driven','Fuel_Type','Seller_Type']]
y = df['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"\nR2 Score: {r2_score(y_test, y_pred):.2f}")
print(f"Model trained successfully!")
print("Task 3 Completed!")
