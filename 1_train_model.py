# Import Required Libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load Data
data = pd.read_csv("simulated_data.csv")

# Feature Engineering: Add Interaction Feature (Age * Income)
data['age_income_interaction'] = data['age'] * data['income']

# Train-Test-Validation Split
# Split the data into training (60%), validation (20%), and test (20%) sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)  # 60-20-20 split

# Split Features and Targets
# Input features and target variable for training, validation, and testing
features = ['age', 'income', 'purchase_history', 'age_income_interaction']
X_train = train_data[features]
y_train = train_data['customer_satisfaction']
X_val = val_data[features]
y_val = val_data['customer_satisfaction']
X_test = test_data[features]
y_test = test_data['customer_satisfaction']

# Scale Features
# Standardize features to have zero mean and unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# Save Scaler
# Save the trained scaler to a file for future use
joblib.dump(scaler, "scaler.pkl")



# Train Random Forest Regressor with Hyperparameter Tuning
# Using GridSearchCV to find the best hyperparameters for the Random Forest Regressor
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_train_scaled, y_train)

# Save the Best Model for future use
best_rf_reg = grid_search.best_estimator_
joblib.dump(best_rf_reg, "best_customer_satisfaction_model.pkl")

# Evaluate on Test Data
y_pred = best_rf_reg.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print Evaluation Metrics
print("Mean Absolute Error:", mae)
print("R-squared Score:", r2)

# Save Evaluation Metrics to a Text File
with open("evaluation_metrics.txt", "w") as f:
    f.write(f"Mean Absolute Error (MAE): {mae}\n")
    f.write(f"R-squared Score (R2): {r2}\n")

# Save Predictions and Actual Values to a CSV File
# Create a DataFrame to store actual vs predicted values along with input features for reference
predictions_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred,
    "Age": X_test['age'],
    "Income": X_test['income'],
    "Purchase History": X_test['purchase_history']
})
predictions_df.to_csv("predictions.csv", index=False)


# Confirm Completion
# Indicate that the model, scaler, predictions, and metrics have been successfully saved
print("\nModel, Scaler, Predictions, and Metrics Saved Successfully!")
