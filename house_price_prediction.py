"""
House Price Prediction App
A complete Machine Learning workflow demonstrating data collection, 
exploration, preprocessing, model training, and predictions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("HOUSE PRICE PREDICTION APP")
print("="*60)

# Step 1: Understand the Dataset
print("\n[STEP 1] Understanding the Dataset")
print("-" * 60)
print("Input Features (Independent Variables):")
print("  • Area (square feet)")
print("  • Number of Bedrooms")
print("  • Number of Bathrooms")
print("  • Location")
print("  • Age of Property (years)")
print("  • Parking Availability")
print("\nTarget Variable (What we predict):")
print("  • Price (in dollars)")

# Step 2: Load the Dataset
print("\n[STEP 2] Loading the Dataset")
print("-" * 60)
data = pd.read_csv("house_data.csv")
print(f"Dataset Shape: {data.shape[0]} rows, {data.shape[1]} columns")
print("\nFirst 5 rows of the dataset:")
print(data.head())

# Step 3: Explore the Data
print("\n[STEP 3] Exploring the Data")
print("-" * 60)

print("\n3.1 Dataset Information:")
print(data.info())

print("\n3.2 Missing Values:")
missing_values = data.isnull().sum()
print(missing_values)
if missing_values.sum() == 0:
    print("✓ No missing values found!")

print("\n3.3 Statistical Summary:")
print(data.describe())

# Step 4: Visualize the Data
print("\n[STEP 4] Visualizing the Data")
print("-" * 60)

# Create visualization directory if needed
import os
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

# Plot 1: Area vs Price
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(data["area"], data["price"], alpha=0.6, color='blue')
plt.xlabel("Area (sq ft)", fontsize=10)
plt.ylabel("Price ($)", fontsize=10)
plt.title("Area vs Price")
plt.grid(True, alpha=0.3)

# Plot 2: Bedrooms vs Price
plt.subplot(1, 3, 2)
plt.scatter(data["bedrooms"], data["price"], alpha=0.6, color='green')
plt.xlabel("Bedrooms", fontsize=10)
plt.ylabel("Price ($)", fontsize=10)
plt.title("Bedrooms vs Price")
plt.grid(True, alpha=0.3)

# Plot 3: Age vs Price
plt.subplot(1, 3, 3)
plt.scatter(data["age"], data["price"], alpha=0.6, color='red')
plt.xlabel("Age (years)", fontsize=10)
plt.ylabel("Price ($)", fontsize=10)
plt.title("Age vs Price")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/feature_analysis.png', dpi=100, bbox_inches='tight')
print("✓ Visualization saved: visualizations/feature_analysis.png")
plt.close()

print("\nObservations:")
print("  • Larger houses (more area) tend to have higher prices")
print("  • More bedrooms generally correlate with higher prices")
print("  • Newer houses (lower age) tend to be more expensive")

# Step 5: Data Preprocessing
print("\n[STEP 5] Data Preprocessing")
print("-" * 60)

# Create a copy for preprocessing
df = data.copy()

# Encode categorical variable (location)
print("\n5.1 Encoding Categorical Variables (Location):")
le = LabelEncoder()
df['location_encoded'] = le.fit_transform(df['location'])
print(f"Location Encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Select features and target
X = df[["area", "bedrooms", "bathrooms", "location_encoded", "age", "parking"]]
y = df["price"]

print("\n5.2 Features (X):")
print(f"  Shape: {X.shape}")
print(f"  Columns: {list(X.columns)}")

print("\n5.3 Target (y):")
print(f"  Shape: {y.shape}")
print(f"  Mean Price: ${y.mean():.2f}")
print(f"  Median Price: ${y.median():.2f}")

# Train-Test Split
print("\n5.4 Train-Test Split (80-20):")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"  Training Set Size: {X_train.shape[0]} samples")
print(f"  Testing Set Size: {X_test.shape[0]} samples")
print(f"  Train-Test Ratio: {X_train.shape[0]/len(X)*100:.1f}% - {X_test.shape[0]/len(X)*100:.1f}%")

# Step 6: Train the AI Model
print("\n[STEP 6] Training the AI Model (Linear Regression)")
print("-" * 60)

model = LinearRegression()
model.fit(X_train, y_train)

print("✓ Model trained successfully!")
print("\nModel Coefficients (Feature Importance):")
feature_names = X.columns
for feature, coef in zip(feature_names, model.coef_):
    print(f"  {feature:20s}: ${coef:12,.2f} per unit")
print(f"  {'Intercept':20s}: ${model.intercept_:12,.2f}")

# Step 7: Make Predictions
print("\n[STEP 7] Making Predictions")
print("-" * 60)

# Predictions on training set
y_train_pred = model.predict(X_train)

# Predictions on test set
y_test_pred = model.predict(X_test)

print("✓ Predictions made on both training and test sets")

# Step 8: Model Evaluation
print("\n[STEP 8] Model Evaluation")
print("-" * 60)

# Calculate metrics for training set
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, y_train_pred)

print("\nTraining Set Metrics:")
print(f"  Mean Squared Error (MSE): ${train_mse:,.2f}")
print(f"  Root Mean Squared Error (RMSE): ${train_rmse:,.2f}")
print(f"  R² Score: {train_r2:.4f}")

# Calculate metrics for test set
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_test_pred)

print("\nTest Set Metrics:")
print(f"  Mean Squared Error (MSE): ${test_mse:,.2f}")
print(f"  Root Mean Squared Error (RMSE): ${test_rmse:,.2f}")
print(f"  R² Score: {test_r2:.4f}")

# Visualization: Actual vs Predicted
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred, alpha=0.6, color='blue', label='Training')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel("Actual Price ($)", fontsize=10)
plt.ylabel("Predicted Price ($)", fontsize=10)
plt.title("Training Set: Actual vs Predicted")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred, alpha=0.6, color='green', label='Testing')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel("Actual Price ($)", fontsize=10)
plt.ylabel("Predicted Price ($)", fontsize=10)
plt.title("Test Set: Actual vs Predicted")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/actual_vs_predicted.png', dpi=100, bbox_inches='tight')
print("\n✓ Visualization saved: visualizations/actual_vs_predicted.png")
plt.close()

# Step 9: Sample Predictions
print("\n[STEP 9] Sample Predictions on New Data")
print("-" * 60)

# Create sample houses to predict
sample_houses = pd.DataFrame({
    'area': [2000, 2500, 1500],
    'bedrooms': [3, 4, 2],
    'bathrooms': [2, 3, 1],
    'location_encoded': [0, 1, 2],  # Downtown, Suburb, Rural
    'age': [10, 5, 20],
    'parking': [2, 2, 1]
})

predictions = model.predict(sample_houses)

print("\nPredicting prices for sample houses:\n")
for i, (idx, row) in enumerate(sample_houses.iterrows()):
    location_map = {0: 'Downtown', 1: 'Suburb', 2: 'Rural'}
    print(f"House {i+1}:")
    print(f"  Area: {int(row['area'])} sq ft")
    print(f"  Bedrooms: {int(row['bedrooms'])}")
    print(f"  Bathrooms: {int(row['bathrooms'])}")
    print(f"  Location: {location_map[int(row['location_encoded'])]}")
    print(f"  Age: {int(row['age'])} years")
    print(f"  Parking: {int(row['parking'])} spaces")
    print(f"  🏠 Predicted Price: ${predictions[i]:,.2f}\n")

# Step 10: Summary and Insights
print("\n" + "="*60)
print("SUMMARY AND INSIGHTS")
print("="*60)

print("\n✓ Project Completion Checklist:")
print("  ✓ Dataset loaded and explored")
print("  ✓ Missing values checked")
print("  ✓ Data visualized and analyzed")
print("  ✓ Categorical variables encoded")
print("  ✓ Data split into train-test sets (80-20)")
print("  ✓ Linear Regression model trained")
print("  ✓ Model evaluated on both sets")
print("  ✓ Predictions made on new data")

print("\n📊 Model Performance:")
if test_r2 > 0.8:
    print("  → Excellent model performance! R² > 0.8")
elif test_r2 > 0.6:
    print("  → Good model performance. R² > 0.6")
else:
    print("  → Model has room for improvement. Consider using advanced models.")

print("\n💡 Next Steps for Improvement:")
print("  • Try polynomial regression or tree-based models (Random Forest, XGBoost)")
print("  • Include more features or create interaction features")
print("  • Use cross-validation for better evaluation")
print("  • Try feature scaling or normalization")
print("  • Collect more training data for better predictions")

print("\n" + "="*60)
print("Thank you for using the House Price Prediction App!")
print("="*60)
