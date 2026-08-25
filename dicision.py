# ============================================================
# MUSHROOM EDIBILITY DATASET ANALYSIS
# Decision Tree, Linear Regression, Polynomial Regression,
# Multivariate Regression and Normalization
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score
)

# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 60)
print("MUSHROOM EDIBILITY DATASET")
print("=" * 60)

# Change the filename/path if required
file_path = "11_mushroom_edibility.csv"

df = pd.read_csv(file_path)

print("\nDataset loaded successfully!")
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 2. DATA PREPROCESSING
# ============================================================

print("\n" + "=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

# Remove SampleID because it is only an identifier
if "SampleID" in df.columns:
    df = df.drop("SampleID", axis=1)

# Target column
target_column = "Class"

# Separate features and target
X = df.drop(target_column, axis=1)
y = df[target_column]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print(target_column)


# ============================================================
# 3. LABEL ENCODING
# Convert categorical data into numerical values
# ============================================================

label_encoders = {}

for column in X.columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# Encode target variable
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

print("\nTarget classes:")
for i, class_name in enumerate(target_encoder.classes_):
    print(f"{class_name} = {i}")

print("\nEncoded dataset:")
print(X.head())


# ============================================================
# 4. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ============================================================
# 5. NORMALIZATION
# StandardScaler normalizes data using mean and standard deviation
# ============================================================

print("\n" + "=" * 60)
print("NORMALIZATION")
print("=" * 60)

scaler = StandardScaler()

X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

print("\nData normalization completed.")

print("\nNormalized training data:")
print(X_train_normalized[:5])


# ============================================================
# 6. DECISION TREE CLASSIFIER
# ============================================================

print("\n" + "=" * 60)
print("DECISION TREE CLASSIFIER")
print("=" * 60)

decision_tree = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=10,
    random_state=42
)

decision_tree.fit(X_train, y_train)

y_pred_tree = decision_tree.predict(X_test)

tree_accuracy = accuracy_score(y_test, y_pred_tree)

print("\nDecision Tree Accuracy:")
print(f"{tree_accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_tree))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_tree,
        target_names=target_encoder.classes_
    )
)


# ============================================================
# 7. LINEAR REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("LINEAR REGRESSION")
print("=" * 60)

linear_model = LinearRegression()

linear_model.fit(
    X_train_normalized,
    y_train
)

y_pred_linear = linear_model.predict(X_test_normalized)

# Convert regression output into binary classes
y_pred_linear_class = np.where(y_pred_linear >= 0.5, 1, 0)

linear_accuracy = accuracy_score(
    y_test,
    y_pred_linear_class
)

linear_mse = mean_squared_error(
    y_test,
    y_pred_linear
)

linear_r2 = r2_score(
    y_test,
    y_pred_linear
)

print("\nLinear Regression Accuracy:")
print(f"{linear_accuracy * 100:.2f}%")

print("Mean Squared Error:", linear_mse)
print("R² Score:", linear_r2)


# ============================================================
# 8. POLYNOMIAL REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("POLYNOMIAL REGRESSION")
print("=" * 60)

# Degree 2 polynomial features
polynomial = PolynomialFeatures(
    degree=2,
    include_bias=False
)

X_train_poly = polynomial.fit_transform(
    X_train_normalized
)

X_test_poly = polynomial.transform(
    X_test_normalized
)

poly_model = LinearRegression()

poly_model.fit(
    X_train_poly,
    y_train
)

y_pred_poly = poly_model.predict(
    X_test_poly
)

# Convert continuous predictions to class values
y_pred_poly_class = np.where(
    y_pred_poly >= 0.5,
    1,
    0
)

poly_accuracy = accuracy_score(
    y_test,
    y_pred_poly_class
)

poly_mse = mean_squared_error(
    y_test,
    y_pred_poly
)

poly_r2 = r2_score(
    y_test,
    y_pred_poly
)

print("\nPolynomial Regression Accuracy:")
print(f"{poly_accuracy * 100:.2f}%")

print("Mean Squared Error:", poly_mse)
print("R² Score:", poly_r2)


# ============================================================
# 9. MULTIVARIATE LINEAR REGRESSION
# Uses all independent variables
# ============================================================

print("\n" + "=" * 60)
print("MULTIVARIATE LINEAR REGRESSION")
print("=" * 60)

multivariate_model = LinearRegression()

multivariate_model.fit(
    X_train_normalized,
    y_train
)

y_pred_multi = multivariate_model.predict(
    X_test_normalized
)

# Convert predicted values into classes
y_pred_multi_class = np.where(
    y_pred_multi >= 0.5,
    1,
    0
)

multi_accuracy = accuracy_score(
    y_test,
    y_pred_multi_class
)

multi_mse = mean_squared_error(
    y_test,
    y_pred_multi
)

multi_r2 = r2_score(
    y_test,
    y_pred_multi
)

print("\nMultivariate Regression Accuracy:")
print(f"{multi_accuracy * 100:.2f}%")

print("Mean Squared Error:", multi_mse)
print("R² Score:", multi_r2)


# ============================================================
# 10. FEATURE IMPORTANCE FROM DECISION TREE
# ============================================================

print("\n" + "=" * 60)
print("DECISION TREE FEATURE IMPORTANCE")
print("=" * 60)

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": decision_tree.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance)


# ============================================================
# 11. FINAL MODEL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

results = pd.DataFrame({
    "Model": [
        "Decision Tree",
        "Linear Regression",
        "Polynomial Regression",
        "Multivariate Regression"
    ],
    "Accuracy (%)": [
        tree_accuracy * 100,
        linear_accuracy * 100,
        poly_accuracy * 100,
        multi_accuracy * 100
    ]
})

print(results.to_string(index=False))


# ============================================================
# 12. BEST MODEL
# ============================================================

best_model = results.loc[
    results["Accuracy (%)"].idxmax()
]

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print("Model:", best_model["Model"])
print(f"Accuracy: {best_model['Accuracy (%)']:.2f}%")

print("\nProgram completed successfully!")