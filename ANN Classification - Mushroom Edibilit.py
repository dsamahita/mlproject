# ANN Classification - Mushroom Edibility

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv(r"C:\Users\STUDENT\Downloads\11_mushroom_edibility.csv")

# Remove SampleID
data = data.drop("SampleID", axis=1)

# Separate input and output
X = data.drop("Class", axis=1)
y = data["Class"]

# Convert categorical data into numbers
encoder = LabelEncoder()

for column in X.columns:
    X[column] = encoder.fit_transform(X[column])

# Convert target into numbers
y = encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create ANN model
model = MLPClassifier(
    hidden_layer_sizes=(10,),
    max_iter=1000,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("ANN Accuracy:", accuracy)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))