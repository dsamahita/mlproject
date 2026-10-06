# K-Means Clustering - Mushroom Dataset

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv(r"C:\Users\STUDENT\Downloads\11_mushroom_edibility.csv")

# Remove ID and Class
X = data.drop(["SampleID", "Class"], axis=1)

# Convert categorical data into numbers
encoder = LabelEncoder()

for column in X.columns:
    X[column] = encoder.fit_transform(X[column])

# Create K-Means model
kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

# Train K-Means
kmeans.fit(X)

# Get cluster labels
clusters = kmeans.labels_

# Add clusters to dataset
data["Cluster"] = clusters

# Display first 10 results
print(data.head(10))

# Display cluster counts
print("\nCluster Counts:")
print(data["Cluster"].value_counts())

# Plot clusters using first two features
plt.scatter(
    X.iloc[:, 0],
    X.iloc[:, 1],
    c=clusters
)

plt.xlabel("CapShape")
plt.ylabel("CapColor")
plt.title("K-Means Mushroom Clustering")
plt.show()