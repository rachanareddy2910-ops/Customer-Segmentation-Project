# Customer Segmentation using K-Means Clustering

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# Load Dataset
df = pd.read_csv("Mall_Customers.csv")

# Display First 5 Rows
print("First 5 Rows:")
print(df.head())

# Dataset Information
print("\nDataset Info:")
print(df.info())

# Check Missing Values
print("\nMissing Values:")
print("Reached Elbow Method")
print(df.isnull().sum())

# Convert Gender to Numeric
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# Select Features for Clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Elbow Method to Find Optimal Clusters
wcss = []

for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        init='k-means++',
        random_state=42,
        n_init=10
    )
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()
print("Displaying Elbow Graph")
plt.savefig("graph.png")
print("Graph saved successfully")

# Apply K-Means
kmeans = KMeans(
    n_clusters=3,
    init='k-means++',
    random_state=42,
    n_init=10
)

y_kmeans = kmeans.fit_predict(X)

# Add Cluster Column
df['Cluster'] = y_kmeans

# Display Cluster Counts
print("\nCustomers in Each Cluster:")
print(df['Cluster'].value_counts())

# Cluster Centers
print("\nCluster Centers:")
print(kmeans.cluster_centers_)

# Visualization
plt.figure(figsize=(10,6))

sns.scatterplot(
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Cluster',
    palette='Set1',
    data=df,
    s=100
)

# Plot Centroids
plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    s=300,
    c='black',
    marker='X',
    label='Centroids'
)

plt.title('Customer Segmentation')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# Save Output
df.to_csv("Customer_Segmentation_Output.csv", index=False)

print("\nProject Completed Successfully!")
print("Output file saved as Customer_Segmentation_Output.csv")