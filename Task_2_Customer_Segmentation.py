# ---------------- IMPORT LIBRARIES ----------------
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ---------------- LOAD DATASET ----------------
data = pd.read_csv("Mall_Customers.csv")
print("First 5 rows:\n", data.head())

# ---------------- CHECK MISSING VALUES ----------------
print("\nMissing Values:\n", data.isnull().sum())

# ---------------- SELECT FEATURES ----------------
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# ---------------- SCALE DATA ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- FIND OPTIMAL NUMBER OF CLUSTERS (Elbow Method) ----------------
wcss = []   # within-cluster sum of squares
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method for Optimal K")
plt.show()

# ---------------- K-MEANS CLUSTERING ----------------
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X_scaled)

# Add cluster labels to dataset
data['Cluster'] = y_kmeans

# ---------------- VISUALIZE CLUSTERS ----------------
plt.figure(figsize=(8,6))
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=y_kmeans, cmap='rainbow', s=50)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
            s=200, c='black', marker='X', label='Centroids')
plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.title("Customer Segmentation using K-Means")
plt.legend()
plt.show()

# ---------------- CLUSTER ANALYSIS ----------------
cluster_summary = data.groupby('Cluster').agg({
    'CustomerID': 'count',
    'Annual Income (k$)': 'mean',
    'Spending Score (1-100)': 'mean'
}).rename(columns={
    'CustomerID': 'Num_Customers',
    'Annual Income (k$)': 'Avg_Income',
    'Spending Score (1-100)': 'Avg_SpendingScore'
})

print("\nCluster Summary:\n", cluster_summary)

# ---------------- BONUS: Silhouette Score ----------------
score = silhouette_score(X_scaled, y_kmeans)
print("\nSilhouette Score:", score)
