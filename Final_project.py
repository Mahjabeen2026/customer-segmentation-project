

# ============================================================
# FINAL PROJECT Q3: Customer Segmentation Using K-Means + PCA
# Problem Statement: Segment customers into meaningful groups
#                    using purchasing behavior.
# Files:
#   create_csv.py
#   Final_project.py 
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ------------------------------------------------------------
# 1. LOAD CUSTOMER TRANSACTION DATASET
# ------------------------------------------------------------
df = pd.read_csv('customer_transaction_data.csv')
print("Shape:", df.shape)  # (20, 7)
print(df)

# Purchasing Behavior Features
features = ['Age', 'AnnualIncome', 'TotalSpent',
            'NumTransactions', 'AvgTransactionValue', 'RecencyDays']
X = df[features]

# ------------------------------------------------------------
# 2. HANDLE MISSING & INFINITE VALUES
# ------------------------------------------------------------
X = X.replace([np.inf, -np.inf], np.nan)
print("\nMissing Values Before:", X.isnull().sum().sum())
X = X.fillna(X.median())
print("Missing Values After:", X.isnull().sum().sum())

# ------------------------------------------------------------
# 3. FEATURE SCALING (MANDATORY)
# ------------------------------------------------
# Age 20-60 but  Income 20000-90000 so scaling is necessary otherwise income will dominate the distance calculation.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------------
# 5. ELBOW METHOD TO SELECT K
# ------------------------------------------------------------
inertia = []
K_range = range(1, 11)
for k in K_range:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    inertia.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o', color='blue')
plt.title('Elbow Method - Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (WCSS)')
plt.grid(True)
plt.savefig('elbow_plot.png')
plt.show()

# Elbow K=4 ,so we select k= 4
OPTIMAL_K = 4

# ------------------------------------------------------------
# 4. APPLY K-MEANS CLUSTERING
# ------------------------------------------------------------
kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print("\nCustomers per Cluster:")
print(df['Cluster'].value_counts().sort_index())

# ------------------------------------------------------------
# 6. REDUCE DIMENSION USING PCA
# ------------------------------------------------------------
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

print("\nVariance captured by each PCA component:")
print(np.round(pca.explained_variance_ratio_ * 100, 2))
print("Total Variance:", round(pca.explained_variance_ratio_.sum() * 100, 2))

# ------------------------------------------------------------
# 7. VISUALISE CLUSTERS IN 2D
# ------------------------------------------------------------
plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green', 'purple']
for cluster in range(OPTIMAL_K):
    cluster_data = df[df['Cluster'] == cluster]
    plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'],
                color=colors[cluster], s=100, label=f'Cluster {cluster}')

plt.title('Customer Segments - PCA (2D) - Purchasing Behavior')
plt.xlabel(f"PCA1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PCA2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.legend()
plt.grid(True)
plt.savefig('segments.png')
plt.show()

# ------------------------------------------------------------
# 8. BUSINESS MEANING OF EACH CLUSTER
# ------------------------------------------------------------
summary = df.groupby('Cluster')[features].mean().round(1)
print("\nCluster Summary (Average Values):")
print(summary)
print("\nFull Result:")
print(df[['CustomerID', 'TotalSpent', 'NumTransactions', 'AvgTransactionValue', 'RecencyDays', 'Cluster', 'PCA1', 'PCA2']])

# Save Final Output
df.to_csv('customer_segmented_output.csv', index=False)
print("\nSaved: customer_segmented_output.csv")