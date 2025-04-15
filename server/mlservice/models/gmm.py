import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import pickle
import numpy as np

# Load the dataset
data = pd.read_csv("mydata.csv")  

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# PCA for dimensionality reduction
pca = PCA(n_components=2, svd_solver='full')
X_pca = pca.fit_transform(X_scaled)

# Train the Gaussian Mixture Model
model = GaussianMixture(n_components=3, random_state=42)
model.fit(X_pca)

# Predict clusters for the dataset
clusters = model.predict(X_pca)

# Analyze cluster means to assign labels
data_with_clusters = data.copy()
data_with_clusters['Cluster'] = clusters
cluster_means = data_with_clusters.groupby('Cluster').mean()

# Example logic to assign meaningful labels based on mean scores
cluster_labels = {}

for cluster_id, row in cluster_means.iterrows():
    avg_score = row.mean()
    if avg_score >= 7:
        label = "High Performer"
    elif avg_score >= 5:
        label = "Average Performer"
    else:
        label = "Low Performer"
    cluster_labels[cluster_id] = label

# Save model along with cluster labels
model_bundle = {
    "scaler": scaler,
    "pca": pca,
    "model": model,
    "cluster_labels": cluster_labels
}

with open("gmm.pkl", "wb") as f:
    pickle.dump(model_bundle, f)

print("Model saved as 'gmm.pkl'")
