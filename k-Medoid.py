import pandas as pd
from kmedoids import KMedoids
import matplotlib.pyplot as plt
df = pd.read_csv("Mall_Customers.csv")
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

kmedoids = KMedoids(n_clusters=5, metric='manhattan', random_state=42)
kmedoids.fit(X)

labels = kmedoids.labels_
medoids = kmedoids.cluster_centers_

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.scatter(medoids[:, 0], medoids[:, 1], c='black', marker='X', s=200, label='Medoids')
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using K-Medoids")
plt.legend()
plt.show()