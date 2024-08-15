import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Sample data: Replace this with your abstracts
abstracts = [
    "Deep learning methods for computer vision",
    "Quantum computing for chemical simulations",
    "Machine learning techniques in healthcare",
    "Applications of blockchain technology in finance",
    "Gene editing using CRISPR technology"
]

# Step 1: Vectorize the abstracts using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(abstracts)

# Step 2: Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

# Step 3: Dimensionality Reduction for Visualization
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(X.toarray())

# # Step 4: Plot the clusters
# plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis')
# plt.title('K-Means Clustering of Abstracts')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.show()

print(X)
