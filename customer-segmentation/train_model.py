import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import joblib

np.random.seed(42)

print("🔄 Chargement des données...")
df = pd.read_csv("Mall_Customers.csv")

# Features
features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
X = df[features].copy()

# Scaling robuste
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Choix optimal de K
print("🔍 Recherche du meilleur K...")
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

best_k = K_range[np.argmax(silhouette_scores)]
print(f"🎯 Meilleur K: {best_k}")

# Modèle final
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Métriques
sil_score = silhouette_score(X_scaled, clusters)
ch_score = calinski_harabasz_score(X_scaled, clusters)

print(f"📊 Silhouette Score: {sil_score:.3f}")
print(f"📊 Calinski-Harabasz: {ch_score:.0f}")

# Ajout clusters
df["Cluster"] = clusters

# Sauvegarde
joblib.dump(kmeans, "kmeans_model.pkl")
joblib.dump(scaler, "scaler.pkl")
df.to_csv("customers_clustered.csv", index=False)

print("✅ Modèle sauvegardé !")