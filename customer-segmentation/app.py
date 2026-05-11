import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Customer Segmentation PRO", layout="wide")

sns.set_palette("husl")

st.title("🛒 Customer Segmentation ")

# -------------------------
# LOAD
# -------------------------
@st.cache_data
def load():
    kmeans = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
    df = pd.read_csv("customers_clustered.csv")
    return kmeans, scaler, df

kmeans, scaler, df_data = load()

features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]

# -------------------------
# CLUSTER NAMES
# -------------------------
cluster_names = {
    0: "💎 High Value Customers",
    1: "🟡 Average Customers",
    2: "🔴 Low Engagement",
    3: "📈 Potential Customers",
    4: "👥 Regular Customers"
}

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.header("📂 Upload CSV")

uploaded_file = st.sidebar.file_uploader("Upload file", type=["csv"])

# -------------------------
# INPUT
# -------------------------
st.subheader("🎯 Predict Customer Segment")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 70, 30)
with col2:
    income = st.slider("Income", 15, 150, 60)
with col3:
    score = st.slider("Spending Score", 1, 100, 50)

if st.button("Predict"):

    data = pd.DataFrame([[age, income, score]], columns=features)
    scaled = scaler.transform(data)
    cluster = kmeans.predict(scaled)[0]

    label = cluster_names.get(cluster, f"Cluster {cluster}")

    st.success(f"### 🎉 {label}")

    # BUSINESS RECO
    st.subheader("💡 Recommendation")

    if cluster == 0:
        st.success("Loyalty program, VIP offers")
    elif cluster == 1:
        st.warning("Upselling strategies")
    else:
        st.error("Retention campaigns needed")

# -------------------------
# CSV UPLOAD
# -------------------------
if uploaded_file:
    df_upload = pd.read_csv(uploaded_file)

    st.subheader("📊 Uploaded Data")
    st.write(df_upload.head())

    if all(col in df_upload.columns for col in features):
        X = scaler.transform(df_upload[features])
        df_upload["Cluster"] = kmeans.predict(X)
        df_upload["Segment"] = df_upload["Cluster"].map(cluster_names)

        st.subheader("📌 Results")
        st.write(df_upload)

        csv = df_upload.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "results.csv")
    else:
        st.error("Missing required columns")

# -------------------------
# VISUALIZATION
# -------------------------
st.subheader("📊 Cluster Visualization")

fig, ax = plt.subplots()
sns.scatterplot(
    data=df_data,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    ax=ax
)
st.pyplot(fig)

# -------------------------
# PCA
# -------------------------
st.subheader("🧠 PCA Visualization")

X = scaler.transform(df_data[features])
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

df_data["PCA1"] = X_pca[:, 0]
df_data["PCA2"] = X_pca[:, 1]

fig2, ax2 = plt.subplots()
sns.scatterplot(data=df_data, x="PCA1", y="PCA2", hue="Cluster", ax=ax2)
st.pyplot(fig2)

# -------------------------
# MODEL ANALYSIS
# -------------------------
st.subheader("📉 Model Evaluation")

inertia = []
silhouette = []

for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X)
    inertia.append(km.inertia_)
    silhouette.append(silhouette_score(X, labels))

fig3, ax3 = plt.subplots(1, 2, figsize=(10,4))

ax3[0].plot(range(2,8), inertia, marker='o')
ax3[0].set_title("Elbow")

ax3[1].plot(range(2,8), silhouette, marker='o')
ax3[1].set_title("Silhouette")

st.pyplot(fig3)