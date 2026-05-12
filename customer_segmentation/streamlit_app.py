import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Customer Segmentation", page_icon="🛒")

st.title("🛒 Customer Segmentation")
st.caption("Prodigy Infotech — Task 02 · K-Means Clustering")

st.divider()

# ── Load or generate data ──────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'Mall_Customers.csv')

def generate_sample_data(n=200, seed=42):
    """Synthetic customer data similar to Kaggle Mall Customers dataset."""
    np.random.seed(seed)
    customer_id = range(1, n + 1)
    gender      = np.random.choice(['Male', 'Female'], n)
    age         = np.random.randint(18, 70, n)
    income      = np.random.randint(15, 140, n)   # Annual Income (k$)
    score        = np.random.randint(1, 101, n)    # Spending Score (1-100)
    return pd.DataFrame({
        'CustomerID':      customer_id,
        'Gender':          gender,
        'Age':             age,
        'Annual Income (k$)':  income,
        'Spending Score (1-100)': score,
    })

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    else:
        return generate_sample_data()

df = load_data()

# ── Dataset info ───────────────────────────────────────────────────────────────
st.subheader("📋 Dataset Overview")

using_real = os.path.exists(DATA_PATH)
if using_real:
    st.success("✅ Using real Kaggle dataset (Mall_Customers.csv)")
else:
    st.info("ℹ️ Kaggle dataset not found — using synthetic data. Place `Mall_Customers.csv` inside the `data/` folder to use real data.")

col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", len(df))
col2.metric("Features",        df.shape[1])
col3.metric("Avg. Spending Score", f"{df['Spending Score (1-100)'].mean():.1f}")

with st.expander("Show raw data"):
    st.dataframe(df, use_container_width=True)

st.divider()

# ── Feature selection ──────────────────────────────────────────────────────────
st.subheader("⚙️ Clustering Settings")

feature_options = {
    "Annual Income & Spending Score":       ["Annual Income (k$)", "Spending Score (1-100)"],
    "Age & Spending Score":                 ["Age", "Spending Score (1-100)"],
    "Age & Annual Income":                  ["Age", "Annual Income (k$)"],
    "Age, Annual Income & Spending Score":  ["Age", "Annual Income (k$)", "Spending Score (1-100)"],
}

selected_feature_label = st.selectbox("Choose features to cluster on:", list(feature_options.keys()))
features = feature_options[selected_feature_label]

n_clusters = st.slider("Number of Clusters (K)", min_value=2, max_value=10, value=5)

st.divider()

# ── Run clustering ─────────────────────────────────────────────────────────────
X = df[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

sil_score = silhouette_score(X_scaled, df['Cluster'])

st.subheader("📊 Results")

c1, c2, c3 = st.columns(3)
c1.metric("Clusters Found",    n_clusters)
c2.metric("Silhouette Score",  f"{sil_score:.3f}", help="Closer to 1.0 = better separated clusters")
c3.metric("Inertia",           f"{kmeans.inertia_:,.0f}", help="Lower = tighter clusters")

st.divider()

# ── Elbow chart ────────────────────────────────────────────────────────────────
st.subheader("📈 Elbow Method — Finding Best K")
st.write("The 'elbow' point in the graph is the ideal number of clusters.")

inertias = []
k_range = range(1, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

fig1, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(list(k_range), inertias, marker='o', color='steelblue', linewidth=2, markersize=7)
ax1.axvline(x=n_clusters, color='red', linestyle='--', alpha=0.6, label=f'Selected K={n_clusters}')
ax1.set_xlabel("Number of Clusters (K)")
ax1.set_ylabel("Inertia")
ax1.set_title("Elbow Method")
ax1.legend()
ax1.grid(True, alpha=0.3)
st.pyplot(fig1)

st.divider()

# ── Cluster scatter plot ───────────────────────────────────────────────────────
st.subheader("🔵 Cluster Visualization")

# Always plot first 2 features on X/Y
x_feat = features[0]
y_feat = features[1]

fig2, ax2 = plt.subplots(figsize=(8, 5))
colors = plt.cm.tab10.colors

for cluster_id in range(n_clusters):
    cluster_data = df[df['Cluster'] == cluster_id]
    ax2.scatter(
        cluster_data[x_feat],
        cluster_data[y_feat],
        label=f'Cluster {cluster_id + 1}',
        color=colors[cluster_id % 10],
        alpha=0.7,
        s=60
    )

# Plot centroids (inverse transform back to original scale)
centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
ax2.scatter(
    centers_original[:, 0],
    centers_original[:, 1],
    color='black', marker='X', s=200, zorder=5, label='Centroids'
)

ax2.set_xlabel(x_feat)
ax2.set_ylabel(y_feat)
ax2.set_title(f"Customer Clusters — {x_feat} vs {y_feat}")
ax2.legend()
ax2.grid(True, alpha=0.3)
st.pyplot(fig2)

st.divider()

# ── Cluster summary ────────────────────────────────────────────────────────────
st.subheader("📋 Cluster Summary")

summary = df.groupby('Cluster')[features].mean().round(1)
summary.index = [f"Cluster {i+1}" for i in summary.index]
summary['Customer Count'] = df.groupby('Cluster').size().values
st.dataframe(summary, use_container_width=True)

st.divider()

# ── Predict new customer ───────────────────────────────────────────────────────
st.subheader("🔍 Predict Segment for a New Customer")
st.write("Enter customer details to find which cluster they belong to.")

input_vals = []
cols = st.columns(len(features))
for i, feat in enumerate(features):
    if feat == "Age":
        val = cols[i].number_input(feat, min_value=18, max_value=100, value=35)
    elif feat == "Annual Income (k$)":
        val = cols[i].number_input(feat, min_value=10, max_value=200, value=60)
    else:
        val = cols[i].number_input(feat, min_value=1, max_value=100, value=50)
    input_vals.append(val)

if st.button("Find My Cluster", type="primary", use_container_width=True):
    new_customer = np.array([input_vals])
    new_scaled   = scaler.transform(new_customer)
    predicted    = kmeans.predict(new_scaled)[0]

    st.success(f"✅ This customer belongs to **Cluster {predicted + 1}**")

    cluster_info = summary.loc[f"Cluster {predicted + 1}"]
    st.write("**Average profile of this cluster:**")
    st.dataframe(cluster_info.to_frame(name="Average Value"), use_container_width=True)
