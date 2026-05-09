import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =========================
# LOAD SENSOR DATA
# =========================

df = pd.read_csv("data/raw_sensor_stream.csv")

# Clean column names
df.columns = df.columns.str.strip()

# =========================
# SELECT FEATURES
# =========================

X = df[
    [
        "temperature",
        "humidity",
        "gas_level",
        "light_intensity"
    ]
]

# =========================
# NORMALIZATION
# =========================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# =========================
# APPLY PCA
# =========================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

# =========================
# CREATE PCA DATAFRAME
# =========================

pca_df = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"]
)

# =========================
# SAVE PCA OUTPUT
# =========================

pca_df.to_csv(
    "data/pca_output.csv",
    index=False
)

print("PCA completed successfully!\n")

# =========================
# EXPLAINED VARIANCE
# =========================

print("Explained Variance Ratio:")

print(pca.explained_variance_ratio_)

print("\n")

# =========================
# PCA COMPONENT ANALYSIS
# =========================

components_df = pd.DataFrame(
    pca.components_,
    columns=X.columns,
    index=["PC1", "PC2"]
)

print("PCA Components:\n")

print(components_df)

# =========================
# VISUALIZATION
# =========================

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1]
)

plt.xlabel("PC1")

plt.ylabel("PC2")

plt.title("PCA Visualization")

plt.grid(True)

plt.show()