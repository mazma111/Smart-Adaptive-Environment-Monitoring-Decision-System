import pandas as pd
import matplotlib.pyplot as plt

from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler

from pylab import bone, pcolor, colorbar, plot

# =========================
# LOAD PCA OUTPUT
# =========================

df = pd.read_csv("data/pca_output.csv")

# =========================
# NORMALIZE DATA
# =========================

scaler = MinMaxScaler()

data = scaler.fit_transform(df)

# =========================
# CREATE SOM
# =========================

som = MiniSom(
    x=2,
    y=2,
    input_len=2,
    sigma=1.0,
    learning_rate=0.5
)

# =========================
# INITIALIZE WEIGHTS
# =========================

som.random_weights_init(data)

# =========================
# TRAIN SOM
# =========================

som.train_random(
    data,
    100
)

print("SOM Training Completed!\n")

# =========================
# CLASSIFICATION
# =========================

clusters = []

states = []

for i, row in enumerate(data):

    winner = som.winner(row)

    clusters.append(winner)

    # -------------------------
    # Example State Mapping
    # -------------------------

    if winner == (0, 0):

        state = "Normal"

    elif winner == (0, 1):

        state = "Warning"

    else:

        state = "Dangerous"

    states.append(state)

    print(
        f"Sample {i} -> Cluster {winner} -> {state}"
    )

# =========================
# SAVE RESULTS
# =========================

results_df = pd.DataFrame(data, columns=["PC1", "PC2"])

results_df["Cluster"] = clusters

results_df["State"] = states

results_df.to_csv(
    "data/som_results.csv",
    index=False
)

print("\nResults saved successfully!")

# =========================
# SOM VISUALIZATION
# =========================

plt.figure(figsize=(7, 7))

bone()

pcolor(
    som.distance_map().T
)

colorbar()

for i, x in enumerate(data):

    w = som.winner(x)

    plot(
        w[0] + 0.5,
        w[1] + 0.5,
        'o',
        markerfacecolor='None',
        markeredgecolor='r',
        markersize=12,
        markeredgewidth=2
    )

plt.title("SOM Clustering Map")

plt.show()