"""
This script can be used to run the parameterization of  standalone.

Usage:
    ...

Output:
    ...

Author:
    Nils Müller
    nilmu@dtu.dk
"""

import pandas as pd
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.metrics import dtw as dtw_metric
from matplotlib import pyplot as plt
import numpy as np
import yaml

DailyFCRNPrices = pd.read_csv(r"/Data/DailyFCRNPrices.csv")
DailyFCRNPrices = DailyFCRNPrices.drop("index", axis=1)


# Transform to numpy array of shape (timeseries, length of timeseries, num of features) (required for TimeSeriesKMeans)
X_train_np = DailyFCRNPrices.values.T  # So each row is a time series and each column a time step of the series
X_train_np = X_train_np.reshape((X_train_np.shape[0], X_train_np.shape[1], 1))

# Scale the time series data
X_train_np_scaled = X_train_np  # TimeSeriesScalerMinMax().fit_transform(X_train_np)


### Run KMeans ###

n_rep_days = 12

# Set up model
KMeans_model = TimeSeriesKMeans(n_clusters=n_rep_days, metric="dtw", max_iter=100, n_jobs=16, n_init=1000, init="random", verbose=1) #, init='k-means++')

# Fit and predict for X_train_np
cluster_predictions = KMeans_model.fit_predict(X_train_np_scaled)

# Get the cluster centers
cluster_centers = KMeans_model.cluster_centers_

# Count the population of each cluster
cluster_population = np.bincount(cluster_predictions)

# Print or use the population of each cluster
for cluster_id, population in enumerate(cluster_population):
    print(f"Cluster {cluster_id}: Population = {population}")


# Get the closest day for each cluster
closest_samples = []

for cluster_id in range(n_rep_days):

    # Get idxs of all samples belonging to the current cluster
    cluster_indices = np.where(cluster_predictions == cluster_id)[0]

    # Get the cluster center
    cluster_center = cluster_centers[cluster_id]

    # Calculate distance between samples of a cluster and the center of the cluster
    dtw_distances = [dtw_metric(cluster_center, X_train_np_scaled[i]) for i in cluster_indices]

    # Find the index of the closest sample
    closest_index = cluster_indices[np.argmin(dtw_distances)]

    # Store the index of closest sample
    closest_samples.append(closest_index)

# Print or use the closest samples
for cluster_id, sample_index in enumerate(closest_samples):
    print(f"Cluster {cluster_id}: Closest sample index = {sample_index}")

# Access the actual time series data for the closest samples
Rep_days = DailyFCRNPrices.iloc[:, closest_samples]

print(Rep_days)

DailySpotPrices = pd.read_csv(r"/Data/DailySpotPrices.csv")
DailySpotPrices = DailySpotPrices.drop("index", axis=1)

rep_day_dict = {f"Day{idx}": {"Weighting": float(cluster_population[idx]/365), "FCRNPriceEUR": DailyFCRNPrices.iloc[:, closest_samples[idx]].tolist(), "SpotPriceEUR": DailySpotPrices.iloc[:, closest_samples[idx]].tolist()} for idx in range(n_rep_days)}

rep_day_dict = rep_day_dict.copy()
print(rep_day_dict)

# Write the dictionary to the YAML file
with open("RepDays.json", 'w') as file:
    yaml.dump(rep_day_dict, file, indent=4)


# Plot the clustering results
for cluster_i in range(n_rep_days):

    plt.subplot(1, n_rep_days, cluster_i+1)

    # Add all time series which are predicted to belong cluster_i to the same plot
    for ts_of_cluster_i in X_train_np_scaled[cluster_predictions == cluster_i]:
        plt.plot(ts_of_cluster_i.ravel(), "k-", alpha=.2)

    # Add the predicted cluster center to the plot
    plt.plot(KMeans_model.cluster_centers_[cluster_i].ravel(), "r-")

    plt.plot(X_train_np_scaled[closest_samples[cluster_i]].ravel(), "g-")

    # Add cluster name to plot
    plt.text(0.55, 0.85, 'Cluster %d' % (cluster_i + 1), transform=plt.gca().transAxes)

plt.show()