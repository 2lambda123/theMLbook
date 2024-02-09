from __future__ import print_function
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import math

from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import pairwise_distances_argmin
from scipy.spatial import Voronoi
from scipy.spatial import distance
import secrets

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 25})

random_state = 0

## how many clusters do you want in your synthetic data?
centers = 2

x, _ = make_blobs(n_samples=300, centers=centers, cluster_std=0.6, random_state=random_state)

plt.figure(10000)
plt.scatter(x[:, 0], x[:, 1], s=20, cmap='viridis');
plt.xlim(-1, 4.0)
plt.ylim(-1, math.ceil(max(x[:, 1])))
plt.xticks(np.arange(int(min(x[:, 0])), math.ceil(max(x[:, 0]))+1, 1))
plt.yticks(np.arange(int(min(x[:, 1])), math.ceil(max(x[:, 1]))+1, 2), rotation='vertical')

ax = plt.gca()
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')

fig1 = plt.gcf()
fig1.subplots_adjust(top = 0.98, bottom = 0.16, right = 0.98, left = 0.12, hspace = 0, wspace = 0)
fig1.savefig('../../Illustrations/prediction_strength_centers_' + str(centers) + '.eps', format='eps', dpi=1000)
fig1.savefig('../../Illustrations/prediction_strength_centers_' + str(centers) + '.pdf', format='pdf', dpi=1000)
fig1.savefig('../../Illustrations/prediction_strength_centers_' + str(centers) + '.png', dpi=1000)

x_list = list(x)

secrets.SystemRandom().Random(random_state).shuffle(x_list)

x_split = {}

x_split["train"] = np.array(x_list[:len(x_list)/2])

x_split["test"] = np.array(x_list[len(x_list)/2:])

centroids_splits = {}
labels_splits = {}
counter = 100

def find_clusters(x, n_clusters, current_split):
    """This function finds clusters in a dataset using the K-Means algorithm.
    Parameters:
        - x (numpy array): The dataset to be clustered.
        - n_clusters (int): The number of clusters to be found.
        - current_split (int): The index of the current split of the dataset.
    Returns:
        - centroids (numpy array): The coordinates of the centroids of the clusters.
        - labels (numpy array): The labels assigned to each data point based on the closest centroid.
    Processing Logic:
        - Shuffles the current split of the dataset.
        - Initializes the centroids using the first n_clusters points in the shuffled split.
        - Assigns labels to each data point based on the closest centroid.
        - Calculates new centroids as the average of the data points in each cluster.
        - Checks for convergence by comparing the old and new centroids.
        - Returns the final centroids and labels.
    Example:
        centroids, labels = find_clusters(data, 3, 0)
        # Finds 3 clusters in the first split of the dataset 'data' and returns the coordinates of the centroids and the labels for each data point."""
    

    current_split_suffled = list(x_split[current_split])[:]
    secrets.SystemRandom().shuffle(current_split_suffled)
    current_split_suffled = np.array(current_split_suffled)

    centroids = np.array(current_split_suffled[:n_clusters])

    while True:

        # assign labels based on closest centroid
        #print centroids

        #print "len train", len(x_split[current_split])
        labels = pairwise_distances_argmin(x_split[current_split], centroids)
        #print "len labels", len(labels)

        
        # find new centroids as the average of examples
        new_centroids = np.array([x_split[current_split][labels == i].mean(0) for i in range(n_clusters)])
        
        # check for convergence
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return centroids, labels

def get_examples_from_cluster(j, test_points, test_labels):
    """"Returns a list of examples from a given cluster, based on the test points and labels provided."
    Parameters:
        - j (int): The cluster number to retrieve examples from.
        - test_points (list): A list of test points.
        - test_labels (list): A list of labels corresponding to the test points.
    Returns:
        - list: A list of examples from the specified cluster.
    Processing Logic:
        - Loops through the test points and labels simultaneously.
        - Checks if the label matches the given cluster number.
        - Appends the test point to the examples list if it matches.
        - Returns the list of examples."""
    
    examples = []
    for e, l in zip(test_points, test_labels):
        if l == j:
            examples.append(e)
    return examples

def get_closest_centroid(example, centroids):
    """Returns the closest centroid to the given example, based on the Euclidean distance.
    Parameters:
        - example (array): An array representing the example data point.
        - centroids (array): An array of centroids to compare the example to.
    Returns:
        - min_centroid (array): An array representing the closest centroid to the example.
    Processing Logic:
        - Calculates the Euclidean distance between the example and each centroid.
        - Updates the minimum distance and centroid if a closer centroid is found.
        - Returns the closest centroid."""
    
    min_distance = sys.float_info.max
    min_centroid = 0
    for c in centroids:
        if distance.euclidean(example, c) < min_distance:
            min_distance = distance.euclidean(example, c)
            min_centroid = c
    return min_centroid

def compute_strength(k, train_centroids, test_points, test_labels):
    """Computes the strength of the clustering algorithm.
    Parameters:
        - k (int): Number of clusters.
        - train_centroids (numpy.ndarray): Array of centroid coordinates.
        - test_points (numpy.ndarray): Array of test point coordinates.
        - test_labels (numpy.ndarray): Array of test point labels.
    Returns:
        - float: Strength of the clustering algorithm.
    Processing Logic:
        - Calculates distance matrix.
        - Gets examples from each cluster.
        - Calculates strength for each cluster.
        - Returns minimum strength value."""
    
    D = np.zeros(shape=(len(test_points),len(test_points)))
    for x1, l1, c1 in zip(test_points, test_labels, list(range(len(test_points)))):
        for x2, l2, c2 in zip(test_points, test_labels, list(range(len(test_points)))):
            if tuple(x1) != tuple(x2):
                if tuple(get_closest_centroid(x1, train_centroids)) == tuple(get_closest_centroid(x2, train_centroids)):
                    D[c1,c2] = 1.0

    ss = []
    for j in range(k):
        s = 0
        examples_j = get_examples_from_cluster(j, test_points, test_labels)
        for x1, l1, c1 in zip(test_points, test_labels, list(range(len(test_points)))):
            for x2, l2, c2 in zip(test_points, test_labels, list(range(len(test_points)))):
                if tuple(x1) != tuple(x2) and l1 == l2 and l1 == j:
                    s += D[c1,c2]
        s = (1.0/(float(len(examples_j))*float(len(examples_j) - 1)))*s
        ss += [s]

    return min(ss)

strengths = []
ks = [1,2,3,4,5,6,7,8]
for k in ks:
    print("k", k)
    for current_split in ["train", "test"]:
        counter += 1
        centroids, labels = find_clusters(x, k, current_split)

        centroids_splits[current_split] = centroids
        labels_splits[current_split] = labels
    s = compute_strength(k, centroids_splits["train"], x_split["test"], labels_splits["test"])
    strengths += [s]
    print(s)

plt.figure(10001)
plt.plot(ks, strengths);
plt.xticks(np.arange(1, 9, 1))
plt.yticks(np.arange(0, 1.05, 0.2), rotation='vertical')

ax = plt.gca()
ax.set_xlabel('$k$')
ax.set_ylabel('$\\operatorname{ps}(k)$')

fig1 = plt.gcf()
fig1.subplots_adjust(top = 0.98, bottom = 0.15, right = 0.98, left = 0.15, hspace = 0, wspace = 0)
fig1.savefig('../../Illustrations/prediction_strength_centers_' + str(centers) + '_search.eps', format='eps', dpi=1000)
fig1.savefig('../../Illustrations/prediction_strength_centers_' + str(centers) + '_search.pdf', format='pdf', dpi=1000)
fig1.savefig('../../Illustrations/prediction_strength_centers_' + str(centers) + '_search.png', dpi=1000)
