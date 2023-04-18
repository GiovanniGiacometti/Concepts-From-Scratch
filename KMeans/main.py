import numpy as np
from sklearn.datasets import make_blobs
from src import *
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

seed  = 24
np.random.seed(seed)


if __name__ == "__main__":
        X, _ = make_blobs(centers=3, n_samples=1500, random_state=seed)
        plt.scatter(X[:,0], X[:,1])
        plt.figure()

        #------------------

        X = StandardScaler().fit_transform(X)

        kmeans = KMeans(n_clusters = 3,init="kmeans++")

        cluster_prediction = kmeans.fit(X).predict()
        centroids = kmeans.centroids_

        plt.scatter(X[:, 0], X[:, 1], c=cluster_prediction)
        plt.scatter(centroids[:,0], centroids[:,1], c="r")

        #------------------

        fig, axs = plt.subplots(2,2)
        axs = axs.flatten()
        plt.tight_layout()

        for i in range(len(axs)):
                kmeans = KMeans(n_clusters = i+2,init="kmeans++")
                
                cluster_prediction = kmeans.fit(X).predict()
                centroids = kmeans.centroids_
                axs[i].scatter(X[:,0], X[:,1], c=cluster_prediction)
                axs[i].scatter(centroids[:,0], centroids[:,1], c="r")
                axs[i].set_title(f" K = {i+2}")


        plt.show()