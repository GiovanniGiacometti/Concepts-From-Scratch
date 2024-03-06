import numpy as np
from enum import Enum
from typing import Optional


class KMeansInitMethod(Enum):
    RANDOM = "random"
    KMEANSPLUSPLUS = "kmeans++"
    CENTROIDS = "centroids"


class KMeans:
    def __init__(self, 
                 n_clusters: int, 
                 init: KMeansInitMethod, 
                 centroids: Optional[np.ndarray] = None, 
                 n_iter: int = 100, 
                 seed: Optional[int] = None,
                 tol: float = 1e-4):

        self.n_clusters = n_clusters

        # number of iterations the algorithm will run
        # if the centroids do not converge
        self.n_iter = n_iter  

        # centroid updates threshold
        self.tol = tol

        if centroids is not None:
            if init != KMeansInitMethod.CENTROIDS:
                raise ValueError("Centroids were provided but init is not 'centroids'")
            
        if seed is not None:
            np.random.seed(seed)
        
        self.centroids_ = centroids
        self.init_method = init

        self.fitted = False
        
    def fit(self, X) -> "KMeans":
        
        if len(X.shape) < 2:
            raise ValueError("X must have at least 2 dimensions")

        if self.init_method == KMeansInitMethod.CENTROIDS:
            self.centroids_ = self.init

        elif self.init_method == KMeansInitMethod.KMEANSPLUSPLUS:
            self.centroids_ = self._kmeansplusplus(X)

        elif self.init_method == KMeansInitMethod.RANDOM: 
            centroids_indexes = np.random.randint(X.shape[0], size = self.n_clusters)
            self.centroids_ = X[centroids_indexes,:]

        else:
            raise ValueError("Invalid initialization method")
        
        self.labels_ = np.empty(X.shape[0])

        for _ in range(self.n_iter):

            for i in range(X.shape[0]):

                point = X[i,:]
                distances_to_centroids = [np.linalg.norm(point - centroid) for centroid in self.centroids_]
                closest = np.argmin(distances_to_centroids)
                self.labels_[i] = closest

            new_centroids = []

            for cluster in range(self.n_clusters):

                cluster_points = X[self.labels_ == cluster]
                new_centroids.append(np.mean(cluster_points, axis = 0))
                
            np_new_centroids = np.array(new_centroids)
            
            if self.tol > np.linalg.norm(np_new_centroids - self.centroids_):               
                break
            else:
                self.centroids_ = np_new_centroids

        self.fitted = True
        return self

    def predict(self) -> np.ndarray:

        if not self.fitted:
            raise ValueError("The model has not been fitted yet")
        
        return self.labels_
    
    def _kmeansplusplus(self, X: np.ndarray) -> np.ndarray:

        possible_indexes = list(range(X.shape[0]))

        first_centroid_index = np.random.choice(possible_indexes)

        possible_indexes.remove(first_centroid_index)
        centroids = [X[first_centroid_index,:]]

        for _ in range(self.n_clusters - 1):

            distances = []
            for i in possible_indexes:
                point = X[i,:]
                distances_to_centroids = [np.linalg.norm(point - centroid) for centroid in centroids]
                distances.append(min(distances_to_centroids))

            distances /= sum(distances)
            next_centroid_index = np.random.choice(possible_indexes,p=distances)
            possible_indexes.remove(next_centroid_index)
            centroids.append(X[next_centroid_index,:])

        return centroids