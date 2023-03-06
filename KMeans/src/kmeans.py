import numpy as np

class KMeans:
    def __init__(self, n_clusters=5,init = "random", 
                            n_iter = 100,tol = 1e-4):

        self.n_clusters = n_clusters
        self.n_iter = n_iter #number of iterations the algorithm will run
        self.tol = tol #centroid updates threshold
        self.init = init #centroids initialization method
        
    def fit(self,X):
        if type(self.init) is np.ndarray:
            self.centroids_ = self.init
        elif "kmeans++" in self.init:
            self.centroids_ = self.kmeansplusplus(X)
        elif "random" in self.init: 
            centroids_indexes = np.random.randint(X.shape[0], size = self.n_clusters)
            self.centroids_ = X[centroids_indexes,:]
        else:
            raise ValueError("Invalid initialization method")
        
        self.labels_ = np.empty(X.shape[0])

        for _ in range(self.n_iter):

            for i in range(X.shape[0]):
                point = X[i,:]
                distances_to_centroids = [np.linalg.norm(point - centroid)         
                                    for centroid in self.centroids_]
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
        return self

    def predict(self):
        return self.labels_
    

    def kmeansplusplus(self, X):
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