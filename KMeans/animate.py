import numpy as np
from sklearn.datasets import make_blobs
from src import KMeans, Animator
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

seed  = 24
np.random.seed(seed)

if __name__ == "__main":

        X, _ = make_blobs(centers=2, n_samples=1500, random_state=seed)

        X = StandardScaler().fit_transform(X)

        indexes = np.array([1400,1000,40,13,33])

        params = {
                "n_clusters":5,
                "n_iter":1,
                # "init" : X[indexes,:]
                "init" : "random"
        }

        Animator(X=X, algorithm=KMeans, params=params, save = False, name = "animation.gif", fps=1).plot()