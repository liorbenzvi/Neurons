import numpy as np
import pandas as pd
from sklearn import datasets, metrics
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

if __name__ == '__main__':
    df = pd.read_csv("../data/Clustering.csv", encoding="UTF-8")
    print('DF info: ')
    print(df.head())
    print("Dataset shape:", df.shape)
    print("Do we see Null values? - " + str(df.isnull().any().any()))
    x = StandardScaler().fit_transform(df)
    print(x)
    print('Amount of samples: ' + str(len(x)))

    print('find eps: ')
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(x)
    distances, indices = nbrs.kneighbors(x)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    plt.plot(distances)
    # plt.show()

    print('\n')
    print('Training DBSCAN..')
    db = DBSCAN(min_samples=5, eps=1.75).fit(x)
    # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    # core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    print('Done training DBSCAN')

    print('\n')
    print('Results: ')
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    print("Reduce dimension by PCA")
    values = PCA(n_components=1).fit_transform(x)
    sample_name = np.array(range(0, 2500))
    plt.scatter(sample_name, values)
    plt.title("Plotting data frame")
    plt.xlabel("X - index of samples")
    plt.ylabel("Y - values")
    plt.legend()
    # plt.show()



