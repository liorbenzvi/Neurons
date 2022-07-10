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


def find_eps_by_knn():
    print('Find eps by knn: ')
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(x)
    distances, indices = nbrs.kneighbors(x)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    plt.plot(distances)
    plt.savefig('distances.png', dpi=150)
    plt.clf()


def get_df():
    df = pd.read_csv("../data/Clustering.csv", encoding="UTF-8")
    print('DF info: ')
    print(df.head())
    print("Dataset shape:", df.shape)
    print("Do we see Null values? - " + str(df.isnull().any().any()))
    x = StandardScaler().fit_transform(df)
    return x


def train_dbscan():
    print('\n')
    print('Training DBSCAN..')
    db = DBSCAN(min_samples=5, eps=1.75).fit(x)
    print('Done training DBSCAN')
    return db.labels_


def print_results():
    print('\n')
    print('Results: ')
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)


def plot_results():
    colors = [
              'black', 'royalblue', 'maroon', 'forestgreen', 'mediumorchid',
              'deeppink', 'olive', 'goldenrod', 'lightcyan', 'navy',
              'blue', 'yellow', 'lime', 'teal', 'aqua',
              'beige', 'tan', 'brown', 'pink', 'coral', 'firebrick', 'purple'
              ]
    vectorizer = np.vectorize(lambda i: colors[i % len(colors)])
    colors = vectorizer(labels)
    print('\n')
    print_color_map(colors)
    print("Reduce dimension by PCA")
    values = PCA(n_components=1).fit_transform(x)
    sample_name = np.array(range(0, 2500))
    plt.scatter(sample_name, values, c=colors)
    plt.title("Plotting data frame clustering results - DBSCAN")
    plt.xlabel("X - index of samples")
    plt.ylabel("Y - values")
    plt.legend()
    plt.savefig('dbscan_clustering_results.png', dpi=150)
    plt.clf()


def print_color_map(colors):
    c_set = set()
    for i in range(0, len(labels)):
        c_set.add((labels[i], 'Cluster: ' + str(labels[i]) + ' colored with: ' + str(colors[i])))
    for c in sorted(c_set):
        print(c)


if __name__ == '__main__':
    x = get_df()
    find_eps_by_knn()
    labels = train_dbscan()
    print_results()
    plot_results()
    # parameter_tuning()
    # print_results()
    # plot_results()



