import numpy as np
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

from clustering.DBSCAN import get_df


def train_KMeans():
    print('\n')
    print('Training KMeans..')
    sse = []
    for  k in range(1, 15):
        kmeans = KMeans(n_clusters=k).fit(x)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(x)
        curr_sse = 0

        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(len(x)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (x[i, 0] - curr_center[0]) ** 2 + (x[i, 1] - curr_center[1]) ** 2

        sse.append(curr_sse)
    plt.plot(sse)
    plt.savefig('kmeans_sse_plots.png', dpi=150)
    plt.clf()

    kmeans = KMeans(n_clusters=4).fit(x)

    print('Done training KMeans')
    return kmeans.labels_


def print_results(labels_print, x_print):
    print('\n')
    print('Results: ')
    n_clusters_ = len(set(labels_print)) - (1 if -1 in labels_print else 0)
    n_noise_ = list(labels_print).count(-1)
    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    print("Silhouette Coefficient: " + str(metrics.silhouette_score(x_print, labels_print)))
    print("Calinski-Harabasz score: " + str(metrics.calinski_harabasz_score(x_print, labels_print)))
    print("Davies-Bouldin score: " + str(metrics.davies_bouldin_score(x_print, labels_print)))


def plot_results(k):
    print('\n')
    colors = get_color()
    print_color_map(colors)
    plt.scatter(x[:, 0], x[:, 1], c=colors)
    plt.title("Plotting data frame clustering results - KMeans")
    plt.legend()
    file_name = 'KMeans_clustering_results' + str(k) + '.png'
    plt.savefig(file_name, dpi=150)
    plt.clf()


def get_color():
    colors = [
        'DarkCyan', 'royalblue', 'maroon', 'forestgreen', 'mediumorchid',
        'deeppink', 'olive', 'goldenrod', 'lightcyan', 'navy',
        'blue', 'yellow', 'lime', 'teal', 'aqua',
        'beige', 'tan', 'brown', 'pink', 'coral', 'firebrick', 'purple',
        'DarkSalmon', 'Crimson', 'Salmon', 'IndianRed', 'HotPink',
        'DeepPink', 'MediumVioletRed', 'PaleVioletRed', 'Orange', 'OrangeRed',
        'Gold', 'LightYellow', 'DarkKhaki', 'Lavender', 'Fuchsia',
        'BlueViolet', 'Indigo', 'DarkSlateBlue', 'SpringGreen', 'black'
    ]
    vectorizer = np.vectorize(lambda i: colors[i % len(colors)])
    colors = vectorizer(labels)
    return colors


def print_color_map(colors):
    c_set = set()
    for i in range(0, len(labels)):
        c_set.add((labels[i], 'Cluster: ' + str(labels[i]) + ' colored with: ' + str(colors[i])))
    for c in sorted(c_set):
        print(c)


def choose_perplexity_for_TSNE():
    df_tsne = pd.read_csv("../data/Clustering.csv", encoding="UTF-8")
    for i, p in enumerate([10, 20, 30, 40, 50, 60, 70, 80, 85, 90]):
        print('**************** ' + str(i) + " , " + str(p) + ' *********************')
        x_tsne = TSNE(n_components=2, perplexity=p).fit_transform(df_tsne)
        db_tsne = KMeans(n_clusters=5).fit(x_tsne)
        print_results(db_tsne.labels_, x_tsne)


if __name__ == '__main__':
    perplexity = 85
    best_k = 4
    x = get_df(perplexity)
    labels = train_KMeans()
    print_results(labels, x)
    plot_results(best_k)

