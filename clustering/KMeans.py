import numpy as np
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt


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
    x = TSNE(n_components=2, perplexity=perplexity).fit_transform(df)
    return x


def train_KMeans():
    print('\n')
    print('Training KMeans..')
    db = KMeans(n_clusters=5).fit(x)
    print('Done training KMeans')
    return db.labels_


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


def plot_results():
    print('\n')
    colors = get_color()
    print_color_map(colors)
    plt.scatter(x[:, 0], x[:, 1], c=colors)
    plt.title("Plotting data frame clustering results - KMeans")
    plt.legend()
    plt.savefig('KMeans_clustering_results.png', dpi=150)
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


def choose_parameters():
    print('******************************** choose_parameters ********************************')
    find_eps_by_knn()
    choose_perplexity_for_TSNE()
    print('******************************** choose_parameters ********************************')


if __name__ == '__main__':
    eps = 1.4
    perplexity = 85

    x = get_df()
    choose_parameters()
    labels = train_KMeans()
    print_results(labels, x)
    plot_results()

