import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.compose import make_column_transformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_preparation import get_train_data

X, y = get_train_data()
data_transformer = make_column_transformer(
    ('Title', make_pipeline(CountVectorizer(), TfidfTransformer())),
    ('BodyMarkdown', make_pipeline(CountVectorizer(), TfidfTransformer())),
    ('Tags', make_pipeline(CountVectorizer(), TfidfTransformer())),
    (
        ['CountTags', 'ReputationAtPostCreation', 'OwnerUndeletedAnswerCountAtPostTime', 'OwnerCreationDate'],
        StandardScaler()
    ),
)
X = data_transformer.fit_transform(X)

colors_map = {
    'not a real question': 0.0,
    'not constructive': 0.25,
    'too localized': 0.5,
    'off topic': 0.75,
    'open': 1.0,
}
colors = [colors_map[c] for c in y]


def plot_2d():
    X_2d = TruncatedSVD(n_components=2).fit_transform(X)
    # Reorder the labels to have colors matching the cluster results
    plt.scatter(X_2d[:, 0], X_2d[:, 1], s=20, c=colors, cmap='rainbow')
    # plt.title(title)
    plt.show()


def plot_3d():
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()

    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=75, azim=135)

    plt.cla()
    X_3d = TruncatedSVD(n_components=3).fit_transform(X)

    # Reorder the labels to have colors matching the cluster results
    ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], s=10, c=colors, cmap="rainbow", edgecolor='k')
    ax.w_xaxis.set_ticklabels([])

    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

    # plt.title(title)
    plt.show()


if __name__ == '__main__':
    plot_2d()
    plot_3d()
