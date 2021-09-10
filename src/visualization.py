import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons, make_blobs, make_circles

from src.adaboost import BoostedPredictor
from src.decision_stump import DecisionStump


def plot(X, y):
    fig, axs = plt.subplots(2)

    y = np.where(y == 0, -1, 1)
    proba = np.ones_like(y) / X.shape[0]
    stump = DecisionStump()
    stump.fit(X, y, proba)
    pred_stump = stump(X)

    boosted_stumps = BoostedPredictor(150)
    boosted_stumps.fit(X, y)
    pred_boost = boosted_stumps(X)

    accuracy_stump = np.mean(np.equal(y, pred_stump))
    accuracy_boost = np.mean(np.equal(y, pred_boost))

    # create a mesh to plot in
    h = .02

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    Z_stump = stump(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z_stump = Z_stump.reshape(xx.shape)
    cs = axs[0].contourf(xx, yy, Z_stump, cmap=plt.cm.Paired)
    axs[0].axis('tight')

    # Plot also the training points
    for i, color in zip((-1, 1), "br"):
        idx = np.where(y == i)
        axs[0].scatter(X[idx, 0], X[idx, 1], c=color, label=i,
                    cmap=plt.cm.Paired, edgecolor='black', s=20)
    axs[0].set_title("Decision surface best decision stump Accuracy = {:.2f} %".format(accuracy_stump * 100))


    axs[0].legend()

    Z_boost = boosted_stumps(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z_boost = Z_boost.reshape(xx.shape)
    cs = axs[1].contourf(xx, yy, Z_boost, cmap=plt.cm.Paired)

    # Plot also the training points
    for i, color in zip((-1, 1), "br"):
        idx = np.where(y == i)
        axs[1].scatter(X[idx, 0], X[idx, 1], c=color, label=i,
                       cmap=plt.cm.Paired, edgecolor='black', s=20)
    axs[1].set_title("Decision surface adaboost Accuracy = {:.2f} %".format(accuracy_boost * 100))

    axs[1].legend()

    plt.tight_layout()


if __name__ == "__main__":
    for i in range(3):
        plot(*make_circles(n_samples=100, random_state=i))
        plot(*make_blobs(n_samples=30, centers=2, n_features=2, random_state=i))
        plot(*make_moons(n_samples=100, random_state=i))
    plt.show()

