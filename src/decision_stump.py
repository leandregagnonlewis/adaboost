import math
import numpy as np
from sklearn.datasets import make_blobs


def generate_dataset() -> (np.ndarray, np.ndarray):
    X, y = make_blobs(n_samples=10, centers=2, n_features=2, random_state=0, cluster_std=9)
    y = np.where(y == 0, -1, 1)
    return X, y


def sort_together_according_to_indice(X: np.ndarray, y: np.ndarray, D: np.ndarray, j: int) -> (np.ndarray, np.ndarray):
    ind = np.argsort(X[:, j])
    sorted_X = X[ind, :]
    sorted_y = y[ind]
    sorted_D = D[ind]

    return sorted_X, sorted_y, sorted_D


class DecisionStump:

    def __init__(self):
        self.fitted = False
        self.theta: float = 0
        self.b: float = 0
        self.j: int = -1

    def __call__(self, X: np.ndarray) -> float:
        if not self.fitted:
            raise RuntimeError("The model but be fitted first")

        temp = (X[:, self.j] - self.theta) * self.b
        temp = np.where(temp == 0, -1, temp)

        return np.sign(temp)

    def fit(self, X: np.ndarray, y: np.ndarray, D: np.ndarray):
        j_pos, theta_pos, F_pos = self._fit(X, y, D, b=1)
        j_neg, theta_neg, F_neg = self._fit(X, y, D, b=-1)

        if F_pos < F_neg:
            self.j = j_pos
            self.theta = theta_pos
            self.b = 1
            self.fitted = True

        else:
            self.b = -1
            self.j = j_neg
            self.theta = theta_neg
            self.fitted = True

    def _fit(self, X: np.ndarray, y: np.ndarray, D: np.ndarray, b=1) -> (int, float, float):
        j_star = None
        theta_star = None
        F_star = math.inf

        m, d = X.shape

        for j in range(d):
            sorted_X, sorted_y, sorted_D = sort_together_according_to_indice(X, y, D, j)
            F = np.sum(sorted_D[np.where(sorted_y == -b)])

            if F < F_star:
                F_star, j_star = F, j
                theta_star = sorted_X[0, j]-1

            for i in range(m-1):
                F = F + sorted_y[i] * sorted_D[i] * b
                if F < F_star and sorted_X[i, j] != sorted_X[i+1, j]:
                    F_star, j_star = F, j
                    theta_star = 0.5 * (sorted_X[i, j] + sorted_X[i+1, j])

        return j_star, theta_star, F_star

    def __eq__(self, o: object) -> bool:
        return isinstance(o, DecisionStump) and self.theta == o.theta and self.b == o.b

    def __hash__(self) -> int:
        return hash(frozenset((self.theta, self.b)))


if __name__ == '__main__':
    samples, labels = generate_dataset()
    proba = np.full_like(labels, 1 / samples.shape[0], dtype=np.float32)
    stump = DecisionStump()
    stump.fit(samples, labels, proba)
    print("Best stump is theta={} for j={}".format(stump.theta, stump.j))
    preds = stump(samples)
    accuracy = np.mean(np.equal(labels, preds))
    print(accuracy)



