import numpy as np
from typing import List

from src.decision_stump import DecisionStump, generate_dataset
import math


class BoostedPredictor:
    def __init__(self, T: int):
        self.T: int = T
        self.W : np.ndarray = None
        self.weak_learners: List[DecisionStump] = None
        self.fitted = False

    def __call__(self, X: np.ndarray):
        if not self.fitted:
            raise RuntimeError("The model but be fitted first")

        preds = np.empty((X.shape[0], self.W.shape[0]))
        for i, (weak_learner, w) in enumerate(zip(self.weak_learners, self.W)):
            preds[:, i] = weak_learner(X) * w

        return np.sign(np.sum(preds, axis=1))

    def fit(self, X: np.ndarray, y: np.ndarray):
        m = y.size
        D = np.full_like(y, 1/m, dtype=np.float64)
        W = np.empty(self.T)
        weak_learners = list()

        for t in range(self.T):
            stump = DecisionStump()
            stump.fit(X, y, D)
            h = stump(X)
            e_t = np.sum(D[np.where(y != h)])
            if e_t == 0:  # no need to boost:
                W[t] = 1
            else:
                W[t] = 0.5 * math.log(1/e_t - 1)
            weak_learners.append(stump)
            temp = D * np.exp(-h * y * W[t])
            D = temp / np.sum(temp)

        self.weak_learners = weak_learners
        self.W = W
        self.fitted = True


if __name__ == '__main__':
    samples, labels = generate_dataset()
    boosted_stumps = BoostedPredictor(100)
    boosted_stumps.fit(samples, labels)
    preds = boosted_stumps(samples)
    accuracy = np.mean(np.equal(labels, preds))
    print(accuracy)
