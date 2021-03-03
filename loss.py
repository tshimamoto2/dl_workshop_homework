import numpy as np

class CrossEntropyError:
    def __init__(self):
        # logの値をinfにしないための微小値。
        self.delta = 1.0e-7

    def loss(self, y, t):
        if y.ndim == 1:
            y = y.reshape(1, -1)
            t = t.reshape(1, -1)

        logy = np.log(y + self.delta)
        E = -1.0 * np.sum(t * logy) / y.shape[0]
        # E = np.sum(np.diag(-1.0 * np.dot(t, logy.T))) / batch_size  #行列積版。計算量が多いのでボツ。
        return E

class MeanSquaredError:
    def __init__(self):
        pass

    def loss(self, y, t):
        if y.ndim == 1:
            y = y.reshape(1, -1)
            t = t.reshape(1, -1)

        diff = y - t
        E = 0.5 * np.sum(np.sum(diff ** 2)) / y.shape[0]
        # E = 0.5 * np.sum(np.diag(np.dot(diff, diff.T))) / batch_size  #行列積版。計算量が多いのでボツ。
        return E
