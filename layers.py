import numpy as np

class HiddenLayer():
    def __init__(self, W, B, actfunc):
        self.affine = Affine(W, B)

        # TODO 活性化関数を切り替えれるようにしたい。
        # Sigmoid：勾配消失問題が発生するためか、学習が進まず100回のエポックでも正解率が20％のままだった。
        # Tanh：
        # ReLU：学習が進み正解率が97％を超過した。
        self.activation = actfunc

    def forward(self, x, t):
        a = self.affine.forward(x, t)
        z = self.activation.forward(a, t)
        return z

    def backward(self, dout):
        dout = self.activation.backward(dout)
        dout = self.affine.backward(dout)
        return dout

class LastLayer():
    def __init__(self, W, B, actfunc):
        # super().__init__()
        self.affine = Affine(W, B)
        self.activation = actfunc

    def forward(self, x, t):
        a = self.affine.forward(x, t)
        z = self.activation.forward(a, t)
        return z

    # クロスエントロピー誤差を用いる場合は引数のdoutは使用されない
    def backward(self, dout):
        dout = self.activation.backward(dout)
        dout = self.affine.backward(dout)
        return dout

# ------------------------------
# 以下、各レイヤー内の個別レイヤーの実装。
# ※注意：ソフトマックスレイヤーだけ、forwardメソッドに教師データが必要。
# 個別レイヤーのforwardメソッドには不要だが造りを合わせておく。
# ------------------------------
class Affine():
    def __init__(self, W, B):
        self.W = W
        self.B = B
        self.dLdW = None
        self.dLdB = None
        self.x = None
        self.numerical_dLdW = None
        self.numerical_dLdB = None

    # tは未使用。
    def forward(self, x, t):
        self.x = x
        return np.dot(x, self.W) + self.B

    def backward(self, dout):
        self.dLdW = np.dot(self.x.T, dout)
        self.dLdB = np.sum(dout, axis=0)
        return np.dot(dout, self.W.T)

class Sigmoid():
    def __init__(self):
        self.out = None

    # tは未使用。
    def forward(self, x, t):
        self.out = 1.0 / (1.0 + np.exp(-x))
        return self.out

    def backward(self, dout):
        dLdx = dout * self.out * (1.0 - self.out)
        return dLdx

class Tanh:
    def __init__(self):
        self.out = None

    # tは未使用。
    def forward(self, x, t):
        self.out = np.tanh(x)
        return self.out

    def backward(self, dout):
        return dout * (1.0 - self.out**2)

class ReLU:
    def __init__(self):
        self.mask = None

    # tは未使用。
    def forward(self, x, t):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        return dout

class SoftmaxWithLoss():
    def __init__(self):
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.y = self.softmax(x)
        self.t = t
        return self.y

    def backward(self, dout):
        # 活性化関数がソフトマックス関数で、且つ、損失関数がクロスエントロピー誤差関数の場合のみ、y-tになる。doutは未使用。
        # よって、損失関数として他の関数を使用する場合はこの式は成立しないので注意。
        dLdx = (self.y - self.t) / self.y.shape[0]
        return dLdx

    def softmax(self, x):
        # そもそもxが1次元配列である場合は和を取る方向を指定しない。
        if x.ndim == 1:
            x = x - np.max(x)
            return np.exp(x) / np.sum(np.exp(x))

        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T

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
