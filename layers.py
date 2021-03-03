import numpy as np

class HiddenLayer():
    def __init__(self, W, B):
        self.affine = Affine(W, B)

        # TODO 活性化関数を切り替えれるようにしたい。
        #self.activation = Sigmoid()  # TODO 勾配消失問題が発生するためか、学習が進まず100回のエポックでも正解率が20％のままだった。
        self.activation = ReLU()  # 学習が進み正解率が97％を超過した。

    def forward(self, x, t):
        a = self.affine.forward(x, t)
        z = self.activation.forward(a, t)
        return z

    def backward(self, dout):
        dout = self.activation.backward(dout)
        dout = self.affine.backward(dout)
        return dout

class LastLayer():
    def __init__(self, W, B):
        # super().__init__()
        self.affine = Affine(W, B)
        self.activation = SoftmaxWithLoss()  #  TODO 活性化関数をソフトマックス関数としておく。切り替えれるようにしたい。

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
        # TODO 損失関数とセットで誤差逆伝播を切り替えれるようにしたい。
        dLdx = (self.y - self.t) / self.y.shape[0]
        return dLdx

    def softmax(self, x):
        if x.ndim == 1:
            x = x - np.max(x)
            return np.exp(x) / np.sum(np.exp(x))

        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T
