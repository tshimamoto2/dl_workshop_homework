import numpy as np

##################################################
# 隠れ層クラス
##################################################
class HiddenLayer:
    def __init__(self, W, B, actfunc, batch_normal=None):
        self.affine = Affine(W, B)
        self.activation = actfunc
        self.batch_normal = batch_normal
        self.act_dist = None

    def forward(self, x, t):
        # アフィン変換。
        a = self.affine.forward(x, t)

        # バッチ正規化。
        if self.batch_normal is not None:
            a = self.batch_normal.forward(a)

        # 活性化。
        z = self.activation.forward(a, t)

        # アクティベーション分布を見るために保持。
        self.act_dist = z

        return z

    def backward(self, dout):
        # 活性化の逆伝播。
        dout = self.activation.backward(dout)

        # バッチ正規化の逆伝播。
        if self.batch_normal is not None:
            dout = self.batch_normal.backward(dout)

        # アフィン変換の逆伝播。
        dout = self.affine.backward(dout)
        return dout

##################################################
# 出力層クラス
##################################################
class LastLayer:
    def __init__(self, W, B, actfunc, batch_normal=False):
        self.affine = Affine(W, B)
        self.activation = actfunc
        self.batch_normal = batch_normal
        self.act_dist = None

    def forward(self, x, t):
        a = self.affine.forward(x, t)
        z = self.activation.forward(a, t)
        self.act_dist = z  # アクティベーション分布を保存するために保持しておく。
        return z

    # クロスエントロピー誤差を用いる場合は引数のdoutは使用されない
    def backward(self, dout):
        dout = self.activation.backward(dout)
        dout = self.affine.backward(dout)
        return dout

##################################################
# 以下、各レイヤー内の個別レイヤーの実装。
# ※注意：ソフトマックスレイヤーだけ、forwardメソッドに教師データが必要。
# 個別レイヤーのforwardメソッドには不要だが造りを合わせておく。
##################################################
class Affine:
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

class Sigmoid:
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

class SoftmaxWithLoss:
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

class BatchNormal:
    def __init__(self, gamma=1.0, beta=0.0):
        self.gamma = gamma
        self.beta = beta
        self.eps = 1.0e-8

        # 誤差逆伝播のために、順伝播時に算出した値を保持しておくための変数。
        self.x = None
        self.mu = None
        self.diff = None
        self.diff2 = None
        self.var = None
        self.stddev = None
        self.invstddev = None
        self.xnormalized = None
        self.xscaled = None

    # 順伝播。
    # 入力データxの行方向にはミニバッチサイズ分だけの処理対象データが並んでいるとする。N個とする。
    # 入力データxの列方向には784個の各画素値が並んでいるものとする。D個（=784）とする。
    # ※なお、「ノード」は紛らわしい用語であるため、以下とする。
    # 　・『ノード』：ここでは計算グラフ理論でのノードを指すものとする。『加算ノード』、『乗算ノード』など。
    # 　・『画素ニューロン』：ニューラルネットワークの各層のいわゆる『ノード』を指すものとする。造語である。
    def forward(self, x):
        # 誤差逆伝播のために入力値を保持。
        self.x = x

        # N：ミニバッチサイズ。ミニバッチ1個に含まれる処理対象データの個数。
        # D：1個のデータ（画素ベクトル）の列方向の個数。本課題であれば1個のデータがD=784個の画素ニューロンから成っている。
        N = x.shape[0]
        D = x.shape[1]

        # (step1)各画素ニューロンについての画素値の平均を取る（よって列方向に加算）。D=784個の列方向のベクトルとなる。
        self.mu = (1.0 / N) * np.sum(x, axis=0)

        # (step2)後で分散を計算するために、各画素ニューロンについて偏差を取る。
        self.diff = x - self.mu

        # (step3)後で分散を計算するために、各画素ニューロンについて偏差平方を取る。
        self.diff2 = self.diff ** 2

        # (step4)各画素ニューロンについての分散を計算する。つまり列方向の偏差平方和を取って平均する。よって分散もD=784個の列方向のベクトルとなる。
        self.var = (1.0 / N) * np.sum(self.diff2, axis=0)

        # (step5)後で正規化するために、各画素ニューロンについての標準偏差を計算する。よって標準偏差もD=784個のベクトルとなる。
        self.stddev = np.sqrt(self.var)

        # (step6)後で正規化するために、各画素ニューロンについての標準偏差の逆数を計算する。
        self.invstddev = 1.0 / self.stddev

        # (step7)各画素ニューロンについて正規化。よって正規化後のxもD=784個のベクトルとなる。
        self.xnormalized = self.diff * self.invstddev

        # (step8)各画素ニューロンについてスケーリング。よってスケーリング後のもD=784個のベクトルとなる。
        self.xscaled = self.gamma * self.xnormalized

        # (step9)各画素ニューロンについてシフト。よってシフト後のもD=784個のベクトルとなる。
        xshifted = self.xscaled + self.beta

        # 分かりやすいように変数名を変えただけ。誤差逆伝播には関係なし。
        out = xshifted

        return out

    def backward(self, dout):
        N = dout.shape[0]
        D = dout.shape[1]

        # (逆step9)（加算ノード）各画素ニューロンについてシフト値の逆伝播。doutが逆伝播して来て、ベータ値ベクトルとスケール化x行列との偏微分値に分かれる。
        dLdBeta = np.sum(dout, axis=0)
        dLdXscaled = 1 * dout  # ↓次工程に伝わる。

        # (逆step8)（乗算ノード）各画素ニューロンについてスケーリング値の逆伝播。前工程からdLdXscaledが逆伝播して来て、ガンマ値ベクトルと正規化x行列の偏微分値に分かれる。
        # ただし元のgammaは1次元であるため総和を取る。
        dLdGamma = np.sum(dLdXscaled * self.xnormalized, axis=0)
        dLdXnormalized = dLdXscaled * self.gamma  # ↓次工程に伝わる。

        # (逆step7)（乗算ノード）各画素ニューロンについて正規化した値の逆伝播。前工程からdLdXnormalizedが逆伝播して来て、平均ベクトルと標準偏差の逆数ベクトルの偏微分値に分かれる。
        # ただし元のinvstddevは1次元であるため各画素ニューロンについての総和を取る。
        dMu = dLdXnormalized * self.invstddev  # (逆step2）の工程に逆伝播するまで出番は無い。
        dInvstddev = np.sum(dLdXnormalized * self.mu, axis=0)  # ↓次工程に伝わる。

        # (逆step6)（除算ノード）各画素ニューロンについての標準偏差の逆数の逆伝播。前工程からdInvstddevが逆伝播して来て、標準偏差ベクトルの偏微分値に変換される。
        dLdStddev = dInvstddev * ((-1.0) / (self.stddev**2))  # ↓次工程に伝わる。

        # (逆step5)（平方根ノード）各画素ニューロンについての標準偏差の逆伝播。前工程からdLdStddevが逆伝播して来て、分散ベクトルの偏微分値に変換される。
        dLdVar = dLdStddev * (1.0 / (2.0 * np.sqrt(self.var + self.eps)))  # ↓次工程に伝わる。

        # (逆step4)（総和ノード。ただし固定値1/Nがかかっている値の総和であることに注意）。前工程からdLdVarが逆伝播して来て、偏差平方の偏微分値に変換される。
        # ただし、逆伝播してきたdLdVarは1次元に集約されていたので、逆変換によって行列（偏差行列self.diff2）の構造に戻している。
        dLdDiff2 = dLdVar * (1.0 / N) * np.ones((N, D))  # ↓次工程に伝わる。

        # (逆step3)（2乗ノード）各画素ニューロンについて偏差平方の逆伝播。前工程からdLdDiff2が逆伝播してきて、偏差の偏微分値に変換される。
        dLdDiff = dLdDiff2 * (2.0 * self.diff)  # ↓次工程に伝わる。

        # (逆step2)（減算ノード＝負の加算ノード）各画素ニューロンについて偏差の逆伝播。前工程からdLdDiff、(逆step7)のdMuが逆伝播して来て、xと平均の偏微分値に変換される。
        # ただし、このノードには2つの偏微分値が逆伝播して来るので、一旦単純に加えて、1つの逆入力変数にしておく（誤差逆伝播法での定義）。
        dout_temp = dLdDiff + dMu
        # xの偏微分値への変換。加算ノードなので1掛けるだけ。
        # ただし、step2系統のxであることを明確にするために添え字を2を付けているので注意。
        dLdX2 = dout_temp * 1.0  # ↓step2系統の逆伝播の入力として最後の工程step0に伝わる。
        # また、元のmuは1次元であるため、各画素ニューロンについての総和を取る。ただし、負の値の加算ノードなので-1を掛けているので注意。
        dLdMu = (-1.0) * np.sum(dout_temp, axis=0)  # ↓次工程（step1系統の最後の工程）に伝わる。

        # (逆step1)各画素ニューロンについての画素値の平均の逆伝播。前工程からdLdMuが伝播して来て、入力データ行列x
        # 逆伝播してきたdLdMuは1次元に集約されていたので、逆変換によって行列（入力データ行列self.x）の構造に戻している。
        # ただし、step1系統のxであることを明確にするために添え字を1を付けているので注意。
        dLdX1 = dLdMu * (1.0 / N) * np.ones((N, D))

        # 最後に、順伝播の最初に戻ってくる。順伝播の最初にstep1系統とstep2系統に分岐していたので、逆伝播では2つの入力となる。
        dout = dLdX1 + dLdX2

        return dout


##################################################
# 以下、損失関数クラスの実装。
##################################################
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
