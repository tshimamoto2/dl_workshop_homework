import numpy as np

##################################################
# 共通クラス、メソッドなど。
##################################################
class XavierWeight:
    def __init__(self, prev_size, crnt_size):
        self.stddev = np.sqrt(2.0 / (prev_size + crnt_size))
    def get_stddev(self):
        return self.stddev

class HeWeight:
    def __init__(self, prev_size):
        self.stddev = np.sqrt(2.0 / prev_size)
    def get_stddev(self):
        return self.stddev

class NormalWeight:
    def __init__(self, stddev):
        self.stddev = stddev
    def get_stddev(self):
        return self.stddev

##################################################
# 以下、各レイヤー内の個別レイヤーの実装。
# ※注意：ソフトマックスレイヤーだけ、forwardメソッドに教師データが必要。
# 個別レイヤーのforwardメソッドには不要だが造りを合わせておく。
##################################################
##################################################
# Affineレイヤー（線形変換）
# 【前提】
#   ・入力の第1次元はバッチ軸であること。
#   ・CNNで使用される場合にはforwardの入力xがテンソルである場合もあるので、2次元化すること。
##################################################
class Affine:
    def __init__(self, crnt_size=100, weight=NormalWeight(0.01)):
        self.crnt_size = crnt_size
        self.weight = weight

        self.W = None
        self.B = None
        self.dLdW = None
        self.dLdB = None

        self.is_input_4d = False
        self.x2d = None
        self.batch_size = None
        self.channel_size = None
        self.height = None
        self.width = None
        self.prev_size = None

        self.numerical_dLdW = None
        self.numerical_dLdB = None

    # TODO tは未使用。使わない変数は渡したくないので、抽象クラスを作って吸収したい。
    def forward(self, x, t, is_learning=False):
        # テンソル対応。
        if x.ndim == 2:
            self.x2d = x
            self.batch_size = x.shape[0]
            self.prev_size = x.shape[1]
        elif x.ndim == 4:
            # 入力xが4次元テンソルの場合2次元化しておく。ただし、逆伝播のために、オリジナルの入力xの次元構造を覚えておく必要がある。
            self.x2d = x.reshape(x.shape[0], -1)

            self.batch_size, self.channel_size, self.height, self.width = x.shape
            self.prev_size = self.channel_size * self.height * self.width
            self.is_input_4d = True
        else:
            # 上記以外の次元構造は想定外。TODO エラーにするべきでは？
            self.is_input_4d = None
            pass

        # 重み、バイアスの生成・初期化。
        if self.W is None:
            self.W, self.B = self.init_weight(self.prev_size, self.crnt_size)

        # アフィン変換。
        out = np.dot(self.x2d, self.W) + self.B
        return out

    def backward(self, dout):
        self.dLdW = np.dot(self.x2d.T, dout)
        self.dLdB = np.sum(dout, axis=0)
        dout = np.dot(dout, self.W.T)
        
        if self.is_input_4d:
            dout = dout.reshape(self.batch_size, self.channel_size, self.height, self.width)

        return dout

    def init_weight(self, prev_size=784, crnt_size=100):
        weight = np.random.randn(prev_size, crnt_size) * self.weight.get_stddev()
        bias = np.zeros(crnt_size)
        return weight, bias


class Sigmoid:
    def __init__(self):
        self.out = None

    # tは未使用。
    # TODO debug 使わない変数は渡したくないので、抽象クラスを作って吸収したい。
    def forward(self, x, t, is_learning=False):
        self.out = 1.0 / (1.0 + np.exp(-x))
        return self.out

    def backward(self, dout):
        dLdx = dout * self.out * (1.0 - self.out)
        return dLdx

class Tanh:
    def __init__(self):
        self.out = None

    # tは未使用。
    # TODO debug 使わない変数は渡したくないので、抽象クラスを作って吸収したい。
    def forward(self, x, t, is_learning=False):
        self.out = np.tanh(x)
        return self.out

    def backward(self, dout):
        return dout * (1.0 - self.out**2)

class ReLU:
    def __init__(self):
        self.mask = None

        self.is_input_4d = False
        self.x2d = None
        self.batch_size = None
        self.channel_size = None
        self.height = None
        self.width = None
        self.prev_size = None

    # tは未使用。
    # TODO debug 使わない変数は渡したくないので、抽象クラスを作って吸収したい。
    def forward(self, x, t, is_learning=False):
        # テンソル対応。
        if x.ndim == 2:
            self.x2d = x
        elif x.ndim == 4:
            # 入力xが4次元テンソルの場合2次元化しておく。ただし、逆伝播のために、オリジナルの入力xの次元構造を覚えておく必要がある。
            self.x2d = x.reshape(x.shape[0], -1)

            self.is_input_4d = True
            self.batch_size, self.channel_size, self.height, self.width = x.shape
            self.prev_size = self.channel_size * self.height * self.width
        else:
            # 上記以外の次元構造は想定外。TODO エラーにするべきでは？
            self.is_input_4d = None
            pass

        self.mask = (self.x2d <= 0)
        out = self.x2d.copy()  # 念のためコピー。入力の中身がどこで使われるか分からないので触りたくないため。
        out[self.mask] = 0

        if self.is_input_4d:
            out = out.reshape(self.batch_size, self.channel_size, self.height, self.width)

        return out

    def backward(self, dout):
        # テンソル対応。
        if self.is_input_4d:
            dout = dout.reshape(self.batch_size, -1)

        dout[self.mask] = 0

        if self.is_input_4d:
            dout = dout.reshape(self.batch_size, self.channel_size, self.height, self.width)

        return dout

class SoftmaxWithLoss:
    def __init__(self):
        self.y = None
        self.t = None

    def forward(self, x, t, is_learning=False):
        self.y = self.softmax(x)
        self.t = t
        return self.y

    def backward(self, dout):
        # 活性化関数がソフトマックス関数で、且つ、損失関数がクロスエントロピー誤差関数の場合のみ、y-tになる。doutは未使用。
        # よって、損失関数として他の関数を使用する場合はこの式は成立しないので注意。
        dLdX = (self.y - self.t) / self.y.shape[0]
        return dLdX

    def softmax(self, x):
        # そもそもxが1次元配列である場合は和を取る方向を指定しない。
        if x.ndim == 1:
            x = x - np.max(x)
            return np.exp(x) / np.sum(np.exp(x))

        elif x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T

        else:
            # 上記以外の次元構造は想定外。TODO エラーにするべきでは？
            return None

class DropoutParams:
    def __init__(self, input_retain_rate=0.8, hidden_retain_rate=0.5):
        self.input_retain_rate = input_retain_rate
        self.hidden_retain_rate = hidden_retain_rate

class Dropout:
    def __init__(self, retain_rate=0.5):
        self.retain_rate = retain_rate
        self.mask = None

    # tは未使用。
    def forward(self, x, t, is_learning=False):
        if is_learning:
            self.mask = np.random.rand(*x.shape) <= self.retain_rate
            return x * self.mask
        else:
            return x * self.retain_rate

    def backward(self, dout):
        # backwardは学習時のみ利用されるため、forwardが呼び出される前提であれば
        # （上記is_learningが必ずTrueになるようにしておけば）、self.maskがNoneとなることはないの以下の式とする。
        # 万が一self.maskがNoneになる状況があるとすればそればバグで、実行時エラー（下記）が発生するので検知できる。
        # 『TypeError: unsupported operand type(s) for *: 'int' and 'NoneType'』
        return dout * self.mask

##################################################
# バッチ正規化（Algorithm1のみ実装版）
##################################################
# class BatchNormal_OnlyAlgorithm1:
#     def __init__(self, gamma=1.0, beta=0.0):
#         self.gamma = gamma
#         self.beta = beta
#         self.eps = 1.0e-8
#
#         # 誤差逆伝播のために、順伝播時に算出した値を保持しておくための変数。
#         self.x = None
#         self.mu = None
#         self.diff = None
#         self.diff2 = None
#         self.var = None
#         self.stddev = None
#         self.invstddev = None
#         self.xnormalized = None
#         self.xscaled = None
#
#     # 順伝播。
#     # 入力データxの行方向にはミニバッチサイズ分だけの処理対象データが並んでいるとする。N個とする。
#     # 入力データxの列方向には784個の各画素値が並んでいるものとする。D個（=784）とする。
#     # ※なお、「ノード」は紛らわしい用語であるため、以下とする。
#     # 　・『ノード』：ここでは計算グラフ理論でのノードを指すものとする。『加算ノード』、『乗算ノード』など。
#     # 　・『画素ニューロン』：ニューラルネットワークの各層のいわゆる『ノード』を指すものとする。造語である。
#     def forward(self, x):
#         # 誤差逆伝播のために入力値を保持。
#         self.x = x
#
#         # N：ミニバッチサイズ。ミニバッチ1個に含まれる処理対象データの個数。
#         # D：1個のデータ（画素ベクトル）の列方向の個数。本課題であれば1個のデータがD=784個の画素ニューロンから成っている。
#         N = x.shape[0]
#         D = x.shape[1]
#
#         # (step1)各画素ニューロンについての画素値の平均を取る（よって列方向に加算）。D=784個の列方向のベクトルとなる。
#         self.mu = (1.0 / N) * np.sum(x, axis=0)
#
#         # (step2)後で分散を計算するために、各画素ニューロンについて偏差を取る。
#         self.diff = x - self.mu
#
#         # (step3)後で分散を計算するために、各画素ニューロンについて偏差平方を取る。
#         self.diff2 = self.diff ** 2
#
#         # (step4)各画素ニューロンについての分散を計算する。つまり列方向の偏差平方和を取って平均する。よって分散もD=784個の列方向のベクトルとなる。
#         self.var = (1.0 / N) * np.sum(self.diff2, axis=0)
#
#         # (step5)後で正規化するために、各画素ニューロンについての標準偏差を計算する。よって標準偏差もD=784個のベクトルとなる。
#         self.stddev = np.sqrt(self.var)
#
#         # (step6)後で正規化するために、各画素ニューロンについての標準偏差の逆数を計算する。
#         self.invstddev = 1.0 / self.stddev
#
#         # (step7)各画素ニューロンについて正規化。よって正規化後のxもD=784個のベクトルとなる。
#         self.xnormalized = self.diff * self.invstddev
#
#         # (step8)各画素ニューロンについてスケーリング。よってスケーリング後のもD=784個のベクトルとなる。
#         self.xscaled = self.gamma * self.xnormalized
#
#         # (step9)各画素ニューロンについてシフト。よってシフト後のもD=784個のベクトルとなる。
#         xshifted = self.xscaled + self.beta
#
#         # 分かりやすいように変数名を変えただけ。誤差逆伝播には関係なし。
#         out = xshifted
#
#         return out
#
#     def backward(self, dout):
#         N = dout.shape[0]
#         D = dout.shape[1]
#
#         # (逆step9)（加算ノード）各画素ニューロンについてシフト値の逆伝播。doutが逆伝播して来て、ベータ値ベクトルとスケール化x行列との偏微分値に分かれる。
#         dLdBeta = np.sum(dout, axis=0)
#         dLdXscaled = 1 * dout  # ↓次工程に伝わる。
#
#         # (逆step8)（乗算ノード）各画素ニューロンについてスケーリング値の逆伝播。前工程からdLdXscaledが逆伝播して来て、ガンマ値ベクトルと正規化x行列の偏微分値に分かれる。
#         # ただし元のgammaは1次元であるため総和を取る。
#         dLdGamma = np.sum(dLdXscaled * self.xnormalized, axis=0)
#         dLdXnormalized = dLdXscaled * self.gamma  # ↓次工程に伝わる。
#
#         # (逆step7)（乗算ノード）各画素ニューロンについて正規化した値の逆伝播。前工程からdLdXnormalizedが逆伝播して来て、平均ベクトルと標準偏差の逆数ベクトルの偏微分値に分かれる。
#         # ただし元のinvstddevは1次元であるため各画素ニューロンについての総和を取る。
#         dMu = dLdXnormalized * self.invstddev  # (逆step2）の工程に逆伝播するまで出番は無い。
#         dInvstddev = np.sum(dLdXnormalized * self.mu, axis=0)  # ↓次工程に伝わる。
#
#         # (逆step6)（除算ノード）各画素ニューロンについての標準偏差の逆数の逆伝播。前工程からdInvstddevが逆伝播して来て、標準偏差ベクトルの偏微分値に変換される。
#         dLdStddev = dInvstddev * ((-1.0) / (self.stddev**2))  # ↓次工程に伝わる。
#
#         # (逆step5)（平方根ノード）各画素ニューロンについての標準偏差の逆伝播。前工程からdLdStddevが逆伝播して来て、分散ベクトルの偏微分値に変換される。
#         dLdVar = dLdStddev * (1.0 / (2.0 * np.sqrt(self.var + self.eps)))  # ↓次工程に伝わる。
#
#         # (逆step4)（総和ノード。ただし固定値1/Nがかかっている値の総和であることに注意）。前工程からdLdVarが逆伝播して来て、偏差平方の偏微分値に変換される。
#         # ただし、逆伝播してきたdLdVarは1次元に集約されていたので、逆変換によって行列（偏差行列self.diff2）の構造に戻している。
#         dLdDiff2 = dLdVar * (1.0 / N) * np.ones((N, D))  # ↓次工程に伝わる。
#
#         # (逆step3)（2乗ノード）各画素ニューロンについて偏差平方の逆伝播。前工程からdLdDiff2が逆伝播してきて、偏差の偏微分値に変換される。
#         dLdDiff = dLdDiff2 * (2.0 * self.diff)  # ↓次工程に伝わる。
#
#         # (逆step2)（減算ノード＝負の加算ノード）各画素ニューロンについて偏差の逆伝播。前工程からdLdDiff、(逆step7)のdMuが逆伝播して来て、xと平均の偏微分値に変換される。
#         # ただし、このノードには2つの偏微分値が逆伝播して来るので、一旦単純に加えて、1つの逆入力変数にしておく（誤差逆伝播法での定義）。
#         dout_temp = dLdDiff + dMu
#         # xの偏微分値への変換。加算ノードなので1掛けるだけ。
#         # ただし、step2系統のxであることを明確にするために添え字を2を付けているので注意。
#         dLdX2 = dout_temp * 1.0  # ↓step2系統の逆伝播の入力として最後の工程step0に伝わる。
#         # また、元のmuは1次元であるため、各画素ニューロンについての総和を取る。ただし、負の値の加算ノードなので-1を掛けているので注意。
#         dLdMu = (-1.0) * np.sum(dout_temp, axis=0)  # ↓次工程（step1系統の最後の工程）に伝わる。
#
#         # (逆step1)各画素ニューロンについての画素値の平均の逆伝播。前工程からdLdMuが伝播して来て、入力データ行列x
#         # 逆伝播してきたdLdMuは1次元に集約されていたので、逆変換によって行列（入力データ行列self.x）の構造に戻している。
#         # ただし、step1系統のxであることを明確にするために添え字を1を付けているので注意。
#         dLdX1 = dLdMu * (1.0 / N) * np.ones((N, D))
#
#         # 最後に、順伝播の最初に戻ってくる。順伝播の最初にstep1系統とstep2系統に分岐していたので、逆伝播では2つの入力となる。
#         dout = dLdX1 + dLdX2
#
#         return dout

##################################################
# バッチ正規化（Algorithm1＋Algorithm2）
##################################################
class BatchNormalParams:
    def __init__(self, gamma=1.0, beta=0.0, moving_decay=0.1):
        self.gamma = gamma
        self.beta = beta
        self.moving_decay = moving_decay

class BatchNormal:
    def __init__(self, batch_normal_params):
        self.gamma = batch_normal_params.gamma
        self.beta = batch_normal_params.beta
        self.moving_decay = batch_normal_params.moving_decay
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

        self.moving_ex = None  #  E[x]の移動平均（xはD個のベクトル）
        self.moving_varx = None  #  Var[x]の移動平均（xはD個のベクトル）

    # 順伝播。
    # 入力データxの行方向にはミニバッチサイズ分だけの処理対象データが並んでいるとする。N個とする。
    # 入力データxの列方向には784個の各画素値が並んでいるものとする。D個（=784）とする。
    # ※なお、「ノード」は紛らわしい用語であるため、以下とする。
    # 　・『ノード』：ここでは計算グラフ理論でのノードを指すものとする。『加算ノード』、『乗算ノード』など。
    # 　・『画素ニューロン』：ニューラルネットワークの各層のいわゆる『ノード』を指すものとする。造語である。
    def forward(self, x, is_learning=False):
        # 誤差逆伝播のために入力値を保持。
        self.x = x

        # N：ミニバッチサイズ。ミニバッチ1個に含まれる処理対象データの個数。
        # D：1個のデータ（画素ベクトル）の列方向の個数。本課題であれば1個のデータがD=784個の画素ニューロンから成っている。
        N = x.shape[0]
        D = x.shape[1]

        if is_learning:
            # 最初のE[x]とVar[x]の移動平均は0（ベクトル）とする。
            if self.moving_ex is None:
                self.moving_ex = np.zeros(D, dtype=np.float32)
                self.moving_varx = np.zeros(D, dtype=np.float32)

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
            self.invstddev = 1.0 / (self.stddev + self.eps)

            # (step7)各画素ニューロンについて正規化。よって正規化後のxもD=784個のベクトルとなる。
            self.xnormalized = self.diff * self.invstddev

            # (step8)各画素ニューロンについてスケーリング。よってスケーリング後のもD=784個のベクトルとなる。
            self.xscaled = self.gamma * self.xnormalized

            # (step9)各画素ニューロンについてシフト。よってシフト後のもD=784個のベクトルとなる。
            out = self.xscaled + self.beta

            # 移動平均の更新。
            self.moving_ex = self.moving_ex * self.moving_decay + (1 - self.moving_decay) * self.mu
            self.moving_varx = self.moving_varx * self.moving_decay + (1 - self.moving_decay) * (N / np.max([N-1, 1])) * self.var

        else:
            out = self.gamma * (self.x - self.moving_ex) / np.sqrt(self.moving_varx + self.eps) + self.beta

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
        dLdStddev = dInvstddev * (-1.0) / (self.stddev**2 + self.eps)  # ↓次工程に伝わる。

        # (逆step5)（平方根ノード）各画素ニューロンについての標準偏差の逆伝播。前工程からdLdStddevが逆伝播して来て、分散ベクトルの偏微分値に変換される。
        dLdVar = dLdStddev * 0.5 * np.sqrt(self.var + self.eps)  # ↓次工程に伝わる。

        # (逆step4)（総和ノード。ただし固定値1/Nがかかっている値の総和であることに注意）。前工程からdLdVarが逆伝播して来て、偏差平方の偏微分値に変換される。
        # ただし、逆伝播してきたdLdVarは1次元に集約されていたので、逆変換によって行列（偏差行列self.diff2）の構造に戻している。
        dLdDiff2 = dLdVar * (1.0 / N) * np.ones((N, D))  # ↓次工程に伝わる。

        # (逆step3)（2乗ノード）各画素ニューロンについて偏差平方の逆伝播。前工程からdLdDiff2が逆伝播してきて、偏差の偏微分値に変換される。
        dLdDiff = dLdDiff2 * 2.0 * self.diff  # ↓次工程に伝わる。

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
# 隠れ層クラス
##################################################
class HiddenLayer:
    def __init__(self, W, B, actfunc, batch_normal_params=None, input_dropout=None, hidden_dropout=None):
        self.affine = Affine(W, B)
        self.activation = actfunc
        self.act_dist = None

        self.input_dropout = input_dropout
        self.hidden_dropout = hidden_dropout

        self.batch_normal_params = batch_normal_params
        self.batch_normal = None
        if self.batch_normal_params is not None:
            self.batch_normal = BatchNormal(self.batch_normal_params)

    def forward(self, x, t, is_learning=False):
        z = x

        # 入力層のノードのドロップアウト。
        if self.input_dropout is not None:
            z = self.input_dropout.forward(z, t, is_learning)

        # アフィン変換。
        z = self.affine.forward(z, t)

        # バッチ正規化。アフィン変換と活性化の間。
        if self.batch_normal_params is not None:
            z = self.batch_normal.forward(z, is_learning)

        # 活性化。
        z = self.activation.forward(z, t)

        # 隠れ層のノードのドロップアウト。活性化の後。
        if self.hidden_dropout is not None:
            z = self.hidden_dropout.forward(z, t, is_learning)

        # アクティベーション分布を見るために保持。
        self.act_dist = z

        return z

    def backward(self, dout):
        # 隠れ層のノードのドロップアウトの逆伝播。
        if self.hidden_dropout is not None:
            dout = self.hidden_dropout.backward(dout)

        # 活性化の逆伝播。
        dout = self.activation.backward(dout)

        # バッチ正規化の逆伝播。
        if self.batch_normal is not None:
            dout = self.batch_normal.backward(dout)

        # アフィン変換の逆伝播。
        dout = self.affine.backward(dout)

        # 入力層のノードのドロップアウトの逆伝播。
        if self.input_dropout is not None:
            dout = self.input_dropout.backward(dout)

        return dout

##################################################
# 出力層クラス
##################################################
class LastLayer:
    def __init__(self, W, B, actfunc):
        self.affine = Affine(W, B)
        self.activation = actfunc
        self.act_dist = None

    # is_learningは未使用。HiddenLayerクラスと形を合わせただけ。
    # TODO debug 使わない変数は渡したくないので、抽象クラスを作って吸収したい。
    def forward(self, x, t, is_learning=False):
        a = self.affine.forward(x, t)
        z = self.activation.forward(a, t)
        self.act_dist = z  # アクティベーション分布を保存するために保持しておく。
        return z

    # クロスエントロピー誤差を用いる場合は引数のdoutは使用されない
    def backward(self, dout):
        dout = self.activation.backward(dout)
        dout = self.affine.backward(dout)
        return dout


