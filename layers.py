import numpy as np
from abc import ABCMeta, abstractmethod

##################################################
# 重みの初期値クラス
##################################################
class XavierWeight:
    def __init__(self):
        pass
    def get_stddev(self, prev_size, crnt_size):
        return np.sqrt(2.0 / (prev_size + crnt_size))

class HeWeight:
    def __init__(self):
        pass
    def get_stddev(self, prev_size):
        return np.sqrt(2.0 / prev_size)

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
class Layer(metaclass=ABCMeta):
    @abstractmethod
    def has_weight(self):
        pass

class Affine(Layer):
    def __init__(self, node_size=100, weight=NormalWeight(0.01)):
        self.node_size = node_size
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
        self.prev_node_size = None

        self.numerical_dLdW = None
        self.numerical_dLdB = None

    def has_weight(self):
        return True

    def init_weight(self, prev_node_size=784, node_size=100):
        stddev = None
        if isinstance(self.weight, XavierWeight):
            stddev = self.weight.get_stddev(prev_node_size, node_size)
        elif isinstance(self.weight, HeWeight):
            stddev = self.weight.get_stddev(prev_node_size)
        elif isinstance(self.weight, NormalWeight):
            stddev = self.weight.get_stddev()
        else:
            pass
        # デフォルトでは64ビット浮動小数点数であり、メモリ効率が悪いため、float32で保持する。
        # ⇒こうすることでNN自体のpklファイルへの保存容量も減る。
        weight = np.random.randn(prev_node_size, node_size).astype(np.float32) * stddev
        bias = np.zeros(node_size).astype(np.float32)
        # weight = np.random.randn(prev_node_size, node_size) * stddev
        # bias = np.zeros(node_size)
        return weight, bias

    # TODO tは未使用。使わない変数は渡したくないので、抽象クラスを作って吸収したい。
    def forward(self, x, t, is_learning=False):
        # テンソル対応。
        if x.ndim == 2:
            self.x2d = x
            self.batch_size = x.shape[0]
            self.prev_node_size = x.shape[1]
        elif x.ndim == 4:
            # 入力xが4次元テンソルの場合2次元化しておく。ただし、逆伝播のために、オリジナルの入力xの次元構造を覚えておく必要がある。
            self.x2d = x.reshape(x.shape[0], -1)

            self.batch_size, self.channel_size, self.height, self.width = x.shape
            self.prev_node_size = self.channel_size * self.height * self.width
            self.is_input_4d = True
        else:
            # 上記以外の次元構造は想定外。TODO エラーにするべきでは？
            self.is_input_4d = None
            pass

        # 重み、バイアスの生成・初期化。
        if self.W is None:
            self.W, self.B = self.init_weight(self.prev_node_size, self.node_size)

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

class Sigmoid(Layer):
    def __init__(self):
        self.out = None

    def has_weight(self):
        return False

    # tは未使用。
    # TODO debug 使わない変数は渡したくないので、抽象クラスを作って吸収したい。
    def forward(self, x, t, is_learning=False):
        self.out = 1.0 / (1.0 + np.exp(-x))
        return self.out

    def backward(self, dout):
        dLdx = dout * self.out * (1.0 - self.out)
        return dLdx

class Tanh(Layer):
    def __init__(self):
        self.out = None

    def has_weight(self):
        return False

    # tは未使用。
    # TODO debug 使わない変数は渡したくないので、抽象クラスを作って吸収したい。
    def forward(self, x, t, is_learning=False):
        self.out = np.tanh(x)
        return self.out

    def backward(self, dout):
        return dout * (1.0 - self.out**2)

class ReLU(Layer):
    def __init__(self):
        self.mask = None

        self.is_input_4d = False
        self.x2d = None
        self.batch_size = None
        self.channel_size = None
        self.height = None
        self.width = None
        self.prev_size = None

    def has_weight(self):
        return False

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

class SoftmaxWithLoss(Layer):
    def __init__(self):
        self.y = None
        self.t = None

    def has_weight(self):
        return False

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

class Dropout(Layer):
    def __init__(self, retain_rate=0.5):
        self.retain_rate = retain_rate
        self.mask = None

    def has_weight(self):
        return False

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

class BatchNormal(Layer):
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

    def has_weight(self):
        return False

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
class HiddenLayer(Layer):
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

    def has_weight(self):
        return True

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
class LastLayer(Layer):
    def __init__(self, W, B, actfunc):
        self.affine = Affine(W, B)
        self.activation = actfunc
        self.act_dist = None

    def has_weight(self):
        return True

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

##################################################
# CNNを実装するにあたり、下記前提を置いた。
# 【前提】
#     ・入力データxは4階のテンソルであること。
#     ・入力データxの第1次元はバッチ軸であること。よって、x.shape[0]は一度に処理されるミニバッチサイズである。
#     ・入力データxの第2次元はチャネル軸であること。よって、x.shape[1]は一度に処理されるチャネル数である。
#         - ニューラルネットワークの最初の入力データの場合は、画像のチャネルそのものを指す。すなわち、グレースケール形式なら1、RGB形式なら3である。
#         - また、ニューラルネットワーク途中で本Convレイヤーが呼び出された場合は、前レイヤーでのフィルター数（変数ではFN）となる。
#     ・入力データxの第3次元は画像の高さ軸である。よって、x.shape[2]は画像の高さ[pixel]である。
#     ・入力データxの第4次元は画像の幅軸である。よって、x.shape[3]は画像の幅[pixel]である。
#     ・畳み込みフィルターWは4階のテンソルであること。
#     ・畳み込みフィルターWの第1次元はフィルター軸であること。よって、x.shape[0]は一度に処理されるフィルター個数である。
#     ・畳み込みフィルターWの第2次元はチャネル軸であること。よって、x.shape[1]は一度に処理されるフィルター個数である。
#     ・畳み込みフィルターWの第3次元は、入力データxの画像の高さに対応する、フィルターの高さ軸である。よって、x.shape[2]はフィルター高さ[pixel]である。
#     ・畳み込みフィルターWの第4次元は、入力データxの画像の幅に対応する、フィルターの幅軸である。よって、x.shape[3]はフィルター幅[pixel]である。
#     ・入力、フィルタ、出力ともに正方行列を前提とする。畳み込みのstrideは高さ軸方向、幅軸方向とも同じ値とすること。
#          - 入力データxの画像データは正方行列とする。よってx.shape[2]==x.shape[3]であること。ただし変数としてはIH、IWなどと分けておく。
#          - フィルタについても高さと幅が等しい正方行列であること。ただし変数としてはFH、FWなどと分けておく。
#          - 出力の画像サイズも正方行列になること。ただし変数としてはOH、OWなどと分けておく。
#     ・ハイパーパラメータとして指定するパディング値は、画像の高さ方向と幅方向の両方に同時に適用されること（高さ方向にも幅方向にも同数だけパディングされること）
##################################################
# 畳み込み層
##################################################
# 畳み込み層は、実装の難易度を下げるため、以下の前提を置いた。
class Conv:
    def __init__(self, FN=1, FH=2, FW=2, padding=0, stride=1, weight=NormalWeight(0.01)):
        self.FN = FN  # フィルター数。畳み込みの出力のチャンネル数になる。
        self.FH = FH  # フィルター高さ[ピクセル]
        self.FW = FW  # フィルター幅[ピクセル]
        self.padding = padding  # パディングサイズ[ピクセル]
        self.stride = stride  # フィルター（ウィンドウ）の1回の走査あたりの移動サイズ[ピクセル]
        self.weight = weight  # 重みの初期値オブジェクト
        self.OH = None
        self.OW = None

        self.x = None
        self.x2d = None
        self.f2d = None

        self.W = None  # フィルターの重み
        self.B = None  # フィルターのバイアス
        self.dLdW = None  # 損失関数の、本畳み込み層の重みによる偏微分値
        self.dLdB = None  # 損失関数の、本畳み込み層のバイアスによる偏微分値

    def has_weight(self):
        return True

    def forward(self, x, t, is_learning=False):
        # 誤差逆伝播のために保持。
        self.x = x

        # 入力データの各次元のサイズを取得。
        batch_size, channel_size, IH, IW = x.shape
        OH = out_size(IH, self.FH, self.padding, self.stride)
        OW = out_size(IW, self.FW, self.padding, self.stride)

        # フィルターの重みとバイアスを生成。
        if self.W is None:
            # デフォルトでは64ビット浮動小数点数であり、メモリ効率が悪いため、float32で保持する。
            # ⇒こうすることでNN自体のpklファイルへの保存容量も減る。
            self.W = np.random.randn(self.FN, channel_size, self.FH, self.FW).astype(np.float32) * self.weight.get_stddev()
            self.B = np.zeros((self.FN, OH, OW)).astype(np.float32)
            # self.W = np.random.randn(self.FN, channel_size, self.FH, self.FW) * self.weight.get_stddev()
            # self.B = np.zeros((self.FN, OH, OW))

        # 入力データとフィルターとの畳み込み演算。2次元化したデータは逆伝播のために保持。
        self.x2d = im2col_HP(x, self.FH, self.FW, self.padding, self.stride)
        self.f2d = self.W.reshape(self.FN, -1)
        out = np.dot(self.x2d, self.f2d.T)

        # 4階テンソルに戻す。
        out = out.reshape(batch_size, OH, OW, -1).transpose(0, 3, 1, 2)
        out += self.B
        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape

        # 出力を2次元化。ただしim2colの仕様に合わせて、チャネルが列方向に並ぶように変形。
        dout2d = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        # dLdB
        self.dLdB = np.sum(dout, axis=0)

        # dLdW
        dLdW2d = np.dot(self.x2d.T, dout2d)
        self.dLdW = dLdW2d.transpose(1, 0).reshape(FN, C, FH, FW)

        # dLdX
        dx2d = np.dot(dout2d, self.f2d)
        dLdX = col2im_HP(dx2d, self.x.shape, FH, FW, self.padding, self.stride)

        return dLdX

class MaxPool:
    def __init__(self, FH=1, FW=1, padding=0, stride=1):
        self.B = None
        self.C = None
        self.IH = None
        self.IW = None
        self.FH = FH
        self.FW = FW
        self.padding = padding
        self.stride = stride
        self.OH = None
        self.OW = None

        self.x = None
        self.x2d = None
        self.out = None
        self.arg_max = None

    def has_weight(self):
        return False

    def forward(self, x, t, is_learning_False):
        self.x = x
        self.B, self.C, self.IH, self.IW = self.x.shape
        self.OH = out_size(self.IH, self.FH, self.padding, self.stride)
        self.OW = out_size(self.IW, self.FW, self.padding, self.stride)

        # 入力値（4次元）の2次元化。
        x2d = im2col(x, self.FH, self.FW, self.padding, self.stride)

        # 列の幅を指定して、チャネル軸も縦1列に並べ替える。
        # よって、第1次元：バッチ軸（B個）、第2次元：高さ軸（OH個）ー幅軸（OW個）ーチャネル軸（C個）となる。
        x2d = x2d.reshape(-1, self.FH * self.FW)

        # 逆伝播のために、入力xをフィルターで走査してみて「切り取られた区画（ウィンドウ）の中のどの位置が」最大値だったのかを保持しておく。
        # （例）入力xがIHxIW=3x3で、フィルターサイズがPHxPW=2x2、padding=0、stride=1の場合、出力Oは2x2となる。
        #   例えば第3ウィンドウには左上から右下の順に[x21,x23,x31,x32]が含まれる。
        #   x2d、out、argmaxの関係例を以下に示す。
        #     第1ウィンドウx2d[0]（出力out11に対応）: [x11, x12, x21, x22] →最大値がx11だとすると、argmax=0
        #     第2ウィンドウx2d[1]（出力out12に対応）: [x12, x13, x22, x23] →最大値がx23だとすると、argmax=3
        #     第3ウィンドウx2d[2]（出力out21に対応）: [x21, x22, x31, x32] →最大値がx22だとすると、argmax=1
        #     第4ウィンドウx2d[3]（出力out22に対応）: [x22, x23, x32, x33] →最大値がx32だとすると、argmax=2
        self.arg_max = np.argmax(x2d, axis=1)

        # 第2次元（次元のインデックスは1。FH*FW個の要素がある）中の要素の最大値を取る。
        out2d = np.max(x2d, axis=1)

        # 4次元に戻す。
        #   上記次元構造（2次元）：バッチ軸（B個）／高さ軸（OH個）ー幅軸（OW個）ーチャネル軸（C個）
        #   ⇒4次元化：バッチ軸（B個）／高さ軸（OH個）／幅軸（OW個）／チャネル軸（C個）
        #   ⇒フィルタ軸を先に持ってくる：バッチ軸（B個）／チャネル軸（C個）／高さ軸（OH個）／幅軸（OW個）／
        out = out2d.reshape(self.B, self.OH, self.OW, self.C)
        out = out.transpose(0, 3, 1, 2)
        return out

    def backward(self, dout):
        # doutのチャンネル数軸を4番目に移動させる
        dout = dout.transpose(0, 2, 3, 1)

        # プーリング適応領域の要素数(プーリング適応領域の高さ × プーリング適応領域の幅)
        pool_size = self.FH * self.FW

        # 勾配を入れる配列を初期化する
        dx2d = np.zeros((dout.size, pool_size))

        # 順伝播計算時に最大値となった場所に、doutを配置する。
        # 逆に言うと、順伝播時に信号が流れなかった場所には逆伝播しない。
        # （例）入力データが最初の2行2列フィルターに当たるウィンドウ[x00, x01, x10, x11]のうち、
        # 　　　最大値がx10だとすると、x10の場所(argmaxが2)が1で、その他は0として信号が順伝播してきたとイメージするとよい。
        dx2d[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()

        # 勾配を4階テンソルに変換。
        dLdX = col2im_HP(dx2d, self.x.shape, self.FH, self.FW, self.padding, self.stride, is_maxpooling=True)

        return dLdX

##############################
# 以下ユーティリティ。特に畳み込み層、プーリング層で使用するメソッド。
##############################
##############################
# 出力サイズ算出
##############################
def out_size(in_size, fil_size, padding, stride):
    return (in_size + 2 * padding - fil_size) // stride + 1

##############################
# 4階テンソルの2次元化【高速版；参考資料の改変版】
# （参考）斎藤康毅『ゼロから作るDeep Learning』（オライリー・ジャパン）
# ただし、上記参考資料とは異なり、中間データとして利用する6階のテンソルの次元構造の一部を、
# 出力（の高さ軸、幅軸）→フィルタ（の高さ軸、幅軸）の順にした。
##############################
def im2col(x, FH, FW, padding=0, stride=1):
    B, C, H, W = x.shape
    OH = out_size(H, FH, padding, stride)
    OW = out_size(W, FW, padding, stride)

    # 画像データの高さ方向と幅方向を0パディング。
    # 第1, 2次元：ヘッドもテイルもパディングしない。第3, 4次元：ヘッドとテイル両方にパディング。
    img = np.pad(x, [(0, 0), (0, 0), (padding, padding), (padding, padding)], 'constant')

    # 一旦6次元配列を準備。ただし、参考資料のコードと違って、出力側のループを先に持ってくる。こちらの方がロジックが分かりやすい。
    x6d = np.zeros((B, C, OH, OW, FH, FW))

    # 入力データの移し替え。
    for i in range(OH):
        il = i * stride
        ir = il + FH
        for j in range(OW):
            jl = j * stride
            jr = jl + FW
            x6d[:, :, i, j, :, :] = img[:, :, il:ir, jl:jr]

    # チャネル次元を最後に持って行く。チャネル次元は列方向に並べたいため。
    x2d = x6d.transpose(0, 2, 3, 4, 5, 1).reshape(B * OH * OW, -1)

    return x2d

##############################
# 4階テンソルの2次元化【高速版】
# （参考）斎藤康毅『ゼロから作るDeep Learning』（オライリー・ジャパン）のロジックそのまま。
# 入力xから、stride個分飛ばしながらスライスして値を取ってくる方法。
##############################
def im2col_HP(x, FH, FW, padding=0, stride=1):
    B, C, H, W = x.shape
    OH = out_size(H, FH, padding, stride)
    OW = out_size(W, FW, padding, stride)

    # 画像データの高さ方向と幅方向を0パディング。
    # 第1, 2次元：ヘッドもテイルもパディングしない。第3, 4次元：ヘッドとテイル両方にパディング。
    img = np.pad(x, [(0, 0), (0, 0), (padding, padding), (padding, padding)], 'constant')

    # 一旦6次元配列を準備。
    x6d = np.zeros((B, C, FH, FW, OH, OW))

    # 入力データの移し替え。
    for h in range(FH):
        hmax = h + stride * OH
        for w in range(FW):
            wmax = w + stride * OW
            x6d[:, :, h, w, :, :] = img[:, :, h:hmax:stride, w:wmax:stride]

    x2d = x6d.transpose(0, 4, 5, 1, 2, 3).reshape(B * OH * OW, -1)
    return x2d

##############################
# 4次元化処理【高速版】
# （参考）斎藤康毅『ゼロから作るDeep Learning』（オライリー・ジャパン）
# ただし、参考資料とは異なり、畳み込み層で使う場合と、最大値プーリング層で使う場合とで処理を分けることができるようにしている。
##############################
def col2im_HP(col, input_shape, FH, FW, padding=0, stride=1, is_maxpooling=False):
    B, C, IH, IW = input_shape
    OH = out_size(IH, FH, padding, stride)
    OW = out_size(IH, FW, padding, stride)

    # 配列の形を変えて、軸を入れ替える
    col = col.reshape(B, OH, OW, C, FH, FW).transpose(0, 3, 4, 5, 1, 2)

    # 配列の初期化。pad分を大きくとっておく。stride分も大きくとっておく。
    img = np.zeros((B, C, IH + 2 * padding + stride - 1, IW + 2 * padding + stride - 1))

    # 配列を並び替える
    for y in range(FH):
        y_max = y + stride * OH
        for x in range(FW):
            x_max = x + stride * OW
            if is_maxpooling:
                img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
            else:
                img[:, :, y:y_max:stride, x:x_max:stride] = col[:, :, y, x, :, :]

    return img[:, :, padding:IH + padding, padding:IW + padding]  # pad分は除いておく(pad分を除いて真ん中だけを取り出す)

##############################
# TODO 以下、初期実装時に作成したが廃止したユーティリティメソッド。
##############################
# TODO メソッド化してはみたが、実際に使用する側のメソッド内で実装できるため廃止。
# def convolute(x, f, padding, stride):
#     batch_size, channel_size, IH, IW = x.shape
#     FN, FC, FW, FH = f.shape
#     OH = out_size(IH, FH, padding, stride)
#     OW = out_size(IW, FW, padding, stride)
#
#     # 入力とフィルターそれぞれの2次元化。
#     x2d = im2col(x, FH, FW, padding, stride)
#     f2d = im2col_filter(f)
#
#     # 行列積を取る。
#     # 次元構造は、バッチ軸（B個）／フィルタ軸（FN個）×高さ軸（OH個）×幅軸（OW個）の2次元構造である。
#     # print("x2d.shape=", x2d.shape)
#     # print("f2d.shape=", f2d.shape)
#     out2d = np.dot(x2d, f2d.T)
#
#     # 4次元に戻す。
#     #   上記次元構造（2次元）：バッチ軸（B個）／高さ軸（OH個）ー幅軸（OW個）ーフィルタ軸（FN個）
#     #   ⇒4次元化：バッチ軸（B個）／高さ軸（OH個）／幅軸（OW個）／フィルタ軸（FN個）
#     #   ⇒フィルタ軸を先に持ってくる：バッチ軸（B個）／フィルタ軸（FN個）／高さ軸（OH個）／幅軸（OW個）
#     # 　※フィルタ数が新たなチャネル数になることに注意。
#     out = out2d.reshape(batch_size, OH, OW, FN)
#     out = out.transpose(0, 3, 1, 2)
#     return out

# TODO メソッド化してみたが、メモリ消費が過大になるため使用しない。廃止。
# # ------------------------------
# # 同一データ（行、列）挿入メソッド。
# # 与えられたテンソルの与えられた軸の各データの直後に、そのデータのコピーを挿入する。
# # 実際には、引数のxは4次元テンソルを想定し、画像データの高さ軸（第3次元目；次元インデックスは2）、
# # または幅軸（第3次元目；次元インデックスは2）方向の処理を想定している。
# # （例）
# # [
# #     [1,2,3],
# #     [4,5,6],
# #     [7,8,9],
# # ]
# # ↓　軸として第1次元（行方向）、insert_numとして2を指定した場合、以下となる。
# # [
# #     [1,2,3],　元の1行目をコピー。
# #     [1,2,3],　ペースト1回目。
# #     [1,2,3],　ペースト2回目。
# #     [4,5,6],　元の2行目をコピー。
# #     [4,5,6],　ペースト1回目。
# #     [4,5,6],　ペースト2回目。
# #     [7,8,9],　元の3行目をコピー。
# #     [7,8,9],　ペースト1回目。
# #     [7,8,9],　ペースト2回目。
# # ]
# # ------------------------------
# def insert_same_values(x, insert_num=0, axis=2, inner_only=False):
#     if inner_only:
#         org_shape = x.shape[axis] - 1
#     else:
#         org_shape = x.shape[axis]
#
#     for i in range(org_shape):
#         pos = 1 + i * (insert_num + 1)
#         for j in range(insert_num):
#             if axis == 2:
#                 values = x[:,:,pos-1,:]
#             elif axis == 3:
#                 values = x[:,:,:,pos-1]
#             else:
#                 # TODO エラーにするべき。
#                 pass
#             x = np.insert(x, pos, values, axis=axis)
#     return x

# TODO 多重ループを使用していて低速なため廃止。
# ##############################
# # 4階テンソルの2次元化。for文を利用した独自ロジック。低速のため廃止。
# ##############################
# def im2col_ORIGINAL(x, FH,FW, padding, stride):
#     # TODO debug
#     # print("x.shape=", x.shape)
#     # print(x)
#     # print()
#
#     # 入力値行列およびフィルター行列から各次元の要素数を取得する。
#     # ただし、バッチ数B、チャネル数Cはそれぞれ、入力値行列とフィルター行列とで一致していなければならない。
#     B, C, IH, IW = x.shape
#     # TODO debug
#     OH = int(np.floor((IH + 2 * padding - FH) / stride)) + 1
#     OW = int(np.floor((IW + 2 * padding - FW) / stride)) + 1
#     # print("IH,IW={0},{1}".format(IH, IW))
#     # print("FH,FW={0},{1}".format(FH, FW))
#     # print("OH,OW={0},{1}".format(OH, OW))
#     # print()
#
#     # パディングした行列を作る。
#     x_pad_H = IH + 2 * padding
#     x_pad_W = IW + 2 * padding
#     x_pad = np.zeros((B, C, x_pad_H, x_pad_W))
#     for b in range(B):
#         for c in range(C):
#             for h in range(x_pad_H):
#                 if (padding <= h) & (h < padding + IH):  # except padding area
#                     for w in range(x_pad_W):
#                         if (padding <= w) & (w < padding + IW):  # except padding area
#                             x_pad[b][c][h][w] = x[b][c][h - padding][w - padding]
#     # print("x_pad_H, x_pad_W={0},{1}".format(x_pad_H, x_pad_W))
#     # print(x_pad)
#     # print()
#
#     # 2次元化データを作る。
#     x2d_h = B * OH * OW
#     x2d_w = C * FH * FW
#     x2d = np.zeros((x2d_h, x2d_w))
#     # print("x2d_h(=B*OH*OW), x2d_w(=C*FH*FW) = {0},{1}".format(x2d_h, x2d_w))
#     # print()
#
#     for b in range(B):
#         batch_offset = b * OH * OW
#         for c in range(C):
#             channel_offset = c * FH * FW
#             row = batch_offset
#             for i in range(OH):
#                 for j in range(OW):
#                     h0 = stride * i  # 高さ方向の開始位置。
#                     w0 = stride * j  # 幅方向の開始位置。
#                     mat = x_pad[b][c][h0:(h0 + FH), w0:(w0 + FW)]  # filterと同じサイズの行列を抜き出す。
#                     arr = mat.reshape(1, -1)  # 1行、FH*FW列の行列に変形（実体は2次元配列だが、内容としては1次元配列）。
#                     for col in range(arr.shape[1]):
#                         x2d[row][channel_offset + col] = arr[0][col]  # 1行しかないので第1次元は0を指定。
#                     row += 1  # x_pad内のウィンドウを1ストライド動かすたびにカウントアップする。
#
#     return x2d

# TODO 以下独自ロジックにはバグがあるためdeprecated.
# ------------------------------
# 以下エラー内容：
#   File "C:\study\dldev\2_notebook\convolutions.py", line 165, in backward
#     dLdX = col2im(dcol, self.x.shape[0], self.x.shape[1], self.x.shape[2], self.x.shape[3], self.FH, self.FW, self.padding, self.stride)
#   File "C:\study\dldev\2_notebook\convolutions.py", line 336, in col2im
#     x_pad[b][c][h_start + i][w_start + j] = mat3[i][j]
# IndexError: index 27 is out of bounds for axis 0 with size 27
# ------------------------------
# def col2im(x2d, B, C, IH, IW, FH, FW, padding, stride):
#     OH = int(np.floor((IH + 2 * padding - FH) / stride)) + 1
#     OW = int(np.floor((IW + 2 * padding - FW) / stride)) + 1
#
#     # まず行方向をバッチ軸として分ける。（第1次元：バッチ軸。次元のインデックスは0）
#     # 次に各バッチをウィンドウの走査個数ずつ（OH*OW個ある）に分ける。（第2次元：ウィンドウの走査軸。次元のインデックスは1）
#     # さらに各ウィンドウをチャネルの個数ずつに分ける。（第3次元：チャネル軸。次元のインデックスは2）
#     # 残った軸は自動的に決まり、各チャネル内の各ウィンドウ内の走査軸となる。（第4次元：ウィンドウ内の走査軸。次元のインデックスは3）
#     # 最後に、第2次元と第3次元を転置することによって、元の画像データの次元構造に戻る。
#     mat = x2d.reshape(B, OH * OW, C, FH * FW).transpose(0, 2, 1, 3)
#     x_pad = np.zeros((B, C, IH + 2 * padding, IW + 2 * padding))
#     x = np.zeros((B, C, IH, IW))
#     for b in range(B):
#         for c in range(C):
#             h_start = 0
#             w_start = 0
#             mat2 = mat[b][c]
#             for row in range(mat2.shape[0]):
#                 mat3 = mat2[row].reshape(FH, FW)
#                 for i in range(FH):
#                     for j in range(FW):
#                         x_pad[b][c][h_start+i][w_start+j] = mat3[i][j]
#                 if row == (OW - 1):
#                     h_start += stride
#                     w_start = 0
#                 else:
#                     w_start += stride
#             x[b][c] = x_pad[b][c][padding:padding+IH, padding:padding+IW]
#
#     return x

# # ------------------------------
# # 行列の内部0パディング
# # 損失関数をL、畳み込みの結果をOとすると、dL/dWおよびdL/dXを求めるために、dL/dOに内部パディングを入れる必要がある。そのための関数。
# # ------------------------------
# def inner_padding(x, padding_num=0, axis=[2,3]):
#     for i in axis:
#         x = insert_zero(x, padding_num=padding_num, axis=i, inner_only=True)
#     return x
#
# def insert_zero(x, padding_num=0, axis=0, inner_only=False):
#     ZERO = 0
#     if inner_only:
#         org_shape = x.shape[axis] - 1
#     else:
#         org_shape = x.shape[axis]
#
#     for i in range(org_shape):
#         pos = 1 + i * (padding_num + 1)
#         for j in range(padding_num):
#             x = np.insert(x, pos, ZERO, axis=axis)
#     return x
#
# # ------------------------------
# # 180度フリップ。指定軸（array-like）で反転する。
# # ------------------------------
# def flip180(a, axis=[2,3]):
#     for i in axis:
#         a = np.flip(a, axis=i)
#     return a
#

# TODO フィルターの2次元化は1行で済むのでメソッド化の必要なし。
# # ------------------------------
# # フィルターの2次元化。
# # ------------------------------
# def im2col_filter(f):
#     FN, C, FH, FW = f.shape
#     # フィルター軸に沿ってFN個に分割することによって、自動的に（フィルター軸、チャネルの個数×高さの個数×幅の個数）に直列化する。
#     # なぜなら、そもそも、フィルター軸を1つ固定すると、（チャネル軸、高さ軸、幅軸）で分類されているため、
#     # 直列化する順番としては、横軸方向に並ぶ⇒高さ軸方向に並ぶ⇒さらにチャネル軸方向に並ぶ、となるため。
#     f2d = f.reshape(FN, -1)
#
#     # TODO 以下のようにゴリゴリ作らなくても、上記のように第1次元でreshape掛ければ終わりなので以下不要ロジック。
#     # 行方向がフィルター個数の方向、列方向がチャネルの方向（およびフィルター内データ1次元化の方向）とする。
#     # f2d = np.zeros((FN, FH * FW * C))
#     # for fn in range(FN):
#     #     for c in range(C):
#     #         channel_offset = c * FH * FW
#     #         for i in range(FH):
#     #             for j in range(FW):
#     #                 h0 = stride * i  # 高さ方向の開始位置。
#     #                 w0 = stride * j  # 幅方向の開始位置。
#     #                 # mat = f[fn][c][h0:(h0 + FH), w0:(w0 + FW)]  # filterと同じサイズの行列を抜き出す。# TODO 実体はfilterなので不要。以下ロジックで終わり。
#     #                 mat = f[fn][c]
#     #                 arr = mat.reshape(1, -1)  # 1行、FH*FW列の行列に変形（実体は2次元配列だが、内容としては1次元配列）。
#     #                 for col in range(arr.shape[1]):
#     #                     f2d[fn][channel_offset + col] = arr[0][col]  # 1行しかないので第1次元は0を指定。
#     return f2d
