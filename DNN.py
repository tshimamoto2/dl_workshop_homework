import numpy as np
from layers import HiddenLayer, LastLayer, Sigmoid, Tanh, ReLU, BatchNormal, SoftmaxWithLoss, Dropout
from losses import CrossEntropyError
from learners import MiniBatch, KFoldCrossValidation
from optimizers import SGD
import pickle

class DNN:
    def __init__(self,
                 input_size=784,
                 layer_size_list=[100, 5],
                 hidden_actfunc=ReLU(),
                 output_actfunc=SoftmaxWithLoss(),
                 loss_func=CrossEntropyError(),
                 init_weight_stddev=0.01,
                 init_weight_change=False,  # 重みの初期値変更
                 learner=None,  # 学習アルゴリズム（必須）
                 regularization=None,  # 正則化（任意）
                 dropout_params=None,  # ドロップアウト（任意）
                 early_stopping_params=None,  # 早期終了（任意）
                 batch_normal_params=None  # バッチ正規化（任意）
                 ):

        # 保存対象。
        self.input_size = input_size
        self.layer_size_list = layer_size_list
        self.hidden_actfunc = hidden_actfunc
        self.output_actfunc = output_actfunc
        self.loss_func = loss_func
        self.init_weight_stddev = init_weight_stddev
        self.init_weight_change = init_weight_change
        self.learner = learner
        self.regularization = regularization
        self.dropout_params = dropout_params
        self.early_stopping_params = early_stopping_params
        self.batch_normal_params = batch_normal_params

        # 以下保存対象外。
        self.layers = None
        self.W = None
        self.B = None
        self.learner.set_NN(self)

        # TODO debug 提出時はNoneにすること。
        self.act_dist = None
        # self.act_dist = ActivationDistribution()

    def init_weight(self):
        # TODO debug デバッグしやすさのため、再現性があるように指定。
        np.random.seed(1234)
        self.W = []  # 各層の重み配列。要素のインデックスは、層のインデックスと一致。
        self.B = []  # 各層のバイアス配列。要素のインデックスは、層のインデックスと一致。
        prev_size = self.input_size

        for i, crnt_size in enumerate(self.layer_size_list):
            # デフォルトでは指定された標準偏差を使う。
            stddev = self.init_weight_stddev
            if self.init_weight_change:
                # print("★self.init_weight_changeが指定されました。", self.init_weight_change)
                if (self.hidden_actfunc.__class__ == Sigmoid) | (self.hidden_actfunc.__class__ == Tanh):
                    # print("★SigmoidまたはTanhです。Xavierの初期値を使用します。")
                    stddev = np.sqrt(2.0 / (prev_size + crnt_size))
                elif self.hidden_actfunc.__class__ == ReLU:
                    # print("★ReLUです。Heの初期値を使用します。")
                    stddev = np.sqrt(2.0 / prev_size)
                else:
                    # print("★活性化関数がSigmoid、Tanh、ReLU以外です。：self.hidden_actfunc.__class__=", self.hidden_actfunc.__class__)
                    # print("★標準偏差{0}の標準正規分布からの無作為抽出を行います。".format(self.init_weight_stddev))
                    pass
            else:
                #print("★self.init_weight_changeが指定されませんでした。", self.init_weight_change)
                #print("★標準偏差{0}の標準正規分布からの無作為抽出を行います。".format(self.init_weight_stddev))
                pass

            self.W.append(np.random.randn(prev_size, crnt_size) * stddev)
            self.B.append(np.zeros(crnt_size))
            prev_size = crnt_size

    def init_layers(self):
        self.layers = []
        last_index = len(self.layer_size_list) - 1
        for i, layer_size in enumerate(self.layer_size_list):
            input_dropout = None
            hidden_dropout = None
            if self.dropout_params is not None:
                input_dropout = Dropout(retain_rate=self.dropout_params.input_retain_rate)
                hidden_dropout = Dropout(retain_rate=self.dropout_params.hidden_retain_rate)

            # ドロップアウト対応。入力層、隠れ層のドロップアウトの組み合わせがあるので分岐。
            if i == 0:
                layer = HiddenLayer(self.W[i], self.B[i], self.hidden_actfunc, self.batch_normal_params, input_dropout=input_dropout, hidden_dropout=hidden_dropout)
            elif i != last_index:
                layer = HiddenLayer(self.W[i], self.B[i], self.hidden_actfunc, self.batch_normal_params, input_dropout=None, hidden_dropout=hidden_dropout)
            else:
                layer = LastLayer(self.W[i], self.B[i], self.output_actfunc)
            self.layers.append(layer)

    def fit(self, train_data, train_label):
        # 初期化。
        self.init_weight()
        self.init_layers()

        # 学習。
        self.learner.learn(train_data, train_label)

    def gradient(self, x, t):
        # 順伝播。
        y, loss, accuracy = self.predict(x, t, is_learning=True)

        # 逆伝播。
        dout = 1
        for i, layer in enumerate(reversed(self.layers)):
            dout = layer.backward(dout)

    # 順伝播による出力層、損失、精度の算出。
    def predict(self, x, t, is_learning=False):
        # 順伝播。
        z = x
        for i, layer in enumerate(self.layers):
            z = layer.forward(z, t, is_learning)
            # 各レイヤーのアクティベーション分布を保持。
            if self.act_dist is not None:
                self.act_dist.put("layer" + str(i), layer.act_dist)

        # 全レイヤーのアクティベーション分布を保存。
        if self.act_dist is not None:
            self.act_dist.finish_1st()
            self.act_dist.dump()

        # 教材での説明と変数名を合わせただけ。
        y = z

        # 正則化。
        regular = 0.0
        if self.regularization is not None:
            regular = self.regularization.regular(self.layers)

        # 損失、精度の計算。
        loss = self.loss_func.loss(y, t) + regular
        accuracy = self.accuracy(y, t)

        return y, loss, accuracy

    def accuracy(self, y, t):
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        return np.sum(y == t) / y.shape[0]

    # 数値微分。勾配確認（誤差逆伝播法での微分値との比較）のために使用。
    def numerical_gradient(self, x, t):
        def f(W):
            return self.loss(x, t)

        for i, layer in enumerate(self.layers):
            layer.affine.numerical_dLdW = self._numerical_gradient(f, layer.affine.W)
            layer.affine.numerical_dLdB = self._numerical_gradient(f, layer.affine.B)

    def _numerical_gradient0(self, f, W):
        h = 1e-4  # 0.0001
        grad = np.zeros_like(W)

        it = np.nditer(W, flags=['multi_index'])

        while not it.finished:
            idx = it.multi_index
            tmp_val = W[idx]

            W[idx] = tmp_val + h
            fxh1 = f(W)

            W[idx] = tmp_val - h
            fxh2 = f(W)
            grad[idx] = (fxh1 - fxh2) / (2 * h)

            W[idx] = tmp_val  # 値を元に戻す

            # 次のindexへ進める
            it.iternext()

        return grad

    # TODO 不要になったのでは？
    # def increment_batch_normal_moving_iter(self):
    #     if self.batch_normal_params is not None:
    #         for layer in self.layers:
    #             if hasattr(layer, "batch_normal"):
    #                 if layer.batch_normal is not None:
    #                     layer.batch_normal.increment_moving_iter()
    #
    # def reset_batch_normal_moving_iter(self):
    #     if self.batch_normal_params is not None:
    #         for layer in self.layers:
    #             if hasattr(layer, "batch_normal"):
    #                 if layer.batch_normal is not None:
    #                     layer.batch_normal.reset_moving_iter()

# TODO debug用：アクティベーション分布を保存するクラス。
class ActivationDistribution:
    savepath_1 = "./act_dist_1.pkl"  # 1st epoch
    savepath_999 = "./act_dist_999.pkl"  # last epoch

    def __init__(self):
        self.act_dist_1_already_saved = False
        self.is_1st_done = False

        self.act_dist_1 = {}
        self.act_dist_999 = {}

    def put(self, key, dist):
        if self.is_1st_done == False:
            self.act_dist_1[key] = dist

        # TODO 最後になるまで毎回上書きしているので無駄。
        self.act_dist_999[key] = dist

    def finish_1st(self):
        self.is_1st_done = True

    def dump(self):
        # 1st epoch
        if self.act_dist_1_already_saved == False:
            with open(ActivationDistribution.savepath_1, 'wb') as f:
                pickle.dump(self.act_dist_1, f)
                f.close()
                self.act_dist_1_already_saved = True

        # TODO 最後になるまで毎回上書きしているので無駄。
        # last epoch
        with open(ActivationDistribution.savepath_999, 'wb') as f:
            pickle.dump(self.act_dist_999, f)
            f.close()
