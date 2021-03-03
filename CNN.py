import numpy as np
from layers import HiddenLayer, LastLayer, SoftmaxWithLoss, Sigmoid, Tanh, ReLU, BatchNormal, Dropout, Conv, MaxPool
import pickle

class CNN:
    def __init__(self,
                 layers=None,  # レイヤーリスト（必須）
                 loss_func=None,  # 損失関数（必須）
                 learner=None,  # 学習アルゴリズム（必須）
                 regularization=None,  # 正則化（任意）
                 # dropout_params=None,  # ドロップアウト（任意）
                 # batch_normal_params=None  # バッチ正規化（任意）
                 ):

        self.layers = layers
        self.learner = learner
        self.loss_func = loss_func
        self.regularization = regularization
        # self.dropout_params = dropout_params
        # self.batch_normal_params = batch_normal_params

        self.learner.set_NN(self)

    ###############################
    # fit method
    #     train_data
    #       shape[0]: batch size
    #       shape[1]: channel size
    #       shape[2]: image width
    #       shape[3]: image height
    #     train_label
    #       shape[0]: batch size
    #       shape[1]: output neuron size
    ###############################
    def fit(self, train_data, train_label):
        # 学習。
        self.learner.learn(train_data, train_label)

    def gradient(self, x, t):
        # print("★勾配計算")

        # 全レイヤー順伝播。predictを共通メソッドとして使用するが、その結果は逆伝播には不要なので取得はしない。
        # print("★★順伝播(start)")
        self.predict(x, t, is_learning=True)
        # print("★★順伝播(end)")
        # print()

        # 全レイヤー逆伝播。
        # 損失関数Lの最終出力Oによる偏微分dL/dO＝1を逆流させる。
        # print("★★逆伝播(start)")
        prev_dout = 1
        for i, layer in enumerate(reversed(self.layers)):
            dout = layer.backward(prev_dout)
            # print("{0:20s}の逆伝播：output.shape({1}) --> input.shape({2})".format(layer.__class__.__name__, dout.shape, x.shape))
            prev_dout = dout
        # print("★★逆伝播(end)")
        # print()

    def predict(self, x, t, is_learning=False):
        z = x
        for layer in self.layers:
            z = layer.forward(z, t, is_learning)
            # print("{0:20s}の順伝播：input.shape({1}) --> output.shape({2})".format(layer.__class__.__name__, x.shape, z.shape))

        # 正則化項の算出。
        regular = 0.0
        if self.regularization is not None:
            regular = self.regularization.regular(self.layers)

        # 損失、精度の計算。
        loss = self.loss_func.loss(z, t) + regular
        accuracy = self.accuracy(z, t)

        return z, loss, accuracy

    def accuracy(self, y, t):
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        return np.sum(y == t) / y.shape[0]
