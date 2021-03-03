import numpy as np
from LearnedModel import LearnedModel
from layers import HiddenLayer, LastLayer, Sigmoid, ReLU, SoftmaxWithLoss, CrossEntropyError
from learners import MiniBatch, KFoldCrossValidation
from optimizers import SGD

class DNN:
    savepath = "./DNN.pkl"

    def __init__(self,
                 input_size=784,
                 layer_size_list=[100, 5],
                 hidden_actfunc=ReLU(),
                 output_actfunc=SoftmaxWithLoss(),
                 loss_func=CrossEntropyError(),
                 init_weight_stddev=0.01,
                 learner=MiniBatch(epoch_num=100, mini_batch_size=100, optimizer=SGD(learning_rate=0.01), is_numerical_gradient=False)):

        # LearnedModelとして保存したい引数の保持。
        self.lm = LearnedModel()
        self.lm.input_size = input_size
        self.lm.layer_size_list = layer_size_list
        self.lm.init_weight_stddev = init_weight_stddev
        self.lm.epoch_num = learner.epoch_num if hasattr(learner, 'epoch_num') else None
        self.lm.mini_batch_size = learner.mini_batch_size if hasattr(learner, 'mini_batch_size') else None
        self.lm.learning_rate = learner.optimizer.learning_rate if hasattr(learner.optimizer, 'learning_rate') else 0.01
        self.lm.W = None
        self.lm.B = None

        # 以下保存対象外。
        self.hidden_actfunc = hidden_actfunc
        self.output_actfunc = output_actfunc
        self.layers = None
        self.loss_func = loss_func
        self.learner = learner
        self.learner.set_NN(self)

    def init_weight(self):
        # TODO debug デバッグしやすさのため、再現性があるように指定。
        np.random.seed(1)
        self.lm.W = []  # 各層の重み配列。要素のインデックスは、層のインデックスと一致。
        self.lm.B = []  # 各層のバイアス配列。要素のインデックスは、層のインデックスと一致。
        prev_size = self.lm.input_size
        for i, crnt_size in enumerate(self.lm.layer_size_list):
            self.lm.W.append(np.random.randn(prev_size, crnt_size) * self.lm.init_weight_stddev)
            self.lm.B.append(np.zeros(crnt_size))
            prev_size = crnt_size

    def init_layers(self):
        self.layers = []
        last_index = len(self.lm.layer_size_list) - 1
        for i, layer_size in enumerate(self.lm.layer_size_list):
            if (i != last_index):
                layer = HiddenLayer(self.lm.W[i], self.lm.B[i], self.hidden_actfunc)
            else:
                layer = LastLayer(self.lm.W[i], self.lm.B[i], self.output_actfunc)
            self.layers.append(layer)

    def fit(self, train_data, train_label):
        # 初期化。
        self.init_weight()
        self.init_layers()

        # 学習。
        self.learner.learn(train_data, train_label)

        # 以下、クラスにする前の古いコード。ただし実験コメントがあるので一旦残しておく。
        #self.learn_mini_batch(train_data, train_label, numerical_gradient)  # ミニバッチ学習。

        # 以下k分割交差検証　※kfold_numは訓練データ数1000を割り切るように設定すること。
        # 10分割　⇒学習進まず。
        # ★kfold_num= 10: Avg.Loss=1.609, Avg.Accuracy=0.197, Max.Accuracy=0.290, Argmax=9
        #self.learn_cross_validation(train_data, train_label, kfold_num=10)

        # 20分割　⇒70%程度までは進んだ。
        # ★kfold_num= 20: Avg.Loss=1.609, Avg.Accuracy=0.306, Max.Accuracy=0.700, Argmax=19
        #self.learn_cross_validation(train_data, train_label, kfold_num=20)

        # 50分割　⇒エポック40（インデックスは39）で100%になった。⇒過学習の気配。。。
        # ★kfold_num= 50: Avg.Loss=1.065, Avg.Accuracy=0.630, Max.Accuracy=1.000, Argmax=39
        # self.learn_cross_validation(train_data, train_label, kfold_num=50)

        # 100分割　⇒エポック19（インデックスは18）で100%になった。⇒過学習の気配。。。
        # ★kfold_num = 100: Avg.Loss = 0.268, Avg.Accuracy = 0.906, Max.Accuracy = 1.000, Argmax = 18
        # ⇒一旦これで提出してみる。
        #self.learn_cross_validation(train_data, train_label, kfold_num=100)

    def gradient(self, x, t):
        # 順伝播。
        y, loss, accuracy = self.predict(x, t)

        # 逆伝播。
        dout = 1
        #dout = loss
        for i, layer in enumerate(reversed(self.layers)):
            dout = layer.backward(dout)

    # 順伝播による出力層、損失、精度の算出。
    # 学習済みモデル、学習済みレイヤーが決まっていることが前提。
    def predict(self, x, t):
        # 順伝播。
        z = x
        for layer in self.layers:
            z = layer.forward(z, t)  # 内部でアフィン変換と活性化関数による変換を行う。

        # 教材での説明と変数名を合わせただけ。
        y = z

        # 損失、精度の計算。
        loss = self.loss_func.loss(y, t)
        accuracy = self.accuracy(y, t)

        return y, loss, accuracy

    def loss(self, x, t):
        y, loss, accuracy = self.predict(x, t)
        return loss

    # 保存された学習済みモデルをロードして予測を行う。
    def predict_with_learned_model(self, model_path, x, t):
        # 学習済みモデル（重み、バイアス）のロード。
        self.lm = LearnedModel().load(model_path)

        # 学習済みモデルを使ってレイヤーを初期化。
        self.init_layers()

        return self.predict(x, t)

    def accuracy(self, y, t):
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        return np.sum(y == t) / y.shape[0]

    def save(self):
        self.lm.save(DNN.savepath)

    def numerical_gradient(self, x, t):
        def f(W):
            return self.loss(x, t)

        for i, layer in enumerate(self.layers):
            layer.affine.numerical_dLdW = self.numerical_gradient0(f, layer.affine.W)
            layer.affine.numerical_dLdB = self.numerical_gradient0(f, layer.affine.B)

    def numerical_gradient0(self, f, W):
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
