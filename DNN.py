import numpy as np
from LearnedModel import LearnedModel
from layers import HiddenLayer, LastLayer
from loss import CrossEntropyError

class DNN:
    # TODO 保存ファイル名固定でよいか？将来の拡張性を考えて__class__を使うべきでは？
    savepath = "./DNN.pkl"

    def __init__(self, input_size, layer_size_list):
        # 各引数の保持。
        self.lm = LearnedModel()
        self.lm.input_size = input_size
        self.lm.layer_size_list = layer_size_list
        self.lm.init_weight_stddev = None
        self.lm.epoch_num = None
        self.lm.mini_batch_size = None
        self.lm.learning_rate = None
        self.lm.W = None
        self.lm.B = None

        # 以下保存対象外。
        self.layers = None
        self.lossfunc = CrossEntropyError()  # TODO クロスエントロピー誤差関数固定にしている。切り替えれるようにしたい。

    def init_weight(self):
        np.random.seed(1)  # TODO debug デバッグしやすさのため、再現性があるように指定。
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
                layer = HiddenLayer(self.lm.W[i], self.lm.B[i])
            else:
                layer = LastLayer(self.lm.W[i], self.lm.B[i])
            self.layers.append(layer)

    def fit(self, train_data, train_label, init_weight_stddev, epoch_num, mini_batch_size, learning_rate):
        # 各引数の保持。
        self.lm.init_weight_stddev = init_weight_stddev
        self.lm.epoch_num = epoch_num
        self.lm.mini_batch_size = mini_batch_size
        self.lm.learning_rate = learning_rate

        # 初期化。
        self.init_weight()
        self.init_layers()

        # 学習。
        # TODO 学習ロジックを切り替えられるようにすること。
        #self.learn_mini_batch(train_data, train_label)  # ミニバッチ学習。

        # 以下k分割交差検証（ただし1個抜き交差検証）
        # ※kfold_numは訓練データ数1000を割り切るように設定すること。

        # 10分割
        # 学習進まず。
        # ★kfold_num= 10: Avg.Loss=1.609, Avg.Accuracy=0.197, Max.Accuracy=0.290, Argmax=9
        #self.learn_cross_validation(train_data, train_label, kfold_num=10)

        # 20分割
        # 70%程度までは進んだ。
        # ★kfold_num= 20: Avg.Loss=1.609, Avg.Accuracy=0.306, Max.Accuracy=0.700, Argmax=19
        #self.learn_cross_validation(train_data, train_label, kfold_num=20)

        # 50分割
        # エポック40（インデックスは39）で100%になった。
        # ★kfold_num= 50: Avg.Loss=1.065, Avg.Accuracy=0.630, Max.Accuracy=1.000, Argmax=39
        # ⇒過学習の気配。。。
        # self.learn_cross_validation(train_data, train_label, kfold_num=50)

        # 100分割
        # エポック19（インデックスは18）で100%になった。
        # ★kfold_num = 100: Avg.Loss = 0.268, Avg.Accuracy = 0.906, Max.Accuracy = 1.000, Argmax = 18
        # ⇒過学習の気配。。。
        # ⇒一旦これで提出してみる。
        self.learn_cross_validation(train_data, train_label, kfold_num=100)

    def learn_mini_batch(self, train_data, train_label):
        # ミニバッチ回数を算出。
        # 例）訓練データが1000個で、ミニバッチサイズが100なら、訓練データを10分割したことなる。
        mini_batch_num = int(np.ceil(train_data.shape[0] / self.lm.mini_batch_size))

        # 指定したエポック回数分ループ。
        loss_list = []
        accuracy_list = []
        for i in range(self.lm.epoch_num):
            # print("★epoch[{0}]開始".format(i))

            # 訓練データのインデックスをシャッフル。
            shuffled_indexes = np.arange(train_data.shape[0])
            np.random.shuffle(shuffled_indexes)  # TODO debug この行をコメントアウトしてシャッフル無しにすると、デバッグしやすい。

            # 分割された各訓練データごとに学習（ミニバッチ学習）を行う。
            for j in range(mini_batch_num):
                # ミニバッチサイズだけ訓練データ、教師データを抽出。
                mask = shuffled_indexes[(self.lm.mini_batch_size * j) : (self.lm.mini_batch_size * (j + 1))]

                # 勾配計算。
                self.gradient(train_data[mask], train_label[mask])

                # 重みの更新。
                self.update_weight()

            # エポックごとの精度を表示。ただし、訓練データを元に算出。
            y, loss, accuracy = self.predict(train_data, train_label)
            loss_list.append(loss)
            accuracy_list.append(accuracy)
            print("★epoch[{0}]終了 loss={1:.3f}, accuracy={2:.3f}".format(i, loss, accuracy))

        # 平均を見てみる。
        # TODO debug
        print("★Avg.Loss={0:.3f}, Avg.Accuracy={1:.3f}, Max.Accuracy={2:.3f}, Argmax={3}".format(
            np.average(loss_list), np.average(accuracy_list), np.max(accuracy_list), np.argmax(accuracy_list)))
        print()

    # k分割交差検証（k-Fold Cross Validation）による学習。
    # なお、k分割交差検証の1種である『1個抜き交差検証（Leave-One-Out:LOO）』を実装。
    # ★前提★：kの値は、データ数を割り切るような値にすること。
    def learn_cross_validation(self, train_data, train_label, kfold_num):
        # 訓練データのインデックスをシャッフル。
        shuffled_indexes = np.arange(train_data.shape[0])
        np.random.shuffle(shuffled_indexes)  # TODO debug この行をコメントアウトしてシャッフル無しにすると、デバッグしやすい。

        # 1個分のデータ個数
        each_data_num = int(np.floor(train_data.shape[0] / kfold_num))
        # TODO debug
        # print("each_data_num=", each_data_num)

        # 訓練データのk分割。
        split_data = []
        split_label = []
        for k in range(kfold_num):
            mask = shuffled_indexes[(each_data_num * k) : (each_data_num * (k + 1))]
            split_data.append(train_data[mask])
            split_label.append(train_label[mask])

        # 分割個数分のループ。
        loss_list = []
        accuracy_list = []
        for k in range(kfold_num):
            # TODO debug
            # print("★epoch[{0}]開始".format(k))
            # 分割データのうちインデックスkのものを検証データとし、残りの分割データを使って学習を行う。
            for j in range(kfold_num):
                if (j == k):
                    continue

                # 勾配計算。
                self.gradient(split_data[j], split_label[j])

                # 重みの更新。
                self.update_weight()

            # 取り分けておいた検証データを使ってエポックごとの精度を表示する。
            y, loss, accuracy = self.predict(split_data[k], split_label[k])
            loss_list.append(loss)
            accuracy_list.append(accuracy)
            # TODO debug
            print("★epoch[{0}]終了 loss={1:.3f}, accuracy={2:.3f}".format(k, loss, accuracy))

        # 平均を見てみる。
        # TODO debug
        print("★kfold_num={0:3d}: Avg.Loss={1:.3f}, Avg.Accuracy={2:.3f}, Max.Accuracy={3:.3f}, Argmax={4}".format(
            kfold_num, np.average(loss_list), np.average(accuracy_list), np.max(accuracy_list), np.argmax(accuracy_list)))
        print()

    def gradient(self, x, t):
        # print("    gradient start")

        # 順伝播。
        y, loss, accuracy = self.predict(x, t)

        # 逆伝播。
        dout = 1
        #dout = loss
        for i, layer in enumerate(reversed(self.layers)):
            dout = layer.backward(dout)

        # print("    gradient end : dout.shape=", dout.shape)

    def update_weight(self):
        # print("    update_weight start")
        for i, layer in enumerate(self.layers):
            # # TODO debug 第1層の重みがW[0]の中で、値が大きい場所を探す。
            # # TODO ⇒ただし、ミニバッチごとに毎回シャッフルされたデータを見ているので、探しても意味はない。
            # # TODO ⇒どうしても探したいのであればシャッフルをやめてみるとよいのでは？
            # if (i==0):
            #     maxval = np.max(layer.affine.dLdW)
            #     argmaxpos = np.argmax(layer.affine.dLdW)
            #     row = int(np.floor(argmaxpos / layer.affine.dLdW.shape[1]))
            #     col = argmaxpos % layer.affine.dLdW.shape[1]
            #     prev_W = layer.affine.W[row][col]
            #     # print("    self.layers[{0}].affine.dLdWの最大値={1:10.7f}, 同argmax(dLdW)={2}({3},{4}), その位置のW({3},{4})の値={5:10.7f}".format(
            #     #     i, maxval, argmaxpos, argmaxposrow, argmaxposcol, layer.affine.W[argmaxposrow][argmaxposcol]))

            # # TODO debug
            # if (i==0):
            #     row = 544
            #     col = 14
            #     prev_W = layer.affine.W[row][col]

            # W = W - 学習率 * dLdW
            layer.affine.W -= self.lm.learning_rate * layer.affine.dLdW
            layer.affine.B -= self.lm.learning_rate * layer.affine.dLdB

            # # TODO debug
            # if (i==0):
            #     print("    self.layers[{0:3d}]: layer.affine.W[{1:3d}][{2:3d}] : {3:10.7f} -> {4:10.7f} by dLdW[{1:3d}][{2:3d}]={5:10.7f}".format(i, row, col, prev_W, layer.affine.W[row][col], layer.affine.dLdW[row][col]))
            #
            # TODO debug
            # print("    self.layers[{0}]: layer.affine.W[{1}][{2}]= 更新前{3:10.7f} -> 更新後{4:10.7f} by 微分値dLdW[{1}][{2}]={5:10.7f}".format(
            #     i, argmaxposrow, argmaxposcol, prev_W, layer.affine.W[argmaxposrow][argmaxposcol], layer.affine.dLdW[argmaxposrow][argmaxposcol]))
            # print("    update_weight end")

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
        loss = self.lossfunc.loss(y, t)
        accuracy = self.accuracy(y, t)

        return y, loss, accuracy

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
