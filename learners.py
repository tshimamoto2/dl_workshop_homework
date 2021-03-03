import sys
import numpy as np
from optimizers import SGD

class MiniBatch:
    def __init__(self, epoch_num=100, mini_batch_size=100, optimizer=SGD(learning_rate=0.01), is_numerical_gradient=False):
        self.epoch_num = epoch_num
        self.mini_batch_size = mini_batch_size
        self.optimizer = optimizer

        self.is_numerical_gradient = is_numerical_gradient

    def set_NN(self, nn):
        self.nn = nn

    def learn(self, train_data, train_label):
        # 検証データの分割。交差検証のために、元データの20%を分割して検証データとする。
        shuffled_indexes = np.arange(train_data.shape[0])
        np.random.shuffle(shuffled_indexes)

        test_num = int(np.ceil(train_data.shape[0]*0.2))
        mask = shuffled_indexes[0:test_num]
        test_data = train_data[mask]
        test_label = train_label[mask]

        # 検証データを除いた分を訓練データとする。
        mask = shuffled_indexes[test_num:]
        train_data = train_data[mask]
        train_label = train_label[mask]

        # ミニバッチ回数を算出。
        # 例）訓練データが1000個で、ミニバッチサイズが100なら、訓練データを10分割したことなる。
        mini_batch_num = int(np.ceil(train_data.shape[0] / self.mini_batch_size))

        # 指定したエポック回数分ループ。
        loss_list = []
        accuracy_list = []
        test_loss_list = []
        test_accuracy_list = []
        for i in range(self.epoch_num):
            # print("★epoch[{0}]開始".format(i))

            # 訓練データのインデックスをシャッフル。
            shuffled_indexes = np.arange(train_data.shape[0])
            np.random.shuffle(shuffled_indexes)

            # 分割された各訓練データごとに学習（ミニバッチ学習）を行う。
            for j in range(mini_batch_num):
                # ミニバッチサイズだけ訓練データ、教師データを抽出。
                mask = shuffled_indexes[(self.mini_batch_size * j): (self.mini_batch_size * (j + 1))]

                # 勾配計算。
                self.nn.gradient(train_data[mask], train_label[mask])
                if self.is_numerical_gradient:
                    self.nn.numerical_gradient(train_data[mask], train_label[mask])

                # 重みの更新。
                # ただし、その前に、正則化がある場合は各レイヤーのAffineのdLdWを正則化項の偏微分で更新しておく。
                if self.nn.regularization is not None:
                    self.nn.regularization.update_dLdW(self.nn.layers)
                self.optimizer.update(self.nn)

            # エポックごとの精度を表示。
            # 訓練データを元に算出した性能（損失値と正解率）。
            y, loss, accuracy = self.nn.predict(train_data, train_label, is_learning=False)
            loss_list.append(loss)
            accuracy_list.append(accuracy)

            # 検証データを元に算出した性能（損失値と正解率）。
            test_y, test_loss, test_accuracy = self.nn.predict(test_data, test_label, is_learning=False)
            test_loss_list.append(test_loss)
            test_accuracy_list.append(test_accuracy)
            print("★epoch[{0}]終了：loss={1:.3f}, accuracy={2:.3f}, test_loss={3:.3f}, test_accuracy={4:.3f}".format(i, loss, accuracy, test_loss, test_accuracy))

        # 平均を見てみる。
        print("★Avg.loss={0:.3f}, Avg.accuracy={1:.3f}, Max.accuracy={2:.3f}, Argmax={3} | "
              "Avg.test_loss={4:.3f}, Avg.test_accuracy={5:.3f}, Max.test_accuracy={6:.3f}, Argmax={7}".format(
            np.average(loss_list), np.average(accuracy_list), np.max(accuracy_list), np.argmax(accuracy_list),
            np.average(test_loss_list), np.average(test_accuracy_list), np.max(test_accuracy_list), np.argmax(test_accuracy_list),
        ))
        print()

        # optimizerに移行。
        # def update_weight(self):
        #     for layer in self.layers:
        #         layer.affine.W -= self.lm.learning_rate * layer.affine.dLdW
        #         layer.affine.B -= self.lm.learning_rate * layer.affine.dLdB

class KFoldCrossValidation:
    def __init__(self, kfold_num, optimizer=SGD(learning_rate=0.01)):
        self.kfold_num = kfold_num
        self.optimizer = optimizer

    def set_NN(self, nn):
        self.nn = nn

    # k分割交差検証（k-Fold Cross Validation）による学習。
    # ★前提★：kの値は、データ数を割り切るような値にすること。
    def learn(self, train_data, train_label):
        # 訓練データのインデックスをシャッフル。
        shuffled_indexes = np.arange(train_data.shape[0])
        np.random.shuffle(shuffled_indexes)

        # 1個分のデータ個数
        each_data_num = int(np.floor(train_data.shape[0] / self.kfold_num))

        # 訓練データのk分割。
        split_data = []
        split_label = []
        for k in range(self.kfold_num):
            mask = shuffled_indexes[(each_data_num * k): (each_data_num * (k + 1))]
            split_data.append(train_data[mask])
            split_label.append(train_label[mask])

        # 分割個数分のループ。
        loss_list = []
        accuracy_list = []

        # 早期終了用。
        prev_loss = sys.float_info.max
        worsen_count = 0
        for k in range(self.kfold_num):
            # 分割データのうちインデックスkのものを検証データとし、残りの分割データを使って学習を行う。
            for j in range(self.kfold_num):
                if (j == k):
                    continue

                # 勾配計算。
                self.nn.gradient(split_data[j], split_label[j])

                # 重みの更新。
                # ただし、その前に、正則化がある場合は各レイヤーのAffineのdLdWを正則化項の偏微分で更新しておく。
                if self.nn.regularization is not None:
                    self.nn.regularization.update_dLdW(self.nn.layers)
                self.optimizer.update(self.nn)

            # 取り分けておいた検証データを使ってエポックごとの精度を表示する。
            y, loss, accuracy = self.nn.predict(split_data[k], split_label[k], is_learning=False)
            loss_list.append(loss)
            accuracy_list.append(accuracy)
            print("★epoch[{0}]終了 loss={1:.3f}, accuracy={2:.3f}".format(k, loss, accuracy))

            # 早期終了判定。
            if self.nn.early_stopping_params is not None:
                if loss > prev_loss:
                    worsen_count += 1
                    if worsen_count > self.nn.early_stopping_params.early_stopping_patience:
                        break
                else:
                    worsen_count = 0
                prev_loss = loss

        # 平均を見てみる。
        print("★kfold_num={0:3d}: Avg.Loss={1:.3f}, Avg.Accuracy={2:.3f}, Max.Accuracy={3:.3f}, Argmax={4}".format(
            self.kfold_num, np.average(loss_list), np.average(accuracy_list), np.max(accuracy_list), np.argmax(accuracy_list)))
        print()

class EarlyStoppingParams:
    def __init__(self, early_stopping_patience=10):
        self.early_stopping_patience = early_stopping_patience

