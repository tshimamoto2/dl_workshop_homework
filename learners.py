import sys
import numpy as np
from optimizers import SGD
import csv

class EarlyStoppingParams:
    def __init__(self, early_stopping_patience=10):
        self.early_stopping_patience = early_stopping_patience

def is_early_stopping(early_stopping_params, prev_loss, loss, worsen_count):
    if early_stopping_params is not None:
        if loss > prev_loss:
            worsen_count += 1
            if worsen_count > early_stopping_params.early_stopping_patience:
                is_stop = True
        else:
            worsen_count = 0
        prev_loss = loss
        return is_stop, prev_loss, worsen_count

class MiniBatch:
    def __init__(self, epoch_num=100, mini_batch_size=100, optimizer=SGD(learning_rate=0.01), is_numerical_gradient=False, early_stopping_params=None):
        self.epoch_num = epoch_num
        self.mini_batch_size = mini_batch_size
        self.optimizer = optimizer
        self.is_numerical_gradient = is_numerical_gradient
        self.early_stopping_params = early_stopping_params

    def set_NN(self, nn):
        self.nn = nn

    def learn(self, train_data, train_label):
        # 検証データの分割。交差検証のために、元データの20%を分割して検証データとする。
        shuffled_indexes = np.arange(train_data.shape[0])
        np.random.shuffle(shuffled_indexes)

        # TODO 「検証データ取り分け率」は20%固定とした。
        holdout_rate = 0.2
        v_num = int(np.ceil(train_data.shape[0] * holdout_rate))
        mask = shuffled_indexes[0:v_num]
        v_data = train_data[mask]
        v_label = train_label[mask]

        # 検証データを除いた分を訓練データとする。
        mask = shuffled_indexes[v_num:]
        train_data = train_data[mask]
        train_label = train_label[mask]

        # ミニバッチ回数を算出。
        # 例）訓練データが1000個で、ミニバッチサイズが100なら、訓練データを10分割したことなる。
        mini_batch_num = int(np.ceil(train_data.shape[0] / self.mini_batch_size))

        # 各エポックの性能。
        l_loss_list = []
        l_accuracy_list = []
        v_loss_list = []
        v_accuracy_list = []

        # 早期終了用の前エポックでの損失の初期化。前エポックの損失よりも低くなれば更新するので、floatの最大値を初期値としておく。
        prev_loss = sys.float_info.max
        worsen_count = 0

        # 指定したエポック回数分ループ。
        for i in range(self.epoch_num):
            # print("★epoch[{0}]開始".format(i))

            # 訓練データのインデックスをシャッフル。
            shuffled_indexes = np.arange(train_data.shape[0])
            np.random.shuffle(shuffled_indexes)

            # 分割された各訓練データごとに学習（ミニバッチ学習）を行う。
            for j in range(mini_batch_num):
                # TODO 不要になったのでは？
                # # バッチ正規化のイテレーションをミニバッチごとに1加算。
                # if self.nn.batch_normal_params is not None:
                #     self.nn.increment_batch_normal_moving_iter()

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

            # TODO 不要になったのでは？
            # # 1エポックごとにバッチ正規化の移動平均カウントをクリアする。
            # if self.nn.batch_normal_params is not None:
            #     self.nn.reset_batch_normal_moving_iter()

            # エポックごとの性能の算出。
            # 訓練データを元に算出した性能（損失値と正解率）。
            l_y, l_loss, l_accuracy = self.nn.predict(train_data, train_label, is_learning=False)
            l_loss_list.append(l_loss)
            l_accuracy_list.append(l_accuracy)

            # 検証データを元に算出した性能（損失値と正解率）。
            v_y, v_loss, v_accuracy = self.nn.predict(v_data, v_label, is_learning=False)
            v_loss_list.append(v_loss)
            v_accuracy_list.append(v_accuracy)

            print("★epoch[{0}]終了：l_loss={1:.4f}, l_accuracy={2:.4f}, v_loss={3:.4f}, v_accuracy={4:.4f}".format(i, l_loss, l_accuracy, v_loss, v_accuracy))

            # 早期終了判定。
            if self.early_stopping_params is not None:
                is_stop, prev_loss, worsen_count = is_early_stopping(self.early_stopping_params, prev_loss, v_loss, worsen_count)
                if is_stop:
                    break

        # 平均を見てみる。
        print("★Avg.l_loss={0:.4f}, Avg.l_accuracy={1:.4f}, Max.l_accuracy={2:.4f}, l_argmax={3} | "
              "Avg.v_loss={4:.4f}, Avg.v_accuracy={5:.4f}, Max.v_accuracy={6:.4f}, v_argmax={7}".format(
            np.average(l_loss_list), np.average(l_accuracy_list), np.max(l_accuracy_list), np.argmax(l_accuracy_list),
            np.average(v_loss_list), np.average(v_accuracy_list), np.max(v_accuracy_list), np.argmax(v_accuracy_list)
        ))
        print()

class KFoldCrossValidation:
    def __init__(self, kfold_num, optimizer=SGD(learning_rate=0.01), early_stopping_params=None):
        self.kfold_num = kfold_num
        self.optimizer = optimizer
        self.early_stopping_params = early_stopping_params

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

        # 各エポックの性能。
        l_loss_list = []
        l_accuracy_list = []
        v_loss_list = []
        v_accuracy_list = []
        perform_csv_list = []

        # 早期終了用の前エポックでの損失の初期化。前エポックの損失よりも低くなれば更新するので、floatの最大値を初期値としておく。
        prev_loss = sys.float_info.max
        worsen_count = 0

        # 分割個数分のループ。
        for k in range(self.kfold_num):
            # 分割データのうちインデックス[k]のものを検証データとして取り分け（hold out）、残りの分割データを使って学習を行う。
            v_data = split_data[k]
            v_label = split_label[k]

            # 分割データのうちインデックス[k]以外のものを、訓練データでの性能算出のために配列として保持しておく。
            l_data = np.delete(split_data.copy(), k, axis=0).reshape(-1, train_data.shape[1])
            l_label = np.delete(split_label.copy(), k, axis=0).reshape(-1, train_label.shape[1])

            for j in range(self.kfold_num):
                if j == k:
                    continue

                # 勾配計算。
                self.nn.gradient(split_data[j], split_label[j])

                # 重みの更新。
                # ただし、その前に、正則化がある場合は各レイヤーのAffineのdLdWを正則化項の偏微分で更新しておく。
                if self.nn.regularization is not None:
                    self.nn.regularization.update_dLdW(self.nn.layers)
                self.optimizer.update(self.nn)

            # 訓練データを元に算出した性能（損失値と正解率）。
            l_y, l_loss, l_accuracy = self.nn.predict(l_data, l_label, is_learning=False)
            l_loss_list.append(l_loss)
            l_accuracy_list.append(l_accuracy)

            # 取り分けておいた検証データを使ってエポックごとの性能を算出。
            v_y, v_loss, v_accuracy = self.nn.predict(v_data, v_label, is_learning=False)
            v_loss_list.append(v_loss)
            v_accuracy_list.append(v_accuracy)

            print("★epoch[{0}]終了：l_loss={1:.4f}, l_accuracy={2:.4f}, v_loss={3:.4f}, v_accuracy={4:.4f}".format(k, l_loss, l_accuracy, v_loss, v_accuracy))
            # print("★epoch[{0}]終了 loss={1:.4f}, accuracy={2:.4f}".format(k, v_loss, v_accuracy))

            perform_csv_list.append(list([k, "{0:.4f}".format(l_loss), "{0:.4f}".format(l_accuracy), "{0:.4f}".format(v_loss), "{0:.4f}".format(v_accuracy)]))

            # 早期終了判定。
            if self.early_stopping_params is not None:
                is_stop, prev_loss, worsen_count = is_early_stopping(self.early_stopping_params, prev_loss, v_loss, worsen_count)
                if is_stop:
                    break

        # 平均を見てみる。
        print("★Avg.l_loss={0:.4f}, Avg.l_accuracy={1:.4f}, Max.l_accuracy={2:.4f}, l_argmax={3} | "
              "Avg.v_loss={4:.4f}, Avg.v_accuracy={5:.4f}, Max.v_accuracy={6:.4f}, v_argmax={7}".format(
            np.average(l_loss_list), np.average(l_accuracy_list), np.max(l_accuracy_list), np.argmax(l_accuracy_list),
            np.average(v_loss_list), np.average(v_accuracy_list), np.max(v_accuracy_list), np.argmax(v_accuracy_list)
        ))
        print()
        # print("★kfold_num={0:3d}: Avg.Loss={1:.4f}, Avg.Accuracy={2:.4f}, Max.Accuracy={3:.4f}, Argmax={4}".format(
        #     self.kfold_num, np.average(v_loss_list), np.average(accuracy_list), np.max(accuracy_list), np.argmax(accuracy_list)))
        # print()

        # 推移の保存
        with open('perform.csv', 'w') as f:
            w = csv.writer(f, lineterminator='\n')
            w.writerow(list(["epoch", "l_loss", "l_accuracy", "v_loss", "v_accuracy"]))
            for k in range(self.kfold_num):
                w.writerow(perform_csv_list[k])

