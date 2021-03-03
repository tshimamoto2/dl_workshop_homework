import sys
import numpy as np
from optimizers import SGD
import csv
import sys
import numpy as np
from optimizers import SGD
import csv

class EarlyStoppingParams:
    def __init__(self, early_stopping_patience=5, eps=1.0e-4):
        self.early_stopping_patience = early_stopping_patience
        self.eps = eps

def is_early_stopping(early_stopping_params, prev_loss, loss, worsen_count):
    next_prev_loss = loss
    next_worsen_count = worsen_count
    if early_stopping_params is not None:
        if (loss > prev_loss) & (np.abs(loss - prev_loss) > early_stopping_params.eps):
            next_worsen_count = worsen_count + 1
            if worsen_count > early_stopping_params.early_stopping_patience:
                return True, next_prev_loss, next_worsen_count
        else:
            next_worsen_count = 0
    return False, next_prev_loss, next_worsen_count

##############################
# ミニバッチ＋ホールドアウト法
##############################
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

        # 検証データの個数を算出する。
        # TODO 「検証データ取り分け率」は20%固定とした。
        holdout_rate = 0.2
        v_num = int(np.ceil(train_data.shape[0] * holdout_rate))

        # インデックスを分離。
        v_mask = shuffled_indexes[0:v_num]  # 最初のv_num個を検証データとする。
        l_mask = shuffled_indexes[v_num:]  # それ以上のインデックスのデータを訓練データとする。

        # 検証データを取り分ける。
        v_data = train_data[v_mask]
        v_label = train_label[v_mask]

        # 残りを訓練データとする。
        l_data = train_data[l_mask]
        l_label = train_label[l_mask]

        # ミニバッチ回数を算出。
        # 例）訓練データが800個で、ミニバッチサイズが20なら、ミニバッチ回数は40となる。
        mini_batch_num = int(np.ceil(l_data.shape[0] / self.mini_batch_size))

        # 各エポックでの性能。
        l_loss_list = []
        l_accuracy_list = []
        v_loss_list = []
        v_accuracy_list = []
        perform_csv_list = []

        # 早期終了用の前エポックでの損失の初期化。前エポックの損失よりも低くなれば更新するので、floatの最大値を初期値としておく。
        prev_loss = sys.float_info.max
        worsen_count = 0
        stop_epoch = -1

        # 指定したエポック回数分ループ。
        for i in range(self.epoch_num):
            # print("★epoch[{0}]開始".format(i))
            stop_epoch = i

            # 訓練データのインデックスをシャッフル。
            shuffled_indexes = np.arange(l_data.shape[0])
            np.random.shuffle(shuffled_indexes)

            # ミニバッチのループ。
            # 分割された各訓練データごとに学習（ミニバッチ学習）を行う。
            for j in range(mini_batch_num):
                # TODO 不要になったのでは？
                # # バッチ正規化のイテレーションをミニバッチごとに1加算。
                # if self.nn.batch_normal_params is not None:
                #     self.nn.increment_batch_normal_moving_iter()

                # 各ミニバッチで使用するデータのインデックスを算出。
                # ミニバッチごとに、ミニバッチサイズずつずらしていく。
                mask = shuffled_indexes[(self.mini_batch_size * j): (self.mini_batch_size * (j + 1))]

                # 勾配計算。
                self.nn.gradient(l_data[mask], l_label[mask])
                if self.is_numerical_gradient:
                    self.nn.numerical_gradient(l_data[mask], l_label[mask])

                # 重みの更新。
                # ただし、その前に、正則化がある場合は各レイヤーのAffineのdLdWを正則化項の偏微分で更新しておく。
                if self.nn.regularization is not None:
                    self.nn.regularization.update_dLdW(self.nn.layers)
                self.optimizer.update(self.nn)

            # 全ミニバッチが終わった時点で（つまり1エポックが終わった時点で）、各エポックの性能を算出。
            # 訓練データを元に算出した性能の算出。
            l_y, l_loss, l_accuracy = self.nn.predict(l_data, l_label, is_learning=False)
            l_loss_list.append(l_loss)
            l_accuracy_list.append(l_accuracy)

            # 検証データを元に算出した性能の算出。
            v_y, v_loss, v_accuracy = self.nn.predict(v_data, v_label, is_learning=False)
            v_loss_list.append(v_loss)
            v_accuracy_list.append(v_accuracy)

            print("★epoch[{0}]終了：l_loss={1:.4f}, l_accuracy={2:.4f}, v_loss={3:.4f}, v_accuracy={4:.4f}".format(i, l_loss, l_accuracy, v_loss, v_accuracy))

            perform_csv_list.append(list([i, "{0:.4f}".format(l_loss), "{0:.4f}".format(l_accuracy), "{0:.4f}".format(v_loss), "{0:.4f}".format(v_accuracy)]))

            # 早期終了判定。
            if self.early_stopping_params is not None:
                is_stop, prev_loss, worsen_count = is_early_stopping(self.early_stopping_params, prev_loss, v_loss, worsen_count)
                if is_stop:
                    # TODO debug
                    print("is_stop={0}, prev_loss={1}, worsen_count={2}, list(perform_csv_list)={3}".format(is_stop, prev_loss, worsen_count, list(perform_csv_list)))
                    break

        # 平均を見てみる。
        print("★Avg.l_loss={0:.4f}, Avg.l_accuracy={1:.4f}, Max.l_accuracy={2:.4f}, Avg.l_argmax={3} | "
              "Avg.v_loss={4:.4f}, Avg.v_accuracy={5:.4f}, Max.v_accuracy={6:.4f}, Avg.v_argmax={7}".format(
            np.average(l_loss_list), np.average(l_accuracy_list), np.max(l_accuracy_list), np.average(np.argmax(l_accuracy_list)),
            np.average(v_loss_list), np.average(v_accuracy_list), np.max(v_accuracy_list), np.average(np.argmax(v_accuracy_list))
        ))
        print()

        # 推移の保存
        with open('perform.csv', 'w') as f:
            w = csv.writer(f, lineterminator='\n')
            w.writerow(list(["epoch", "l_loss", "l_accuracy", "v_loss", "v_accuracy"]))
            for i in range(stop_epoch):
                w.writerow(perform_csv_list[i])

##############################
# k分割交差検証
##############################
class KFoldCrossValidation:
    def __init__(self, epoch_num, kfold_num, optimizer=SGD(learning_rate=0.01), early_stopping_params=None):
        self.epoch_num = epoch_num
        self.kfold_num = kfold_num
        self.optimizer = optimizer
        self.early_stopping_params = early_stopping_params

    def set_NN(self, nn):
        self.nn = nn

    # k分割交差検証（k-Fold Cross Validation）による学習。
    # ★前提★：kの値は、データ数を割り切るような値にすること。
    def learn(self, train_data, train_label):
        # 1個分のデータ個数
        each_data_num = train_data.shape[0] // self.kfold_num

        # 各エポックでの性能。
        l_loss_list = []
        l_accuracy_list = []
        v_loss_list = []
        v_accuracy_list = []
        perform_csv_list = []

        # 早期終了用の前エポックでの損失の初期化。前エポックの損失よりも低くなれば更新するので、floatの最大値を初期値としておく。
        prev_avg_loss = sys.float_info.max
        worsen_count = 0

        # 指定したエポック回数分ループ。
        for i in range(self.epoch_num):
            # 訓練データのインデックスをシャッフル。
            shuffled_indexes = np.arange(train_data.shape[0])
            np.random.shuffle(shuffled_indexes)

            # 訓練データのシャッフル後インデックスをk分割する。
            mask = []
            for k in range(self.kfold_num):
                mask.append(shuffled_indexes[(each_data_num * k): (each_data_num * (k + 1))])

            # 分割個数分のループ。
            for k in range(self.kfold_num):
                #print("k=", k)

                # 分割データのうちインデックス[k]のものを検証データとして取り分け（hold out）、残りの分割データを使って学習を行う。
                dup_data = train_data.copy()
                dup_label = train_label.copy()
                v_data = dup_data[mask[k]]
                v_label = dup_label[mask[k]]
                l_data = np.delete(dup_data, mask[k], axis=0)
                l_label = np.delete(dup_label, mask[k], axis=0)

                for j in range(self.kfold_num):
                    #print("j=", j)
                    if j == k:
                        continue

                    # 勾配計算。
                    self.nn.gradient(l_data, l_label)

                    # 重みの更新。
                    # ただし、その前に、正則化がある場合は各レイヤーのdLdWを正則化項の偏微分で更新しておく。
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

            # 各エポックでの平均性能を取る。
            l_avg_loss = np.average(l_loss_list)
            l_avg_accuracy = np.average(l_accuracy_list)
            l_max_accuracy = np.max(l_accuracy_list)

            v_avg_loss = np.average(v_loss_list)
            v_avg_accuracy = np.average(v_accuracy_list)
            v_max_accuracy = np.max(v_accuracy_list)

            print("★epoch[{0}]終了："
                  "Avg.l_loss={1:.4f}, Avg.l_accuracy={2:.4f}, Max.l_accuracy={3:.4f} | "
                  "Avg.v_loss={4:.4f}, Avg.v_accuracy={5:.4f}, Max.v_accuracy={6:.4f}".format(
                i, l_avg_loss, l_avg_accuracy, l_max_accuracy, v_avg_loss, v_avg_accuracy, v_max_accuracy
            ))
            print()

            perform_csv_list.append(list([i, "{0:.4f}".format(l_avg_loss), "{0:.4f}".format(l_avg_accuracy),
                                          "{0:.4f}".format(v_avg_loss), "{0:.4f}".format(v_avg_accuracy)]))

            l_loss_list.clear()
            l_accuracy_list.clear()
            v_loss_list.clear()
            v_accuracy_list.clear()
            mask.clear()

            # 早期終了判定。
            if self.early_stopping_params is not None:
                is_stop, prev_avg_loss, worsen_count = is_early_stopping(self.early_stopping_params, prev_avg_loss, l_avg_loss, worsen_count)
                if is_stop:
                    break

        # 推移の保存
        with open('perform.csv', 'w') as f:
            w = csv.writer(f, lineterminator='\n')
            w.writerow(list(["epoch", "l_loss", "l_accuracy", "v_loss", "v_accuracy"]))
            for i in range(self.epoch_num):
                w.writerow(perform_csv_list[i])
