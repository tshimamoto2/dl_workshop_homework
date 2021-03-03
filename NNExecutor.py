import numpy as np
from DNN import DNN
from layers import Sigmoid, Tanh, ReLU, SoftmaxWithLoss, DropoutParams, BatchNormalParams
from losses import CrossEntropyError
from learners import MiniBatch, KFoldCrossValidation, EarlyStoppingParams
from optimizers import SGD, Momentum, AdaGrad, AdaDelta, RMSProp, Adam, NAG
from regularizations import L2
import pickle

class NNExecutor:
    def __init__(self):
        self.nn = None
        pass

    def fit(self, model_save_path, train_data, train_label):
        self.nn = self.create_model()
        self.nn.fit(train_data=train_data, train_label=train_label)
        self.save_model(model_save_path)

    def predict(self, model_save_path, test_data, test_label):
        self.nn = self.load_model(model_save_path)
        y, loss, accuracy = self.nn.predict(test_data, test_label, is_learning=False)
        return loss, accuracy

    def save_model(self, fpath):
        with open(fpath, "wb") as f:
            pickle.dump(self.nn, f)

    def load_model(self, fpath) -> DNN:
        with open(fpath, "rb") as f:
            model = pickle.load(f)
        return model

    # 誤差逆伝播法と数値微分の差異があるかどうかを見るためのデバッグ用メソッド。
    # check_gradientの実行結果は以下。差異がないことが判明した。
    # layer[0]: diff=4.4969488372317266e-10
    # layer[1]: diff=5.936436439956446e-09
    def check_gradient(self, train_data, train_label):
        # self.nn = DNN(input_size=784, layer_size_list=[10, 5])
        self.nn = DNN(input_size=784,
                      layer_size_list=[10, 5],
                      hidden_actfunc=ReLU(),
                      output_actfunc=SoftmaxWithLoss(),
                      loss_func=CrossEntropyError(),
                      init_weight_stddev=0.01,
                      learner=MiniBatch(epoch_num=10, mini_batch_size=100, optimizer=SGD(learning_rate=0.01), is_numerical_gradient=True))

        # 数値微分は計算量が多く処理時間がかかる（しばらく返ってこない）ため、
        # 使用するデータ個数とエポック数を絞る。
        x = train_data[:3]
        t = train_label[:3]

        # fitメソッドを動かすことにより、全レイヤーの（中のAffineレイヤー）のgradientとnumerical_gradientを実行してメモリに保持。
        # ただし、numerical_gradient=Trueを指定しているので注意。
        self.nn.fit(train_data=x, train_label=t)
        #self.nn.fit(train_data=x, train_label=t, init_weight_stddev=0.01, epoch_num=3, mini_batch_size=100, learning_rate=0.01, numerical_gradient=True)

        for i, layer in enumerate(self.nn.layers):
            bg = layer.affine.dLdW
            # print(bg)
            ng = layer.affine.numerical_dLdW
            # print(ng)

            # Wの各成分ごとの差異の絶対値を、全成分に渡って平均。
            diff = np.average(np.abs(bg - ng))
            print("layer[{0}]: diff={1}".format(i, diff))

    def create_model(self):
        model = None

        ##################################################
        # 以下リファクタリング後：各活性化関数、損失関数、最適化を切り替えることができるようにした。
        ##################################################
        # （変遷１）入力データは単純正規化（255で割るだけ）／2層／エポック数3／ミニバッチサイズ100／Sigmoid／SGD
        # ⇒Accuracy20%程度。
        # ★Avg.loss=1.608, Avg.accuracy=0.248, Max.accuracy=0.345, Argmax=2 | Avg.test_loss=1.609, Avg.test_accuracy=0.245, Max.test_accuracy=0.335, Argmax=2
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 5],
        #               hidden_actfunc=Sigmoid(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=3, mini_batch_size=100, optimizer=SGD(learning_rate=0.01)))

        # （変遷２）エポック数を変えてみた。エポック数100
        # ⇒★Avg.loss=1.467, Avg.accuracy=0.733, Max.accuracy=0.922, Argmax=93 | Avg.test_loss=1.477, Avg.test_accuracy=0.715, Max.test_accuracy=0.935, Argmax=96
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 5],
        #               hidden_actfunc=Sigmoid(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=100, optimizer=SGD(learning_rate=0.01)))

        # （変遷３－１）ミニバッチサイズを変えてみた。エポック数は100でミニバッチサイズ50の場合
        # ⇒★Avg.loss=1.062, Avg.accuracy=0.818, Max.accuracy=0.959, Argmax=88 | Avg.test_loss=1.082, Avg.test_accuracy=0.810, Max.test_accuracy=0.960, Argmax=82
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 5],
        #               hidden_actfunc=Sigmoid(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=50, optimizer=SGD(learning_rate=0.01)))

        # （変遷３－２）ミニバッチサイズを変えてみた。エポック数は100でミニバッチサイズ10の場合
        # ⇒ミニバッチサイズ50のときよりもさらに学習が進み、正解率も高くなった。
        # ★Avg.loss=0.289, Avg.accuracy=0.936, Max.accuracy=0.995, Argmax=98 | Avg.test_loss=0.309, Avg.test_accuracy=0.931, Max.test_accuracy=0.985, Argmax=30
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 5],
        #               hidden_actfunc=Sigmoid(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=SGD(learning_rate=0.01)))

        # ★★★以降、エポック数100、ミニバッチサイズ10で行く。

        # （変遷４）層数を変えてみた。3層
        # ⇒学習が全く進まない。
        # ★Avg.loss=1.612, Avg.accuracy=0.201, Max.accuracy=0.210, Argmax=74 | Avg.test_loss=1.614, Avg.test_accuracy=0.195, Max.test_accuracy=0.220, Argmax=1
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Sigmoid(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=SGD(learning_rate=0.01)))

        # （変遷４－１）3層で、tanhに変えてみた。
        # ⇒ 学習が進み、最大Accuracyが96%になった。
        # ★Avg.loss=0.222, Avg.accuracy=0.944, Max.accuracy=1.000, Argmax=66 | Avg.test_loss=0.336, Avg.test_accuracy=0.906, Max.test_accuracy=0.960, Argmax=28
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Tanh(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=SGD(learning_rate=0.01)))

        # （変遷４－２）3層で、ReLUに変えてみた。
        # ⇒学習が進んだ。
        # ★Avg.loss=0.308, Avg.accuracy=0.876, Max.accuracy=1.000, Argmax=63 | Avg.test_loss=0.461, Avg.test_accuracy=0.838, Max.test_accuracy=0.965, Argmax=73
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=SGD(learning_rate=0.01)))

        # （変遷４－３）試しにReLUで4層にしてみた。
        # ⇒学習が全く進まない。
        # ★Avg.loss=1.609, Avg.accuracy=0.211, Max.accuracy=0.211, Argmax=0 | Avg.test_loss=1.614, Avg.test_accuracy=0.155, Max.test_accuracy=0.155, Argmax=0
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=SGD(learning_rate=0.01)))

        # （変遷４－４）ReLUで5層にしてみた。
        # ⇒学習が全く進まない。
        # ★Avg.loss=1.609, Avg.accuracy=0.209, Max.accuracy=0.209, Argmax=0 | Avg.test_loss=1.613, Avg.test_accuracy=0.165, Max.test_accuracy=0.165, Argmax=0
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=SGD(learning_rate=0.01)))

        # （変遷４－５）試しにReLUで10層にしてみた。
        # ⇒学習が全く進まない。
        # ★Avg.loss=1.609, Avg.accuracy=0.209, Max.accuracy=0.209, Argmax=0 | Avg.test_loss=1.612, Avg.test_accuracy=0.165, Max.test_accuracy=0.165, Argmax=0
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 100, 100, 100, 100, 100, 100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=SGD(learning_rate=0.01)))

        # （変遷５－１）ノード数の増減をするととどうなるか？
        # ★ノード数を10、100、1000にしてみた。
        # 　ノード数10：★Avg.loss=0.487, Avg.accuracy=0.876, Max.accuracy=0.996, Argmax=93 | Avg.test_loss=0.517, Avg.test_accuracy=0.864, Max.test_accuracy=0.980, Argmax=34
        # 　ノード数100：★Avg.loss=0.285, Avg.accuracy=0.937, Max.accuracy=0.995, Argmax=95 | Avg.test_loss=0.311, Avg.test_accuracy=0.930, Max.test_accuracy=0.980, Argmax=34
        # 　ノード数1000：★Avg.loss=0.192, Avg.accuracy=0.954, Max.accuracy=0.995, Argmax=95 | Avg.test_loss=0.189, Avg.test_accuracy=0.957, Max.test_accuracy=0.990, Argmax=27
        #  ⇒ノード数を大きくすると、性能は良くなっている（特に平均損失が低くなる）。ただし、1000の場合は処理時間が100の10倍以上かかっているため、計算リソースとの兼ね合いを考慮する必要がある。
        # （参考）処理時間の計測：
        # 　ノード数が10の場合：elapsed_time: 2.081 [sec]　elapsed_time: 2.113 [sec]　elapsed_time: 2.108 [sec]　⇒平均 2.1 [sec]
        # 　ノード数が100の場合：elapsed_time: 6.412 [sec]　elapsed_time: 6.511 [sec]　elapsed_time: 6.502 [sec]　⇒平均 6.5 [sec]
        # 　ノード数が1000の場合：elapsed_time: 79.701 [sec] elapsed_time: 79.792 [sec]　elapsed_time: 76.923 [sec]　⇒平均 79.8 [sec]
        # model = DNN(input_size=784,
        #               #layer_size_list=[10, 5],
        #               #layer_size_list=[100, 5],
        #               layer_size_list=[1000, 5],
        #               hidden_actfunc=Sigmoid(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=SGD(learning_rate=0.01)))

        # （変遷５－２）ノード数100
        # （変遷３－２）を元にして、ノード数1に減らした。
        # 参考）（変遷３－２）の結果：★Avg.loss=0.289, Avg.accuracy=0.936, Max.accuracy=0.995, Argmax=98 | Avg.test_loss=0.309, Avg.test_accuracy=0.931, Max.test_accuracy=0.985, Argmax=30
        # 結果）★Avg.loss = 1.261, Avg.accuracy = 0.392, Max.accuracy = 0.479, Argmax = 20 | Avg.test_loss = 1.280, Avg.test_accuracy = 0.361, Max.test_accuracy = 0.480, Argmax = 19
        # ⇒性能が悪くなった。
        # model = DNN(input_size=784,
        #               layer_size_list=[10, 5],
        #               hidden_actfunc=Sigmoid(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=SGD(learning_rate=0.01)))

        ##############################
        # 以下、Day5の内容について調査・実験
        ##############################
        ##############################
        # 重みの初期値変更
        ##############################
        # 元の問題点：（変遷４－４）で学習が全く進まない。という問題点があった。
        # （変遷６－１）Sigmoidの場合にXavierの初期値を使用するとどうなるか？
        # ・標準偏差0.01固定の場合：★Avg.loss=1.612, Avg.accuracy=0.204, Max.accuracy=0.209, Argmax=0 | Avg.test_loss=1.617, Avg.test_accuracy=0.185, Max.test_accuracy=0.235, Argmax=24
        # ↓
        # ・Xavierの初期値の場合：★Avg.loss=1.582, Avg.accuracy=0.296, Max.accuracy=0.726, Argmax=81 | Avg.test_loss=1.589, Avg.test_accuracy=0.271, Max.test_accuracy=0.690, Argmax=81
        # ⇒多少不安定だが（エポックごとに性能がぶれる）が、性能は良くなった。確かに効果があることが分かった。
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 100, 100, 5],
        #               hidden_actfunc=Sigmoid(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=SGD(learning_rate=0.01)),
        #               init_weight_change=True   # 今回新たに実装。
        #               )

        # （変遷６－２）Tanhの場合にXavierの初期値を使用するとどうなるか？
        # ・標準偏差0.01固定の場合：★Avg.loss=1.609, Avg.accuracy=0.209, Max.accuracy=0.209, Argmax=0 | Avg.test_loss=1.613, Avg.test_accuracy=0.165, Max.test_accuracy=0.165, Argmax=0
        # ↓
        # ・Xavierの初期値の場合：★Avg.loss=0.014, Avg.accuracy=0.999, Max.accuracy=1.000, Argmax=18 | Avg.test_loss=0.073, Avg.test_accuracy=0.984, Max.test_accuracy=0.985, Argmax=2
        # ⇒大幅に改善した。
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 100, 100, 5],
        #               hidden_actfunc=Tanh(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=SGD(learning_rate=0.01)),
        #               init_weight_change=True   # 今回新たに実装。
        #               )

        # （変遷６－３）ReLUの場合に『Heの初期値』を使用。
        # ・標準偏差0.01固定の場合：★Avg.loss=1.609, Avg.accuracy=0.209, Max.accuracy=0.209, Argmax=0 | Avg.test_loss=1.613, Avg.test_accuracy=0.165, Max.test_accuracy=0.165, Argmax=0
        # ↓
        # ・Heの初期値の場合：★Avg.loss=0.021, Avg.accuracy=0.996, Max.accuracy=1.000, Argmax=12 | Avg.test_loss=0.069, Avg.test_accuracy=0.982, Max.test_accuracy=0.985, Argmax=9
        # ⇒大幅に改善した。
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=SGD(learning_rate=0.01)),
        #               init_weight_change=True   # 今回新たに実装。
        #               )

        # ##################################################
        # 実験25（Day4までの講義内容を実装した中で最も性能が良かったモデル）について、
        # 重みの初期値を変えてみたらどうなるか？
        # ##################################################
        # ●k分割交差検証で最も良かったモデル：Kfold-Tanh-AdaDelta
        # 3層
        #   初期値変更なし：★kfold_num=100: Avg.Loss=0.001, Avg.Accuracy=1.000, Max.Accuracy=1.000, Argmax=0
        #   初期値変更あり：★kfold_num=100: Avg.Loss=0.001, Avg.Accuracy=1.000, Max.Accuracy=1.000, Argmax=0
        #   ↑あり・なしで結果は変わらない。
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Tanh(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               init_weight_stddev=0.01,
        #               learner=KFoldCrossValidation(kfold_num=100, optimizer=AdaDelta(decay_rate=0.9)),
        #               init_weight_change=True
        #               )

        # 5層に増やした／init_weight_change=Trueを指定。
        # （元の実験）★kfold_num = 100: Avg.Loss = 0.001, Avg.Accuracy = 1.000, Max.Accuracy = 1.000, Argmax = 0
        # （初期値変更版）★kfold_num=100: Avg.Loss=0.000, Avg.Accuracy=1.000, Max.Accuracy=1.000, Argmax=0
        # --------------------------------------------------
        # 以下テストデータでの結果：
        #   講座名:dl_tokyo_2
        #   ファイル名:dl_tokyo_2_submit_katakana_SHIMAMOTO_TATSUYA_20180726.zip
        #   Test loss:0.1751423330705909
        #   Test accuracy:0.9735384615384616
        # --------------------------------------------------
        # model = DNN(input_size=784,
        #             layer_size_list=[100, 100, 100, 100, 5],
        #             hidden_actfunc=Tanh(),
        #             output_actfunc=SoftmaxWithLoss(),
        #             loss_func=CrossEntropyError(),
        #             init_weight_stddev=0.01,
        #             init_weight_change=True,
        #             learner=KFoldCrossValidation(kfold_num=100, optimizer=AdaDelta(decay_rate=0.9))
        #             )

        # ●Day4までの講義内容で、ミニバッチ学習のうち最良のモデル：Minibatch-ReLU-AdaGrad
        # 初期値変更あり・なしをやってみた。5層に増やしたので注意。
        # （元の結果）★Avg.loss=0.001, Avg.accuracy=1.000, Max.accuracy=1.000, Argmax=4 | Avg.test_loss=0.120, Avg.test_accuracy=0.965, Max.test_accuracy=0.970, Argmax=0
        # （初期値変更なし）★Avg.loss=0.091, Avg.accuracy=0.957, Max.accuracy=1.000, Argmax=59 | Avg.test_loss=0.223, Avg.test_accuracy=0.934, Max.test_accuracy=0.985, Argmax=59
        # （初期値変更あり）★Avg.loss=0.002, Avg.accuracy=1.000, Max.accuracy=1.000, Argmax=5 | Avg.test_loss=0.126, Avg.test_accuracy=0.982, Max.test_accuracy=0.985, Argmax=2
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=AdaGrad(learning_rate=0.01)),
        #               init_weight_change=True
        #               )

        # ------------------------------
        # 重みの初期値変更に関して、アクティベーション分布を見てみた。
        # 5層／ReLU／初期値変更なし、ありを2通り実施（比較のため）／層間での違いだけを見たいのでエポック数1とする。
        # ※『./act_dist.pkl』ファイルに出力される。DNNクラスのpredictメソッド参照。
        # ★Avg.loss=1.609, Avg.accuracy=0.209, Max.accuracy=0.209, Argmax=0 | Avg.test_loss=1.613, Avg.test_accuracy=0.165, Max.test_accuracy=0.165, Argmax=0
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=1, mini_batch_size=10, optimizer=SGD(learning_rate=0.01)),
        #               init_weight_change=True
        #               )

        ##############################
        # （変遷７）以下、バッチ正規化（Batch Normalization）を試す。
        ##############################
        # 元の問題点：『（変遷４－４）ReLUで5層にしてみた。』にて、学習が全く進まない。
        # 元の実行結果：★Avg.loss=1.609, Avg.accuracy=0.209, Max.accuracy=0.209, Argmax=0 | Avg.test_loss=1.613, Avg.test_accuracy=0.165, Max.test_accuracy=0.165, Argmax=0
        # ↓
        # BNでの実行結果：★Avg.loss=0.168, Avg.accuracy=0.958, Max.accuracy=0.990, Argmax=7 | Avg.test_loss=0.189, Avg.test_accuracy=0.949, Max.test_accuracy=0.975, Argmax=11
        # ⇒重みの初期値の変更をしなくても性能が改善した。
        # ⇒ただし、『（変遷６－３）ReLUの場合に『Heの初期値』を使用。』と検証データでの性能同士を比較した場合は、主みの初期値の変更の方がよい。
        # ⇒過学習が抑制されたからか？
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=SGD(learning_rate=0.01)),
        #               batch_normal=BatchNormal(gamma=1.0, beta=0.0)
        #               )

        ##############################
        # 以下正則化を試す。（regularization）
        ##############################
        # ■k分割交差検証での最良のモデル（明らかに過学習）：Kfold-Tanh-AdaDelta／3層
        #   正則化なし：★kfold_num=100: Avg.Loss=0.001, Avg.Accuracy=1.000, Max.Accuracy=1.000, Argmax=0
        #   lmda=0.01:★kfold_num=100: Avg.Loss=0.1568, Avg.Accuracy=0.9880, Max.Accuracy=1.0000, Argmax=0
        #       ↑提出したもので97.4%を記録したパターンのテストデータでのロスに最も近いので、これで提出してみる。
        #       提出結果は以下。
        #             講座名:dl_tokyo_2
        #             ファイル名:dl_tokyo_2_submit_katakana_SHIMAMOTO_TATSUYA_20180731.zip
        #             Test loss:0.22774905019689023
        #             Test accuracy:0.9673846153846154
        #   lmda=0.02:★kfold_num=100: Avg.Loss=0.273, Avg.Accuracy=0.982, Max.Accuracy=1.000, Argmax=0
        #   lmda=0.05:★kfold_num=100: Avg.Loss=0.495, Avg.Accuracy=0.972, Max.Accuracy=1.000, Argmax=2
        #   lmda=0.1:★kfold_num=100: Avg.Loss=0.783, Avg.Accuracy=0.958, Max.Accuracy=1.000, Argmax=2
        #        0.1で試しで提出してみた。さすがにダメだった。損失が大きすぎる。
        #             講座名:dl_tokyo_2
        #             ファイル名:dl_tokyo_2_submit_katakana_SHIMAMOTO_TATSUYA_20180731.zip
        #             Test loss:0.8138704762036064
        #             Test accuracy:0.9506153846153846
        #   lmda=0.2:★kfold_num=100: Avg.Loss=1.611, Avg.Accuracy=0.162, Max.Accuracy=0.500, Argmax=14
        #   lmda=0.5:★kfold_num=100: Avg.Loss=1.611, Avg.Accuracy=0.162, Max.Accuracy=0.500, Argmax=14
        #   lmda=1.0:★kfold_num=100: Avg.Loss=1.611, Avg.Accuracy=0.162, Max.Accuracy=0.500, Argmax=14
        # model = DNN(input_size=784,
        #             layer_size_list=[100, 100, 5],
        #             hidden_actfunc=Tanh(),
        #             output_actfunc=SoftmaxWithLoss(),
        #             loss_func=CrossEntropyError(),
        #             init_weight_stddev=0.01,
        #             learner=KFoldCrossValidation(kfold_num=100, optimizer=AdaDelta(decay_rate=0.9)),
        #             regularization=L2(lmda=0.01)
        #             )

        # ■ミニバッチで最良のモデル：Minibatch-ReLU-AdaGrad／3層
        #   正則化なし：★Avg.loss=0.001, Avg.accuracy=1.000, Max.accuracy=1.000, Argmax=4 | Avg.test_loss=0.120, Avg.test_accuracy=0.965, Max.test_accuracy=0.970, Argmax=0
        #   lmda=0.01:★Avg.loss=0.131, Avg.accuracy=0.997, Max.accuracy=1.000, Argmax=32 | Avg.test_loss=0.220, Avg.test_accuracy=0.967, Max.test_accuracy=0.975, Argmax=16
        #   lmda=0.05:★Avg.loss=0.486, Avg.accuracy=0.973, Max.accuracy=0.990, Argmax=34 | Avg.test_loss=0.581, Avg.test_accuracy=0.936, Max.test_accuracy=0.960, Argmax=10
        #   lmda=0.1:★Avg.loss=1.609, Avg.accuracy=0.205, Max.accuracy=0.205, Argmax=8 | Avg.test_loss=1.611, Avg.test_accuracy=0.180, Max.test_accuracy=0.185, Argmax=0
        #   lmda=1.0:★Avg.loss=1.609, Avg.accuracy=0.205, Max.accuracy=0.205, Argmax=10 | Avg.test_loss=1.611, Avg.test_accuracy=0.180, Max.test_accuracy=0.185, Argmax=0
        #   lmda=10.0:★Avg.loss=1.609, Avg.accuracy=0.205, Max.accuracy=0.205, Argmax=10 | Avg.test_loss=1.611, Avg.test_accuracy=0.180, Max.test_accuracy=0.185, Argmax=0
        # ⇒訓練データと検証データとの正解率の推移を見てみたが、L2の効果は分かりづらい。おそらく、最適化まで実施しているためそもそも最良に近いモデルだからと思われる。
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=AdaGrad(learning_rate=0.01)),
        #               regularization=L2(lmda=0.1)
        #               )

        ##############################
        # 以下ドロップアウトを試す。
        ##############################
        # 過学習していると思われる、最良モデル『Kfold-Tanh-AdaDelta』についてドロップアウトを試してみた。
        #   ドロップアウトなし：★kfold_num = 100: Avg.Loss = 0.001, Avg.Accuracy = 1.000, Max.Accuracy = 1.000, Argmax = 0
        #   input_retain_rate=0.8, hidden_retain_rate=0.1の場合：★kfold_num=100: Avg.Loss=0.800, Avg.Accuracy=0.623, Max.Accuracy=1.000, Argmax=88
        #   input_retain_rate=0.8, hidden_retain_rate=0.2の場合：★kfold_num=100: Avg.Loss=0.189, Avg.Accuracy=0.941, Max.Accuracy=1.000, Argmax=13
        #   input_retain_rate=0.8, hidden_retain_rate=0.3の場合：★kfold_num=100: Avg.Loss=0.080, Avg.Accuracy=0.980, Max.Accuracy=1.000, Argmax=3
        #   ↑ちょうどいい感じか？これで提出してみる。
        #   input_retain_rate=0.8, hidden_retain_rate=0.4の場合：★kfold_num=100: Avg.Loss=0.034, Avg.Accuracy=0.989, Max.Accuracy=1.000, Argmax=2
        #   input_retain_rate=0.8, hidden_retain_rate=0.5の場合：★kfold_num=100: Avg.Loss=0.020, Avg.Accuracy=0.996, Max.Accuracy=1.000, Argmax=2
        #       （最新ログ出力での結果）★Avg.l_loss=0.0138, Avg.l_accuracy=0.9969, Max.l_accuracy=1.0000, l_argmax=34 | Avg.v_loss=0.0199, Avg.v_accuracy=0.9960, Max.v_accuracy=1.0000, v_argmax=2
        #   ↑一応推奨値なので提出してみる。
        #   input_retain_rate=0.8, hidden_retain_rate=0.6の場合：★kfold_num=100: Avg.Loss=0.015, Avg.Accuracy=0.996, Max.Accuracy=1.000, Argmax=2
        #   input_retain_rate=0.8, hidden_retain_rate=0.7の場合：★kfold_num=100: Avg.Loss=0.010, Avg.Accuracy=0.998, Max.Accuracy=1.000, Argmax=0
        #   input_retain_rate=0.8, hidden_retain_rate=0.8の場合：★kfold_num=100: Avg.Loss=0.006, Avg.Accuracy=0.998, Max.Accuracy=1.000, Argmax=0
        #   input_retain_rate=0.8, hidden_retain_rate=0.9の場合：★kfold_num=100: Avg.Loss=0.003, Avg.Accuracy=0.998, Max.Accuracy=1.000, Argmax=0
        #   ↑損失が最も低いが・・・過学習か？提出してみるか？
        #
        #   以下、隠れ層を固定して入力層を変化。
        #   input_retain_rate=0.1, hidden_retain_rate=0.5の場合：★kfold_num=100: Avg.Loss=0.151, Avg.Accuracy=0.942, Max.Accuracy=1.000, Argmax=13
        #   input_retain_rate=0.2, hidden_retain_rate=0.5の場合：★kfold_num=100: Avg.Loss=0.091, Avg.Accuracy=0.968, Max.Accuracy=1.000, Argmax=5
        #   ↑これもちょうどいい感じか？
        #   input_retain_rate=0.3, hidden_retain_rate=0.5の場合：★kfold_num=100: Avg.Loss=0.066, Avg.Accuracy=0.984, Max.Accuracy=1.000, Argmax=2
        #   input_retain_rate=0.4, hidden_retain_rate=0.5の場合：★kfold_num=100: Avg.Loss=0.049, Avg.Accuracy=0.985, Max.Accuracy=1.000, Argmax=2
        #   input_retain_rate=0.5, hidden_retain_rate=0.5の場合：★kfold_num=100: Avg.Loss=0.038, Avg.Accuracy=0.989, Max.Accuracy=1.000, Argmax=2
        #   input_retain_rate=0.6, hidden_retain_rate=0.5の場合：★kfold_num=100: Avg.Loss=0.027, Avg.Accuracy=0.994, Max.Accuracy=1.000, Argmax=2
        #   input_retain_rate=0.7, hidden_retain_rate=0.5の場合：★kfold_num=100: Avg.Loss=0.022, Avg.Accuracy=0.995, Max.Accuracy=1.000, Argmax=2
        #   input_retain_rate=0.8, hidden_retain_rate=0.5の場合：★kfold_num=100: Avg.Loss=0.020, Avg.Accuracy=0.996, Max.Accuracy=1.000, Argmax=2
        #   input_retain_rate=0.9, hidden_retain_rate=0.5の場合：★kfold_num=100: Avg.Loss=0.019, Avg.Accuracy=0.995, Max.Accuracy=1.000, Argmax=2
        # model = DNN(input_size=784,
        #             layer_size_list=[100, 100, 5],
        #             hidden_actfunc=Tanh(),
        #             output_actfunc=SoftmaxWithLoss(),
        #             loss_func=CrossEntropyError(),
        #             init_weight_stddev=0.01,
        #             learner=KFoldCrossValidation(kfold_num=100, optimizer=AdaDelta(decay_rate=0.9)),
        #             dropout_params=DropoutParams(input_retain_rate=0.8, hidden_retain_rate=0.5)
        #             )
        # ------------------------------
        # 推奨パラメータでやってみる。
        # 原論文：N.Srivastava, G.Hinton, A.Krizhevsky, I.Sutskever, R.Salakhutdinov.
        #       Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
        #       Journal of Machine Learning Research 15 (2014) 1929-1958
        # ・隠れ層のノード数：nからn/pに増やす。pは隠れ層の保持率0.5。⇒現在100個÷0.5=200個に変更した。
        # ・活性化関数：ReLUを使う。⇒やはりTanhの方がLossが低かったのでそのままTanhを使った。
        # ・学習率ηまたはモーメンタムα：通常のNNで最適なηの10倍～100倍にする。または、モーメント係数（減衰率）を0.95～0.99にする。⇒0.95にした。
        # ・正則化：L∞ノルムを使う（Cの値は3～4。つまりλ＝0.25～0.33）。⇒L∞ノルムでは学習が進まなかったのでL2のままにした（λ＝0.01）。
        # ・保持率：入力層では0.8、隠れ層では0.5とした。
        # ------------------------------
        # 結果：エポック13回で早期終了した。
        # ★epoch[0]終了 loss=1.411, accuracy=0.900
        # ★epoch[1]終了 loss=0.479, accuracy=1.000
        # ★epoch[2]終了 loss=0.631, accuracy=0.900
        # ★epoch[3]終了 loss=0.474, accuracy=0.900
        # ★epoch[4]終了 loss=0.289, accuracy=1.000
        # ★epoch[5]終了 loss=0.284, accuracy=1.000
        # ★epoch[6]終了 loss=0.289, accuracy=1.000
        # ★epoch[7]終了 loss=0.540, accuracy=0.900
        # ★epoch[8]終了 loss=0.281, accuracy=1.000
        # ★epoch[9]終了 loss=0.298, accuracy=1.000
        # ★epoch[10]終了 loss=0.299, accuracy=1.000
        # ★epoch[11]終了 loss=0.375, accuracy=1.000
        # ★epoch[12]終了 loss=0.407, accuracy=0.900
        # ★kfold_num=100: Avg.Loss=0.466, Avg.Accuracy=0.962, Max.Accuracy=1.000, Argmax=1
        #
        # elapsed_time: 16.949 [sec]
        # ------------------------------
        # model = DNN(input_size=784,
        #               layer_size_list=[200, 200, 5],  # 隠れ層のノード数を隠れ層の保持率0.5で割って200に変更。⇒ただし計算時間大。
        #               hidden_actfunc=Tanh(),  # Tanhの方が良いのでそのままにした。⇒以下で重みの初期値変更も入れた。
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               init_weight_stddev=0.01,
        #               learner=KFoldCrossValidation(kfold_num=100, optimizer=AdaDelta(decay_rate=0.95)),  # モーメンタム係数（減衰率）を0.95に変更。
        #               init_weight_change=True,  # 重みの初期値変更も入れてみた。
        #               regularization=L2(lmda=0.01),  # L∞ノルム（λは0.33(c=3に相当)）に変更。⇒L∞ノルムでやってみたが# が学習が進まないので止めた。
        #               dropout_params=DropoutParams(input_retain_rate=0.8, hidden_retain_rate=0.5),
        #               early_stopping_params=EarlyStoppingParams(3)  # 損失が1を超過してしまっていたので早期終了ロジックを入れた。3回連続悪化まで許す。
        #               )

        ##############################
        # 以下ミニバッチサイズ10
        ##############################
        # 2層、シグモイド、ミニバッチサイズ10　⇒学習が進んだ。
        # ★Avg.Loss=0.285, Avg.Accuracy=0.937, Max.Accuracy=0.995, Argmax=95
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 5],
        #               hidden_actfunc=Sigmoid(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, learning_rate=0.01))

        # 2層、tanh、ミニバッチサイズ10　⇒シグモイドと同じ程度。
        # ★Avg.Loss=0.063, Avg.Accuracy=0.994, Max.Accuracy=1.000, Argmax=35
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 5],
        #               hidden_actfunc=Tanh(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, learning_rate=0.01))

        # 2層、ReLU、ミニバッチサイズ＝10　⇒正解率がシグモイドより高いが、tanhよりも低い。tanhｙりも学習が進んでいない。
        # ★Avg.Loss=0.069, Avg.Accuracy=0.989, Max.Accuracy=1.000, Argmax=43
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, learning_rate=0.01))

        # ★★★以上より、ミニバッチサイズは100よりも10の方がよい。⇒以下ミニバッチサイズ10固定とした。
        ##############################
        # 以下ミニバッチサイズ10で3層
        ##############################
        # 3層、シグモイド、ミニバッチサイズ10　⇒ ほぼ20%のまま学習が進まない。
        # ★Avg.Loss = 1.612, Avg.Accuracy = 0.201, Max.Accuracy = 0.210, Argmax = 74
        # ※白黒反転時：★Avg.loss=1.612, Avg.accuracy=0.201, Max.accuracy=0.205, Argmax=4 | Avg.test_loss=1.614, Avg.test_accuracy=0.195, Max.test_accuracy=0.220, Argmax=1
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Sigmoid(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=SGD(learning_rate=0.01)))

        # 3層、tanh、ミニバッチサイズ10　⇒シグモイドに比べて良くなった。おそらくシグモイドでは勾配消失が起こっていると思われる。
        # ★Avg.loss=0.222, Avg.accuracy=0.944, Max.accuracy=1.000, Argmax=66 | Avg.test_loss=0.336, Avg.test_accuracy=0.906, Max.test_accuracy=0.960, Argmax=28
        # ※白黒反転時：★Avg.loss=0.212, Avg.accuracy=0.930, Max.accuracy=1.000, Argmax=75 | Avg.test_loss=0.347, Avg.test_accuracy=0.892, Max.test_accuracy=0.960, Argmax=81
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Tanh(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=SGD(learning_rate=0.01)))

        # 3層、ReLU、ミニバッチサイズ＝10　⇒tanhと比べて早く最大正解率100%に到達した。
        # ★Avg.loss=0.308, Avg.accuracy=0.876, Max.accuracy=1.000, Argmax=63 | Avg.test_loss=0.461, Avg.test_accuracy=0.838, Max.test_accuracy=0.965, Argmax=73
        # ※白黒反転時：★Avg.loss=0.368, Avg.accuracy=0.862, Max.accuracy=1.000, Argmax=71 | Avg.test_loss=0.462, Avg.test_accuracy=0.828, Max.test_accuracy=0.960, Argmax=57
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=SGD(learning_rate=0.01)))

        ##############################
        # リファクタリング後：重みの更新を最適化クラスで実行できるようにした。
        # 以下、最適化の切り替え実験
        ##############################
        ##############################
        # まず、3層、シグモイド、ミニバッチサイズ10 を固定し、各種最適化を試した。
        ##############################
        # Momentum
        # 　⇒ SGDのときより大幅に学習が進んだ。⇒最適化を変えた効果あり。
        # ★Avg.loss=0.704, Avg.accuracy=0.708, Max.accuracy=0.983, Argmax=98 | Avg.test_loss=0.807, Avg.test_accuracy=0.671, Max.test_accuracy=0.935, Argmax=88
        # ※白黒反転時：★Avg.loss=0.397, Avg.accuracy=0.836, Max.accuracy=0.999, Argmax=62 | Avg.test_loss=0.541, Avg.test_accuracy=0.801, Max.test_accuracy=0.960, Argmax=79
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Sigmoid(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=Momentum(learning_rate=0.01, decay_rate=0.9)))

        # AdaGrad
        # 　⇒Momentumより正解率が低く、学習も遅い。
        # ★Avg.loss=0.958, Avg.accuracy=0.562, Max.accuracy=0.640, Argmax=97 | Avg.test_loss=1.016, Avg.test_accuracy=0.561, Max.test_accuracy=0.660, Argmax=86
        # ※白黒反転時：★Avg.loss=0.728, Avg.accuracy=0.680, Max.accuracy=0.751, Argmax=97 | Avg.test_loss=0.781, Avg.test_accuracy=0.656, Max.test_accuracy=0.715, Argmax=84
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Sigmoid(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=AdaGrad(learning_rate=0.01)))

        # AdaDelta
        # 　⇒AdaGradより正解率が高い。学習も早い。ただし、Momentumよりも多少遅れて最大正解率に到達している。
        # ★Avg.loss=0.526, Avg.accuracy=0.781, Max.accuracy=0.969, Argmax=97 | Avg.test_loss=0.796, Avg.test_accuracy=0.738, Max.test_accuracy=0.900, Argmax=97
        # ※白黒反転時：★Avg.loss=0.120, Avg.accuracy=0.949, Max.accuracy=1.000, Argmax=70 | Avg.test_loss=0.327, Avg.test_accuracy=0.910, Max.test_accuracy=0.960, Argmax=34
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Sigmoid(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=AdaDelta(decay_rate=0.9)))

        # RMSProp
        # 　⇒20％程度で、学習が進まない。
        # ★Avg.loss=12.897, Avg.accuracy=0.200, Max.accuracy=0.205, Argmax=3 | Avg.test_loss=12.883, Avg.test_accuracy=0.201, Max.test_accuracy=0.220, Argmax=0
        # ※白黒反転時：★Avg.loss=12.897, Avg.accuracy=0.200, Max.accuracy=0.205, Argmax=3 | Avg.test_loss=12.883, Avg.test_accuracy=0.201, Max.test_accuracy=0.220, Argmax=0
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Sigmoid(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=RMSProp(learning_rate=0.01)))

        # 同上、Adam　⇒最大でも86.8％。
        # ★Avg.loss=0.100, Avg.accuracy=0.962, Max.accuracy=1.000, Argmax=57 | Avg.test_loss=0.317, Avg.test_accuracy=0.918, Max.test_accuracy=0.980, Argmax=92
        # ※白黒反転時：★Avg.loss=0.374, Avg.accuracy=0.868, Max.accuracy=0.922, Argmax=73 | Avg.test_loss=0.561, Avg.test_accuracy=0.820, Max.test_accuracy=0.865, Argmax=48
        # TODO 途中で1回Overflowを起こした。原因不明。
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Sigmoid(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=Adam(learning_rate=0.01, decay_rate1=0.9, decay_rate2=0.9999)))

        # 同上、NAG
        # ★Avg.loss=0.661, Avg.accuracy=0.733, Max.accuracy=0.978, Argmax=75 | Avg.test_loss=0.770, Avg.test_accuracy=0.696, Max.test_accuracy=0.935, Argmax=87
        # ※白黒反転時：★Avg.loss=0.378, Avg.accuracy=0.847, Max.accuracy=0.999, Argmax=69 | Avg.test_loss=0.559, Avg.test_accuracy=0.812, Max.test_accuracy=0.965, Argmax=66
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Sigmoid(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=NAG(learning_rate=0.01, decay_rate=0.9)))

        ##############################
        # 次に3層、tanh、ミニバッチサイズ10 を固定し、各種最適化を試した。
        ##############################
        # Momentum
        # ★Avg.loss=0.030, Avg.accuracy=0.993, Max.accuracy=1.000, Argmax=13 | Avg.test_loss=0.164, Avg.test_accuracy=0.962, Max.test_accuracy=0.975, Argmax=26
        # ※白黒反転時：★Avg.loss=0.022, Avg.accuracy=0.996, Max.accuracy=1.000, Argmax=8 | Avg.test_loss=0.208, Avg.test_accuracy=0.969, Max.test_accuracy=0.980, Argmax=34
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Tanh(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=Momentum(learning_rate=0.01, decay_rate=0.9)))

        # AdaGrad
        # ★Avg.loss=0.017, Avg.accuracy=0.998, Max.accuracy=1.000, Argmax=17 | Avg.test_loss=0.154, Avg.test_accuracy=0.959, Max.test_accuracy=0.975, Argmax=60
        # ※白黒反転時：★Avg.loss=0.002, Avg.accuracy=1.000, Max.accuracy=1.000, Argmax=4 | Avg.test_loss=0.167, Avg.test_accuracy=0.960, Max.test_accuracy=0.960, Argmax=0
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Tanh(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=AdaGrad(learning_rate=0.01)))

        # AdaDelta
        # ★Avg.loss=0.014, Avg.accuracy=0.995, Max.accuracy=1.000, Argmax=14 | Avg.test_loss=0.208, Avg.test_accuracy=0.958, Max.test_accuracy=0.975, Argmax=87
        # ※白黒反転時：★Avg.loss=0.001, Avg.accuracy=1.000, Max.accuracy=1.000, Argmax=5 | Avg.test_loss=0.233, Avg.test_accuracy=0.970, Max.test_accuracy=0.970, Argmax=6
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Tanh(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=AdaDelta(decay_rate=0.9)))

        # RMSProp
        # ★Avg.loss=12.896, Avg.accuracy=0.200, Max.accuracy=0.205, Argmax=3 | Avg.test_loss=12.890, Avg.test_accuracy=0.200, Max.test_accuracy=0.220, Argmax=7
        # ※白黒反転時：★Avg.loss=12.903, Avg.accuracy=0.199, Max.accuracy=0.205, Argmax=3 | Avg.test_loss=12.861, Avg.test_accuracy=0.202, Max.test_accuracy=0.220, Argmax=5
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Tanh(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=RMSProp(learning_rate=0.01)))

        # Adam
        # ★Avg.loss=0.114, Avg.accuracy=0.967, Max.accuracy=0.994, Argmax=21 | Avg.test_loss=0.304, Avg.test_accuracy=0.925, Max.test_accuracy=0.975, Argmax=75
        # ※白黒反転時：★Avg.loss=0.004, Avg.accuracy=0.998, Max.accuracy=1.000, Argmax=12 | Avg.test_loss=0.299, Avg.test_accuracy=0.959, Max.test_accuracy=0.965, Argmax=3
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Tanh(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=Adam(learning_rate=0.01, decay_rate1=0.9, decay_rate2=0.9999)))

        # NAG
        # ★Avg.loss=0.024, Avg.accuracy=0.996, Max.accuracy=1.000, Argmax=9 | Avg.test_loss=0.185, Avg.test_accuracy=0.965, Max.test_accuracy=0.975, Argmax=12
        # ※白黒反転時：★Avg.loss=0.021, Avg.accuracy=0.996, Max.accuracy=1.000, Argmax=8 | Avg.test_loss=0.209, Avg.test_accuracy=0.968, Max.test_accuracy=0.980, Argmax=35
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Tanh(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=NAG(learning_rate=0.01, decay_rate=0.9)))

        ##############################
        # 次に3層、ReLU、ミニバッチサイズ10 を固定し、各種最適化を試した。
        ##############################
        # Momentum
        # 　⇒ SGDのときよりかなり早く最大正解率100%に到達した。
        # ★Avg.loss=0.057, Avg.accuracy=0.978, Max.accuracy=1.000, Argmax=22 | Avg.test_loss=0.253, Avg.test_accuracy=0.937, Max.test_accuracy=0.965, Argmax=32
        # ※白黒反転時：★Avg.loss=0.040, Avg.accuracy=0.985, Max.accuracy=1.000, Argmax=8 | Avg.test_loss=0.188, Avg.test_accuracy=0.954, Max.test_accuracy=0.970, Argmax=10
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=Momentum(learning_rate=0.01, decay_rate=0.9)))

        # AdaGrad
        # 　⇒Momentumよりもさらに早く最大正解率100%に到達した。
        # ★Avg.loss=0.103, Avg.accuracy=0.964, Max.accuracy=1.000, Argmax=75 | Avg.test_loss=0.232, Avg.test_accuracy=0.924, Max.test_accuracy=0.960, Argmax=23
        # ※白黒反転時：★Avg.loss=0.001, Avg.accuracy=1.000, Max.accuracy=1.000, Argmax=4 | Avg.test_loss=0.120, Avg.test_accuracy=0.965, Max.test_accuracy=0.970, Argmax=0
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=AdaGrad(learning_rate=0.01)))

        # 同上、AdaDelta　⇒AdaGradと同程度に早く最大正解率100%に到達した。
        # ★Avg.loss=0.021, Avg.accuracy=0.991, Max.accuracy=1.000, Argmax=17 | Avg.test_loss=0.244, Avg.test_accuracy=0.958, Max.test_accuracy=0.970, Argmax=25
        # ※白黒反転時：★Avg.loss=0.001, Avg.accuracy=1.000, Max.accuracy=1.000, Argmax=4 | Avg.test_loss=0.179, Avg.test_accuracy=0.974, Max.test_accuracy=0.975, Argmax=8
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=AdaDelta(decay_rate=0.9)))

        # RMSProp
        # 　⇒20％程度で、学習が進まない。
        # ★Avg.loss=12.889, Avg.accuracy=0.200, Max.accuracy=0.205, Argmax=3 | Avg.test_loss=12.915, Avg.test_accuracy=0.199, Max.test_accuracy=0.220, Argmax=0
        # ※白黒反転時：★Avg.loss=12.886, Avg.accuracy=0.201, Max.accuracy=0.205, Argmax=5 | Avg.test_loss=12.928, Avg.test_accuracy=0.198, Max.test_accuracy=0.220, Argmax=3
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=RMSProp(learning_rate=0.01)))

        # 同上、Adam　⇒Momentumと同程度に早く最大正解率に到達した。最大正解率への到達エポック回数はAdaGradの方が良かった。
        # ★Avg.loss=1.609, Avg.accuracy=0.204, Max.accuracy=0.205, Argmax=0 | Avg.test_loss=1.611, Avg.test_accuracy=0.184, Max.test_accuracy=0.220, Argmax=74
        # ※白黒反転時：★Avg.loss=0.005, Avg.accuracy=0.999, Max.accuracy=1.000, Argmax=7 | Avg.test_loss=0.237, Avg.test_accuracy=0.966, Max.test_accuracy=0.975, Argmax=27
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=Adam(learning_rate=0.01, decay_rate1=0.9, decay_rate2=0.9999)))

        # 同上、Nesterov　⇒Momentum、Adamと同程度に早く最大正解率に到達した。最大正解率への到達エポック回数はAdaGradの方が良かった。
        # ★Avg.loss=0.039, Avg.accuracy=0.986, Max.accuracy=1.000, Argmax=15 | Avg.test_loss=0.333, Avg.test_accuracy=0.944, Max.test_accuracy=0.970, Argmax=16
        # ※白黒反転時：★Avg.loss=0.039, Avg.accuracy=0.986, Max.accuracy=1.000, Argmax=8 | Avg.test_loss=0.186, Avg.test_accuracy=0.955, Max.test_accuracy=0.970, Argmax=26
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=NAG(learning_rate=0.01, decay_rate=0.9)))

        # ★★以上より、ReLUの場合はAdaGradが最も良い。

        ##############################
        # 以下k分割交差検証
        ##############################
        # 2層、Sigmoid、100分割　⇒分割データの12回目で最大100％に達した。
        # ★kfold_num=100: Avg.Loss=0.236, Avg.Accuracy=0.947, Max.Accuracy=1.000, Argmax=11
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 5],
        #               hidden_actfunc=Sigmoid(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               init_weight_stddev=0.01,
        #               learner=KFoldCrossValidation(kfold_num=100, optimizer=SGD(learning_rate=0.01)))

        # 3層、Sigmoid、100分割　⇒3層にするとかえって正解率が落ちた。しかも正解率が毎回ぶれる。
        # ★kfold_num=100: Avg.Loss=1.613, Avg.Accuracy=0.202, Max.Accuracy=0.500, Argmax=12
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Sigmoid(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               init_weight_stddev=0.01,
        #               learner=KFoldCrossValidation(kfold_num=100, optimizer=SGD(learning_rate=0.01)))

        # tanhにしてみてはどうか？
        ##############################
        # 以下、tanh固定で最適化を変えてみた。
        ##############################
        # 3層、tanh、100分割　⇒tanhにすると学習が進み、最大正解率も100%に到達した。
        # SGD
        # ★kfold_num=100: Avg.Loss=0.195, Avg.Accuracy=0.954, Max.Accuracy=1.000, Argmax=2
        # ※白黒反転時：★kfold_num=100: Avg.Loss=0.185, Avg.Accuracy=0.950, Max.Accuracy=1.000, Argmax=13
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Tanh(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               init_weight_stddev=0.01,
        #               learner=KFoldCrossValidation(kfold_num=100, optimizer=SGD(learning_rate=0.01)))

        # Momentum ⇒SGDと比べて、かなり早く学習が進み、第3エポックで最大正解率に到達した。
        # ★kfold_num=100: Avg.Loss=0.025, Avg.Accuracy=0.996, Max.Accuracy=1.000, Argmax=2
        # ※白黒反転時：★kfold_num=100: Avg.Loss=0.017, Avg.Accuracy=0.998, Max.Accuracy=1.000, Argmax=2
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Tanh(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               init_weight_stddev=0.01,
        #               learner=KFoldCrossValidation(kfold_num=100, optimizer=Momentum(learning_rate=0.01, decay_rate=0.9)))

        # AdaGrad ⇒すでに第1エポックで最大正解率に達した。最も早い。★★★
        # ★kfold_num=100: Avg.Loss=0.031, Avg.Accuracy=0.988, Max.Accuracy=1.000, Argmax=2
        # ※白黒反転時：★kfold_num=100: Avg.Loss=0.003, Avg.Accuracy=1.000, Max.Accuracy=1.000, Argmax=0
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Tanh(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               init_weight_stddev=0.01,
        #               learner=KFoldCrossValidation(kfold_num=100, optimizer=AdaGrad(learning_rate=0.01)))

        # AdaDelta ⇒AdaGradと同様、すでに第1エポックで最大正解率に達した。最も早い。★★★
        # ★kfold_num=100: Avg.Loss=0.012, Avg.Accuracy=0.995, Max.Accuracy=1.000, Argmax=2
        # ※白黒反転時：★kfold_num=100: Avg.Loss=0.001, Avg.Accuracy=1.000, Max.Accuracy=1.000, Argmax=0
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Tanh(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               init_weight_stddev=0.01,
        #               learner=KFoldCrossValidation(kfold_num=100, optimizer=AdaDelta(decay_rate=0.9)))

        # RMSProp ⇒最大でも50%。
        # ★kfold_num=100: Avg.Loss=13.076, Avg.Accuracy=0.188, Max.Accuracy=0.600, Argmax=70
        # ※白黒反転時：★kfold_num=100: Avg.Loss=13.233, Avg.Accuracy=0.179, Max.Accuracy=0.500, Argmax=18
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Tanh(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               init_weight_stddev=0.01,
        #               learner=KFoldCrossValidation(kfold_num=100, optimizer=RMSProp(learning_rate=0.01)))

        # Adam  ⇒AdaGrad、AdaDeltaと同様、すでに第1エポックで最大正解率に達した。最も早い。★★★
        # ★kfold_num=100: Avg.Loss=0.146, Avg.Accuracy=0.960, Max.Accuracy=1.000, Argmax=2
        # ※白黒反転時：★kfold_num=100: Avg.Loss=0.009, Avg.Accuracy=0.995, Max.Accuracy=1.000, Argmax=0
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Tanh(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               init_weight_stddev=0.01,
        #               learner=KFoldCrossValidation(kfold_num=100, optimizer=Adam(learning_rate=0.01, decay_rate1=0.9, decay_rate2=0.9999)))

        # NAG ⇒Momentumと同様、第3エポックで最大正解率に到達した。
        # ★kfold_num=100: Avg.Loss=0.023, Avg.Accuracy=0.996, Max.Accuracy=1.000, Argmax=2
        # ※白黒反転時：★kfold_num=100: Avg.Loss=0.016, Avg.Accuracy=0.997, Max.Accuracy=1.000, Argmax=2
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Tanh(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               init_weight_stddev=0.01,
        #               learner=KFoldCrossValidation(kfold_num=100, optimizer=NAG(learning_rate=0.01, decay_rate=0.9)))

        ##############################
        # では、ReLU固定で最適化を変えてみてはどうか？
        ##############################
        # 3層、ReLU、100分割　⇒tanhよりも最大正解率に到達するのが遅かった。
        # ★kfold_num=100: Avg.Loss=0.259, Avg.Accuracy=0.900, Max.Accuracy=1.000, Argmax=16
        # ※白黒反転時：★kfold_num=100: Avg.Loss=0.299, Avg.Accuracy=0.901, Max.Accuracy=1.000, Argmax=22
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               init_weight_stddev=0.01,
        #               learner=KFoldCrossValidation(kfold_num=100, optimizer=SGD(learning_rate=0.01)))

        # Momentum ⇒第3エポックで最大正解率に達した。tanhでのSGDと比べてかなり早く学習が進んだ。
        # ★kfold_num=100: Avg.Loss=0.047, Avg.Accuracy=0.986, Max.Accuracy=1.000, Argmax=2
        # ※白黒反転時：★kfold_num=100: Avg.Loss=0.030, Avg.Accuracy=0.986, Max.Accuracy=1.000, Argmax=2
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               init_weight_stddev=0.01,
        #               learner=KFoldCrossValidation(kfold_num=100, optimizer=Momentum(learning_rate=0.01, decay_rate=0.9)))

        # AdaGrad ⇒tanhでの場合と同様。すでに第1エポックで最大正解率に達した。最も早い。★★★
        # ★kfold_num=100: Avg.Loss=0.046, Avg.Accuracy=0.986, Max.Accuracy=1.000, Argmax=2
        # ※白黒反転時：★kfold_num=100: Avg.Loss=0.005, Avg.Accuracy=0.999, Max.Accuracy=1.000, Argmax=0
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               init_weight_stddev=0.01,
        #               learner=KFoldCrossValidation(kfold_num=100, optimizer=AdaGrad(learning_rate=0.01)))

        # AdaDelta ⇒AdaGradと同程度で、第3エポックで最大正解率に到達した。
        # ★kfold_num=100: Avg.Loss=0.031, Avg.Accuracy=0.990, Max.Accuracy=1.000, Argmax=2
        # ※白黒反転時：★kfold_num=100: Avg.Loss=0.002, Avg.Accuracy=0.998, Max.Accuracy=1.000, Argmax=2
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               init_weight_stddev=0.01,
        #               learner=KFoldCrossValidation(kfold_num=100, optimizer=AdaDelta(decay_rate=0.9)))

        # RMSProp ⇒最大でも50%。ぶれが大きい。
        # ★kfold_num=100: Avg.Loss=12.943, Avg.Accuracy=0.197, Max.Accuracy=0.600, Argmax=70
        # ※白黒反転時：★kfold_num=100: Avg.Loss=12.346, Avg.Accuracy=0.234, Max.Accuracy=0.500, Argmax=10
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               init_weight_stddev=0.01,
        #               learner=KFoldCrossValidation(kfold_num=100, optimizer=RMSProp(learning_rate=0.01)))

        # Adam ⇒AdaDeltaと同様、第2エポックで最大正解率に到達した。
        # ★kfold_num=100: Avg.Loss=1.613, Avg.Accuracy=0.109, Max.Accuracy=0.400, Argmax=7
        # ※白黒反転時：★kfold_num=100: Avg.Loss=0.001, Avg.Accuracy=0.999, Max.Accuracy=1.000, Argmax=1
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               init_weight_stddev=0.01,
        #               learner=KFoldCrossValidation(kfold_num=100, optimizer=Adam(learning_rate=0.01, decay_rate1=0.9, decay_rate2=0.9999)))

        # NAG ⇒AdaDeltaと同様、第3エポックで最大正解率に到達した。
        # ★kfold_num=100: Avg.Loss=0.031, Avg.Accuracy=0.987, Max.Accuracy=1.000, Argmax=2
        # ※白黒反転時：★kfold_num=100: Avg.Loss=0.030, Avg.Accuracy=0.986, Max.Accuracy=1.000, Argmax=2
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               init_weight_stddev=0.01,
        #               learner=KFoldCrossValidation(kfold_num=100, optimizer=NAG(learning_rate=0.01, decay_rate=0.9)))


        # ##################################################
        # バッチ正規化（Algorithm1の実験）（arXiv:1502.03167v3参照）
        # ##################################################
        # 次に、Day4までの講義内容を実装した中で、
        # ミニバッチ学習を採用したモデルで最も性能が良かったモデル『Minibatch-ReLU-AdaGrad』について、
        # バッチ正規化を実施してみたらどうなるか？過学習は抑制されるか？
        # （元の結果：3層）★Avg.loss=0.001, Avg.accuracy=1.000, Max.accuracy=1.000, Argmax=4 | Avg.test_loss=0.120, Avg.test_accuracy=0.965, Max.test_accuracy=0.970, Argmax=0
        # （元の結果：5層）★Avg.loss=0.091, Avg.accuracy=0.957, Max.accuracy=1.000, Argmax=59 | Avg.test_loss=0.223, Avg.test_accuracy=0.934, Max.test_accuracy=0.985, Argmax=59
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=AdaGrad(learning_rate=0.01))
        #               )
        # ↓
        # ●γ＝1.0、β＝0.0
        #   ★Avg.loss=0.158, Avg.accuracy=0.966, Max.accuracy=0.981, Argmax=21 | Avg.test_loss=0.195, Avg.test_accuracy=0.950, Max.test_accuracy=0.970, Argmax=17
        #   ⇒多少損失が大きくなった。
        # ●γ＝0.1、β＝0.0
        #   ★Avg.loss=0.671, Avg.accuracy=0.697, Max.accuracy=0.786, Argmax=99 | Avg.test_loss=0.719, Avg.test_accuracy=0.671, Max.test_accuracy=0.785, Argmax=99
        #   ⇒γを小さくすると、損失が増加した。学習不足。
        # ●γ＝10.0、β＝0.0
        #   ★Avg.loss=0.025, Avg.accuracy=0.994, Max.accuracy=1.000, Argmax=96 | Avg.test_loss=0.064, Avg.test_accuracy=0.985, Max.test_accuracy=0.990, Argmax=15
        #   ⇒γを大きくすると、損失が大きく減少した。過学習を引き起こすらしい。
        # 〇γ＝1.0、β＝0.5
        #   ★Avg.loss=0.056, Avg.accuracy=0.990, Max.accuracy=0.994, Argmax=15 | Avg.test_loss=0.098, Avg.test_accuracy=0.977, Max.test_accuracy=0.985, Argmax=14
        #   ⇒ReLUの場合、x=0.5が中心となるためなのか、割と良い結果となった。
        # 〇γ＝1.0、β＝1.0
        #   ★Avg.loss=0.030, Avg.accuracy=0.993, Max.accuracy=0.996, Argmax=23 | Avg.test_loss=0.071, Avg.test_accuracy=0.980, Max.test_accuracy=0.990, Argmax=84
        # 〇γ＝1.0、β＝10.0
        #   ★Avg.loss=0.890, Avg.accuracy=0.685, Max.accuracy=0.792, Argmax=99 | Avg.test_loss=0.899, Avg.test_accuracy=0.679, Max.test_accuracy=0.800, Argmax=45
        #   ⇒βを増やすと、分布の平均0からのずれが大きくなるので0の方が良いのでは？
        # ▲γ＝2.5、β＝0.5
        #   ★Avg.loss=0.035, Avg.accuracy=0.993, Max.accuracy=0.996, Argmax=21 | Avg.test_loss=0.065, Avg.test_accuracy=0.983, Max.test_accuracy=0.990, Argmax=21
        # ▲γ＝5.0、β＝0.5
        #   ★Avg.loss=0.026, Avg.accuracy=0.994, Max.accuracy=0.998, Argmax=29 | Avg.test_loss=0.078, Avg.test_accuracy=0.981, Max.test_accuracy=0.985, Argmax=1
        # ▲γ＝10.0、β＝0.5
        #   ★Avg.loss=0.019, Avg.accuracy=0.996, Max.accuracy=1.000, Argmax=39 | Avg.test_loss=0.059, Avg.test_accuracy=0.985, Max.test_accuracy=0.990, Argmax=2
        #   ⇒γ＝10.0、β＝0.5は、平均損失がDay4のときのちょうど半分程度になった。
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=AdaGrad(learning_rate=0.01)),
        #               batch_normal=BatchNormal(gamma=10.0, beta=0.5)
        #               )
        #　↓
        # 10層にするとアクティベーション分布はどうなるか？
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 100, 100, 100, 100, 100, 100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=AdaGrad(learning_rate=0.01)),
        #               batch_normal=BatchNormal(gamma=10.0, beta=1.0)
        #               # batch_normal=BatchNormal(gamma=10.0, beta=0.5)
        #               )
        #
        # ＜Tanhにした場合＞
        # ●γ＝10.0、β＝0.0
        #   ★Avg.loss=0.052, Avg.accuracy=0.988, Max.accuracy=0.998, Argmax=95 | Avg.test_loss=0.082, Avg.test_accuracy=0.980, Max.test_accuracy=0.995, Argmax=36
        # ▲γ＝10.0、β＝0.5
        #   ★Avg.loss=0.050, Avg.accuracy=0.987, Max.accuracy=0.998, Argmax=82 | Avg.test_loss=0.104, Avg.test_accuracy=0.973, Max.test_accuracy=0.985, Argmax=31
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 100, 100, 5],
        #               hidden_actfunc=Tanh(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=AdaGrad(learning_rate=0.01)),
        #               batch_normal=BatchNormal(gamma=10.0, beta=0.5)
        #               )

        ##############################
        # 以下バッチ正規化Algorithm2
        ##############################
        # MiniBatch-ReLU-AdaGrad
        # ■3層
        #   バッチ正規化なし：★Avg.loss=0.001, Avg.accuracy=1.000, Max.accuracy=1.000, Argmax=4 | Avg.test_loss=0.120, Avg.test_accuracy=0.965, Max.test_accuracy=0.970, Argmax=0
        #   gamma=1.0, beta=0.0, moving_decay=0.1：★Avg.loss=0.056, Avg.accuracy=0.986, Max.accuracy=1.000, Argmax=52 | Avg.test_loss=0.251, Avg.test_accuracy=0.937, Max.test_accuracy=0.975, Argmax=25
        #   gamma=5.0, beta=0.0, moving_decay=0.1：★Avg.loss=0.029, Avg.accuracy=0.993, Max.accuracy=1.000, Argmax=16 | Avg.test_loss=0.203, Avg.test_accuracy=0.949, Max.test_accuracy=0.980, Argmax=24
        #   gamma=10.0, beta=0.0, moving_decay=0.1：★Avg.loss=0.037, Avg.accuracy=0.990, Max.accuracy=1.000, Argmax=35 | Avg.test_loss=0.207, Avg.test_accuracy=0.947, Max.test_accuracy=0.980, Argmax=8
        #
        #   gamma=5.0, beta=0.1, moving_decay=0.1：★Avg.loss=0.021, Avg.accuracy=0.995, Max.accuracy=1.000, Argmax=13 | Avg.test_loss=0.172, Avg.test_accuracy=0.955, Max.test_accuracy=0.985, Argmax=79
        #   gamma=5.0, beta=0.5, moving_decay=0.1：★Avg.loss=0.018, Avg.accuracy=0.996, Max.accuracy=1.000, Argmax=13 | Avg.test_loss=0.186, Avg.test_accuracy=0.952, Max.test_accuracy=0.980, Argmax=24
        #   gamma=5.0, beta=1.0, moving_decay=0.1：★Avg.loss=0.018, Avg.accuracy=0.995, Max.accuracy=1.000, Argmax=13 | Avg.test_loss=0.180, Avg.test_accuracy=0.951, Max.test_accuracy=0.980, Argmax=24
        #   （↑gamma=5.0の中では最も損失が低い）
        #   gamma=5.0, beta=10.0, moving_decay=0.1：★Avg.loss=0.150, Avg.accuracy=0.954, Max.accuracy=0.985, Argmax=3 | Avg.test_loss=0.287, Avg.test_accuracy=0.918, Max.test_accuracy=0.965, Argmax=23
        #
        #   gamma=5.0, beta=1.0, moving_decay=0.1：（上記で実施済み）
        #   gamma=5.0, beta=1.0, moving_decay=0.5：★Avg.loss=0.005, Avg.accuracy=0.999, Max.accuracy=1.000, Argmax=12 | Avg.test_loss=0.125, Avg.test_accuracy=0.963, Max.test_accuracy=0.980, Argmax=8
        #   gamma=5.0, beta=1.0, moving_decay=0.9：★Avg.loss=0.003, Avg.accuracy=1.000, Max.accuracy=1.000, Argmax=11 | Avg.test_loss=0.104, Avg.test_accuracy=0.967, Max.test_accuracy=0.985, Argmax=57
        #   （↑gamma=5.0、beta=1.0の中では最も損失が低い）
        #   gamma=5.0, beta=1.0, moving_decay=0.95：★Avg.loss=0.003, Avg.accuracy=1.000, Max.accuracy=1.000, Argmax=11 | Avg.test_loss=0.101, Avg.test_accuracy=0.968, Max.test_accuracy=0.980, Argmax=57
        #   （↑gamma=5.0、beta=1.0の中では最も損失が低いが、正解率は落ちた。）
        #
        #★↓提出
        #    gamma=5.0, beta=0.5, moving_decay=0.9：★Avg.loss=0.004, Avg.accuracy=1.000, Max.accuracy=1.000, Argmax=13 | Avg.test_loss=0.100, Avg.test_accuracy=0.970, Max.test_accuracy=0.985, Argmax=36
        #   （↑beta=0.5にするとさらに損失が低くなった。しかも学習が早くなった。）
        #
        #   gamma=5.0, beta=0.5, moving_decay=0.95：★Avg.loss=0.004, Avg.accuracy=1.000, Max.accuracy=1.000, Argmax=13 | Avg.test_loss=0.097, Avg.test_accuracy=0.970, Max.test_accuracy=0.985, Argmax=57
        #   （↑moving_decay=0.5にするとさらに損失が低くなったが、学習は遅くなった。）
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               init_weight_stddev=0.01,
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=AdaGrad(learning_rate=0.01)),
        #               batch_normal_params=BatchNormalParams(gamma=5.0, beta=0.5, moving_decay=0.9)
        #               )

        # ■5層
        #   バッチ正規化なし：★Avg.loss=0.091, Avg.accuracy=0.957, Max.accuracy=1.000, Argmax=59 | Avg.test_loss=0.223, Avg.test_accuracy=0.934, Max.test_accuracy=0.985, Argmax=59
        #   gamma=1.0, beta=0.0, moving_decay=0.1：★Avg.loss=0.624, Avg.accuracy=0.767, Max.accuracy=0.894, Argmax=73 | Avg.test_loss=0.652, Avg.test_accuracy=0.766, Max.test_accuracy=0.895, Argmax=64
        #   gamma=5.0, beta=0.0, moving_decay=0.1：★Avg.loss=0.382, Avg.accuracy=0.874, Max.accuracy=0.973, Argmax=12 | Avg.test_loss=0.395, Avg.test_accuracy=0.869, Max.test_accuracy=0.970, Argmax=17
        #   gamma=10.0, beta=0.0, moving_decay=0.1：★Avg.loss=0.795, Avg.accuracy=0.731, Max.accuracy=0.931, Argmax=2 | Avg.test_loss=0.789, Avg.test_accuracy=0.744, Max.test_accuracy=0.935, Argmax=2
        #
        #   gamma=5.0, beta=0.1, moving_decay=0.1：★Avg.loss=0.432, Avg.accuracy=0.861, Max.accuracy=0.971, Argmax=3 | Avg.test_loss=0.428, Avg.test_accuracy=0.864, Max.test_accuracy=0.980, Argmax=12
        #   gamma=5.0, beta=0.5, moving_decay=0.1：★Avg.loss=0.252, Avg.accuracy=0.923, Max.accuracy=0.986, Argmax=16 | Avg.test_loss=0.260, Avg.test_accuracy=0.920, Max.test_accuracy=0.990, Argmax=19
        #   （↑gamma=5.0の中では最も損失が低い）
        #   gamma=5.0, beta=1.0, moving_decay=0.1：★Avg.loss=0.373, Avg.accuracy=0.874, Max.accuracy=0.986, Argmax=12 | Avg.test_loss=0.395, Avg.test_accuracy=0.862, Max.test_accuracy=0.990, Argmax=5
        #   gamma=5.0, beta=10.0, moving_decay=0.1：★Avg.loss=0.755, Avg.accuracy=0.698, Max.accuracy=0.824, Argmax=85 | Avg.test_loss=0.779, Avg.test_accuracy=0.674, Max.test_accuracy=0.820, Argmax=80

        #   gamma=5.0, beta=0.5, moving_decay=0.1：（上記で実施済み）
        #   gamma=5.0, beta=0.5, moving_decay=0.5：★Avg.loss=0.161, Avg.accuracy=0.959, Max.accuracy=0.989, Argmax=3 | Avg.test_loss=0.164, Avg.test_accuracy=0.954, Max.test_accuracy=0.990, Argmax=36
        #★↓提出候補
        #   gamma=5.0, beta=0.5, moving_decay=0.9：★Avg.loss=0.136, Avg.accuracy=0.968, Max.accuracy=0.993, Argmax=4 | Avg.test_loss=0.137, Avg.test_accuracy=0.964, Max.test_accuracy=0.990, Argmax=15
        #   （↑gamma=5.0、beta=0.5の中では最も損失が低い）
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               init_weight_stddev=0.01,
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=AdaGrad(learning_rate=0.01)),
        #               batch_normal_params=BatchNormalParams(gamma=5.0, beta=0.5, moving_decay=0.9)
        #               )
        # ------------------------------
        # k分割での最良モデル『KFold-Tanh-AdaDelta』でバッチ正規化をやってみる。
        #   gamma=5.0, beta=0.5, moving_decay=0.9：★Avg.l_loss=0.0062, Avg.l_accuracy=0.9979, Max.l_accuracy=0.9990, l_argmax=9 | Avg.v_loss=0.0051, Avg.v_accuracy=0.9980, Max.v_accuracy=1.0000, v_argmax=0
        #   ↑ミニバッチのときと比べてさらに損失が低い。
        # ------------------------------
        # model = DNN(input_size=784,
        #             layer_size_list=[100, 100, 5],
        #             hidden_actfunc=Tanh(),
        #             output_actfunc=SoftmaxWithLoss(),
        #             loss_func=CrossEntropyError(),
        #             init_weight_stddev=0.01,
        #             learner=KFoldCrossValidation(kfold_num=100, optimizer=AdaDelta(decay_rate=0.9)),
        #             batch_normal_params=BatchNormalParams(gamma=5.0, beta=0.5, moving_decay=0.9)
        #             )
        #
        # ノード数を増やすと？
        #   gamma=5.0, beta=0.5, moving_decay=0.9：★Avg.l_loss=0.0092, Avg.l_accuracy=0.9985, Max.l_accuracy=0.9990, l_argmax=6 | Avg.v_loss=0.0106, Avg.v_accuracy=0.9980, Max.v_accuracy=1.0000, v_argmax=1
        # model = DNN(input_size=784,
        #             layer_size_list=[200, 200, 5],
        #             hidden_actfunc=Tanh(),
        #             output_actfunc=SoftmaxWithLoss(),
        #             loss_func=CrossEntropyError(),
        #             init_weight_stddev=0.01,
        #             learner=KFoldCrossValidation(kfold_num=100, optimizer=AdaDelta(decay_rate=0.9)),
        #             batch_normal_params=BatchNormalParams(gamma=5.0, beta=0.5, moving_decay=0.9)
        #             )
        #
        # 5層ではどうか？：★Avg.l_loss=0.0447, Avg.l_accuracy=0.9899, Max.l_accuracy=0.9949, l_argmax=80 | Avg.v_loss=0.0319, Avg.v_accuracy=0.9900, Max.v_accuracy=1.0000, v_argmax=0
        # 5層＋ノード数200：★Avg.l_loss=0.0089, Avg.l_accuracy=0.9976, Max.l_accuracy=1.0000, l_argmax=31 | Avg.v_loss=0.0063, Avg.v_accuracy=0.9990, Max.v_accuracy=1.0000, v_argmax=0
        # model = DNN(input_size=784,
        #             layer_size_list=[200, 200, 200, 200, 5],
        #             #layer_size_list=[100, 100, 100, 100, 5],
        #             hidden_actfunc=Tanh(),
        #             output_actfunc=SoftmaxWithLoss(),
        #             loss_func=CrossEntropyError(),
        #             init_weight_stddev=0.01,
        #             learner=KFoldCrossValidation(kfold_num=100, optimizer=AdaDelta(decay_rate=0.9)),
        #             batch_normal_params=BatchNormalParams(gamma=5.0, beta=0.5, moving_decay=0.9)
        #             )

        # ------------------------------
        # あまり性能が良くなかったモデルを、正規化で実行してみる。
        # ------------------------------
        # 正規化無し：★Avg.l_loss=0.3680, Avg.l_accuracy=0.8615, Max.l_accuracy=1.0000, l_argmax=71 | Avg.v_loss=0.4619, Avg.v_accuracy=0.8279, Max.v_accuracy=0.9600, v_argmax=57
        # 正規化あり：★Avg.l_loss=0.0416, Avg.l_accuracy=0.9874, Max.l_accuracy=1.0000, l_argmax=12 | Avg.v_loss=0.1593, Avg.v_accuracy=0.9554, Max.v_accuracy=0.9800, v_argmax=10
        # model = DNN(input_size=784,
        #             layer_size_list=[100, 100, 5],
        #             hidden_actfunc=ReLU(),
        #             output_actfunc=SoftmaxWithLoss(),
        #             loss_func=CrossEntropyError(),
        #             learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=SGD(learning_rate=0.01)),
        #             batch_normal_params=BatchNormalParams(gamma=5.0, beta=0.5, moving_decay=0.9)
        #             )

        # ------------------------------
        # グラフ化用
        # ------------------------------
        # 実験No.17：★Avg.l_loss=0.0014, Avg.l_accuracy=0.9998, Max.l_accuracy=1.0000, l_argmax=4 | Avg.v_loss=0.1199, Avg.v_accuracy=0.9653, Max.v_accuracy=0.9700, v_argmax=0
        # 実験No.17'-A（重み初期値）：★Avg.l_loss=0.0010, Avg.l_accuracy=0.9999, Max.l_accuracy=1.0000, l_argmax=3 | Avg.v_loss=0.1277, Avg.v_accuracy=0.9646, Max.v_accuracy=0.9650, v_argmax=0
        # 実験No.17'-E（バッチ正規化）：★Avg.l_loss=0.0041, Avg.l_accuracy=0.9995, Max.l_accuracy=1.0000, l_argmax=13 | Avg.v_loss=0.0995, Avg.v_accuracy=0.9696, Max.v_accuracy=0.9850, v_argmax=36
        # model = DNN(input_size=784,
        #             layer_size_list=[100, 100, 5],
        #             hidden_actfunc=ReLU(),
        #             output_actfunc=SoftmaxWithLoss(),
        #             loss_func=CrossEntropyError(),
        #             init_weight_stddev=0.01,
        #             #init_weight_change=True,  # 実験No.17'-A
        #             learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=AdaGrad(learning_rate=0.01)),
        #             batch_normal_params=BatchNormalParams(gamma=5.0, beta=0.5, moving_decay=0.9)  # 実験No.17'-E
        #             )

        # 実験25について、k分割交差検証のkを減らして提出してみる。
        # ●k分割交差検証で最も良かったモデル：Kfold-Tanh-AdaDelta
        # 3層
        #   kfold_num=100：★Avg.l_loss=0.0017, Avg.l_accuracy=0.9996, Max.l_accuracy=1.0000, l_argmax=4 | Avg.v_loss=0.0010, Avg.v_accuracy=1.0000, Max.v_accuracy=1.0000, v_argmax=0
        #   kfold_num=10：★Avg.l_loss=0.2399, Avg.l_accuracy=0.9353, Max.l_accuracy=0.9978, l_argmax=7 | Avg.v_loss=0.2454, Avg.v_accuracy=0.9360, Max.v_accuracy=1.0000, v_argmax=6
        # model = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Tanh(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               init_weight_stddev=0.01,
        #               #learner=KFoldCrossValidation(kfold_num=100, optimizer=AdaDelta(decay_rate=0.9))
        #               learner=KFoldCrossValidation(kfold_num=10, optimizer=AdaDelta(decay_rate=0.9))
        #               )

        # 実験25派生：KFold-Tanh-AdaDelta／kfold_num=10／10層／初期値変更あり／バッチ正規化あり(gamma=2.0, beta=0.0, moving_decay=0.9)
        model = DNN(input_size=784,
                    layer_size_list=[100, 100, 100, 100, 100, 100, 100, 100, 100, 5],
                    hidden_actfunc=Tanh(),
                    output_actfunc=SoftmaxWithLoss(),
                    loss_func=CrossEntropyError(),
                    init_weight_stddev=0.01,
                    init_weight_change=True,
                    learner=KFoldCrossValidation(kfold_num=10, optimizer=AdaDelta(decay_rate=0.9)),
                    batch_normal_params=BatchNormalParams(gamma=2.0, beta=0.0, moving_decay=0.9)
                    )

        return model