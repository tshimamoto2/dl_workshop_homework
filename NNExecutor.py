import numpy as np
from DNN import DNN
from layers import Sigmoid, Tanh, ReLU, SoftmaxWithLoss, CrossEntropyError
from learners import MiniBatch, KFoldCrossValidation
from optimizers import SGD, Momentum, AdaGrad, AdaDelta, RMSProp, Adam, NAG

# 2018/07/21提出時点での最適組み合わせ：
# ・レイヤー数：3層
# ・隠れ層の活性化関数：ReLU
# ・出力層の活性化関数：ソフトマックス関数
# ・学習ロジック：k分割交差検証
#
# 以下固定
# ・損失関数：クロスエントロピー誤差関数
# ・初期重み標準偏差：0.01
# ・学習率：0.01
class NNExecutor:
    def __init__(self):
        ##################################################
        # 以下リファクタリング後：各活性化関数、損失関数、最適化を切り替えることができるようにした。
        ##################################################
        # （変遷１）入力データは単純正規化（255で割るだけ）／2層／エポック数3／ミニバッチサイズ100／Sigmoid／SGD
        # ⇒Accuracy20%程度。
        # ★Avg.loss=1.608, Avg.accuracy=0.248, Max.accuracy=0.345, Argmax=2 | Avg.test_loss=1.609, Avg.test_accuracy=0.245, Max.test_accuracy=0.335, Argmax=2
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[100, 5],
        #               hidden_actfunc=Sigmoid(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=3, mini_batch_size=100, optimizer=SGD(learning_rate=0.01)))

        # （変遷２）エポック数を変えてみた。エポック数100
        # ⇒★Avg.loss=1.467, Avg.accuracy=0.733, Max.accuracy=0.922, Argmax=93 | Avg.test_loss=1.477, Avg.test_accuracy=0.715, Max.test_accuracy=0.935, Argmax=96
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[100, 5],
        #               hidden_actfunc=Sigmoid(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=100, optimizer=SGD(learning_rate=0.01)))

        # （変遷３－１）ミニバッチサイズを変えてみた。エポック数は100でミニバッチサイズ50の場合
        # ⇒★Avg.loss=1.062, Avg.accuracy=0.818, Max.accuracy=0.959, Argmax=88 | Avg.test_loss=1.082, Avg.test_accuracy=0.810, Max.test_accuracy=0.960, Argmax=82
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[100, 5],
        #               hidden_actfunc=Sigmoid(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=50, optimizer=SGD(learning_rate=0.01)))

        # （変遷３－２）ミニバッチサイズを変えてみた。エポック数は100でミニバッチサイズ10の場合
        # ⇒ミニバッチサイズ50のときよりもさらに学習が進み、正解率も高くなった。
        # ★Avg.loss=0.289, Avg.accuracy=0.936, Max.accuracy=0.995, Argmax=98 | Avg.test_loss=0.309, Avg.test_accuracy=0.931, Max.test_accuracy=0.985, Argmax=30
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[100, 5],
        #               hidden_actfunc=Sigmoid(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=SGD(learning_rate=0.01)))

        # ★★★以降、エポック数100、ミニバッチサイズ10で行く。

        # （変遷４）層数を変えてみた。3層
        # ⇒学習が全く進まない。
        # ★Avg.loss=1.612, Avg.accuracy=0.201, Max.accuracy=0.210, Argmax=74 | Avg.test_loss=1.614, Avg.test_accuracy=0.195, Max.test_accuracy=0.220, Argmax=1
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Sigmoid(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=SGD(learning_rate=0.01)))

        # （変遷４－１）3層で、tanhに変えてみた。
        # ⇒ 学習が進み、最大Accuracyが96%になった。
        # ★Avg.loss=0.222, Avg.accuracy=0.944, Max.accuracy=1.000, Argmax=66 | Avg.test_loss=0.336, Avg.test_accuracy=0.906, Max.test_accuracy=0.960, Argmax=28
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Tanh(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=SGD(learning_rate=0.01)))

        # （変遷４－２）3層で、ReLUに変えてみた。
        # ⇒学習が進んだ。
        # ★Avg.loss=0.308, Avg.accuracy=0.876, Max.accuracy=1.000, Argmax=63 | Avg.test_loss=0.461, Avg.test_accuracy=0.838, Max.test_accuracy=0.965, Argmax=73
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=SGD(learning_rate=0.01)))

        # （変遷４－３）試しにReLUで4層にしてみた。
        # ⇒学習が全く進まない。
        # ★Avg.loss=1.609, Avg.accuracy=0.211, Max.accuracy=0.211, Argmax=0 | Avg.test_loss=1.614, Avg.test_accuracy=0.155, Max.test_accuracy=0.155, Argmax=0
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[100, 100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=SGD(learning_rate=0.01)))

        # （変遷４－４）ReLUで5層にしてみた。
        # ⇒学習が全く進まない。
        # ★Avg.loss=1.609, Avg.accuracy=0.209, Max.accuracy=0.209, Argmax=0 | Avg.test_loss=1.613, Avg.test_accuracy=0.165, Max.test_accuracy=0.165, Argmax=0
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[100, 100, 100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=SGD(learning_rate=0.01)))

        # （変遷４－５）試しにReLUで10層にしてみた。
        # ⇒学習が全く進まない。
        # ★Avg.loss=1.609, Avg.accuracy=0.209, Max.accuracy=0.209, Argmax=0 | Avg.test_loss=1.612, Avg.test_accuracy=0.165, Max.test_accuracy=0.165, Argmax=0
        # self.nn = DNN(input_size=784,
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
        # self.nn = DNN(input_size=784,
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
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[10, 5],
        #               hidden_actfunc=Sigmoid(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=SGD(learning_rate=0.01)))

        ##############################
        # 以下、Day5の内容について調査・実験
        ##############################
        # 元の問題点：（変遷４－４）で学習が全く進まない。という問題点があった。
        # （変遷６－１）Sigmoidの場合にXavierの初期値を使用するとどうなるか？
        # ・標準偏差0.01固定の場合：★Avg.loss=1.612, Avg.accuracy=0.204, Max.accuracy=0.209, Argmax=0 | Avg.test_loss=1.617, Avg.test_accuracy=0.185, Max.test_accuracy=0.235, Argmax=24
        # ↓
        # ・Xavierの初期値の場合：★Avg.loss=1.582, Avg.accuracy=0.296, Max.accuracy=0.726, Argmax=81 | Avg.test_loss=1.589, Avg.test_accuracy=0.271, Max.test_accuracy=0.690, Argmax=81
        # ⇒多少不安定だが（エポックごとに性能がぶれる）が、性能は良くなった。確かに効果があることが分かった。
        # self.nn = DNN(input_size=784,
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
        # self.nn = DNN(input_size=784,
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
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[100, 100, 100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=SGD(learning_rate=0.01)),
        #               init_weight_change=True   # 今回新たに実装。
        #               )

        ##############################
        # 以下ミニバッチサイズ10
        ##############################
        # 2層、シグモイド、ミニバッチサイズ10　⇒学習が進んだ。
        # ★Avg.Loss=0.285, Avg.Accuracy=0.937, Max.Accuracy=0.995, Argmax=95
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[100, 5],
        #               hidden_actfunc=Sigmoid(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, learning_rate=0.01))

        # 2層、tanh、ミニバッチサイズ10　⇒シグモイドと同じ程度。
        # ★Avg.Loss=0.063, Avg.Accuracy=0.994, Max.Accuracy=1.000, Argmax=35
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[100, 5],
        #               hidden_actfunc=Tanh(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, learning_rate=0.01))

        # 2層、ReLU、ミニバッチサイズ＝10　⇒正解率がシグモイドより高いが、tanhよりも低い。tanhｙりも学習が進んでいない。
        # ★Avg.Loss=0.069, Avg.Accuracy=0.989, Max.Accuracy=1.000, Argmax=43
        # self.nn = DNN(input_size=784,
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
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Sigmoid(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=SGD(learning_rate=0.01)))

        # 3層、tanh、ミニバッチサイズ10　⇒シグモイドに比べて良くなった。おそらくシグモイドでは勾配消失が起こっていると思われる。
        # ★Avg.loss=0.222, Avg.accuracy=0.944, Max.accuracy=1.000, Argmax=66 | Avg.test_loss=0.336, Avg.test_accuracy=0.906, Max.test_accuracy=0.960, Argmax=28
        # ※白黒反転時：★Avg.loss=0.212, Avg.accuracy=0.930, Max.accuracy=1.000, Argmax=75 | Avg.test_loss=0.347, Avg.test_accuracy=0.892, Max.test_accuracy=0.960, Argmax=81
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Tanh(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=SGD(learning_rate=0.01)))

        # 3層、ReLU、ミニバッチサイズ＝10　⇒tanhと比べて早く最大正解率100%に到達した。
        # ★Avg.loss=0.308, Avg.accuracy=0.876, Max.accuracy=1.000, Argmax=63 | Avg.test_loss=0.461, Avg.test_accuracy=0.838, Max.test_accuracy=0.965, Argmax=73
        # ※白黒反転時：★Avg.loss=0.368, Avg.accuracy=0.862, Max.accuracy=1.000, Argmax=71 | Avg.test_loss=0.462, Avg.test_accuracy=0.828, Max.test_accuracy=0.960, Argmax=57
        # self.nn = DNN(input_size=784,
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
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Sigmoid(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=Momentum(learning_rate=0.01, decay_rate=0.9)))

        # AdaGrad
        # 　⇒Momentumより正解率が低く、学習も遅い。
        # ★Avg.loss=0.958, Avg.accuracy=0.562, Max.accuracy=0.640, Argmax=97 | Avg.test_loss=1.016, Avg.test_accuracy=0.561, Max.test_accuracy=0.660, Argmax=86
        # ※白黒反転時：★Avg.loss=0.728, Avg.accuracy=0.680, Max.accuracy=0.751, Argmax=97 | Avg.test_loss=0.781, Avg.test_accuracy=0.656, Max.test_accuracy=0.715, Argmax=84
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Sigmoid(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=AdaGrad(learning_rate=0.01)))

        # AdaDelta
        # 　⇒AdaGradより正解率が高い。学習も早い。ただし、Momentumよりも多少遅れて最大正解率に到達している。
        # ★Avg.loss=0.526, Avg.accuracy=0.781, Max.accuracy=0.969, Argmax=97 | Avg.test_loss=0.796, Avg.test_accuracy=0.738, Max.test_accuracy=0.900, Argmax=97
        # ※白黒反転時：★Avg.loss=0.120, Avg.accuracy=0.949, Max.accuracy=1.000, Argmax=70 | Avg.test_loss=0.327, Avg.test_accuracy=0.910, Max.test_accuracy=0.960, Argmax=34
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Sigmoid(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=AdaDelta(decay_rate=0.9)))

        # RMSProp
        # 　⇒20％程度で、学習が進まない。
        # ★Avg.loss=12.897, Avg.accuracy=0.200, Max.accuracy=0.205, Argmax=3 | Avg.test_loss=12.883, Avg.test_accuracy=0.201, Max.test_accuracy=0.220, Argmax=0
        # ※白黒反転時：★Avg.loss=12.897, Avg.accuracy=0.200, Max.accuracy=0.205, Argmax=3 | Avg.test_loss=12.883, Avg.test_accuracy=0.201, Max.test_accuracy=0.220, Argmax=0
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Sigmoid(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=RMSProp(learning_rate=0.01)))

        # 同上、Adam　⇒最大でも86.8％。
        # ★Avg.loss=0.100, Avg.accuracy=0.962, Max.accuracy=1.000, Argmax=57 | Avg.test_loss=0.317, Avg.test_accuracy=0.918, Max.test_accuracy=0.980, Argmax=92
        # ※白黒反転時：★Avg.loss=0.374, Avg.accuracy=0.868, Max.accuracy=0.922, Argmax=73 | Avg.test_loss=0.561, Avg.test_accuracy=0.820, Max.test_accuracy=0.865, Argmax=48
        # TODO 途中で1回Overflowを起こした。原因不明。
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Sigmoid(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=Adam(learning_rate=0.01, decay_rate1=0.9, decay_rate2=0.9999)))

        # 同上、NAG
        # ★Avg.loss=0.661, Avg.accuracy=0.733, Max.accuracy=0.978, Argmax=75 | Avg.test_loss=0.770, Avg.test_accuracy=0.696, Max.test_accuracy=0.935, Argmax=87
        # ※白黒反転時：★Avg.loss=0.378, Avg.accuracy=0.847, Max.accuracy=0.999, Argmax=69 | Avg.test_loss=0.559, Avg.test_accuracy=0.812, Max.test_accuracy=0.965, Argmax=66
        # self.nn = DNN(input_size=784,
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
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Tanh(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=Momentum(learning_rate=0.01, decay_rate=0.9)))

        # AdaGrad
        # ★Avg.loss=0.017, Avg.accuracy=0.998, Max.accuracy=1.000, Argmax=17 | Avg.test_loss=0.154, Avg.test_accuracy=0.959, Max.test_accuracy=0.975, Argmax=60
        # ※白黒反転時：★Avg.loss=0.002, Avg.accuracy=1.000, Max.accuracy=1.000, Argmax=4 | Avg.test_loss=0.167, Avg.test_accuracy=0.960, Max.test_accuracy=0.960, Argmax=0
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Tanh(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=AdaGrad(learning_rate=0.01)))

        # AdaDelta
        # ★Avg.loss=0.014, Avg.accuracy=0.995, Max.accuracy=1.000, Argmax=14 | Avg.test_loss=0.208, Avg.test_accuracy=0.958, Max.test_accuracy=0.975, Argmax=87
        # ※白黒反転時：★Avg.loss=0.001, Avg.accuracy=1.000, Max.accuracy=1.000, Argmax=5 | Avg.test_loss=0.233, Avg.test_accuracy=0.970, Max.test_accuracy=0.970, Argmax=6
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Tanh(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=AdaDelta(decay_rate=0.9)))

        # RMSProp
        # ★Avg.loss=12.896, Avg.accuracy=0.200, Max.accuracy=0.205, Argmax=3 | Avg.test_loss=12.890, Avg.test_accuracy=0.200, Max.test_accuracy=0.220, Argmax=7
        # ※白黒反転時：★Avg.loss=12.903, Avg.accuracy=0.199, Max.accuracy=0.205, Argmax=3 | Avg.test_loss=12.861, Avg.test_accuracy=0.202, Max.test_accuracy=0.220, Argmax=5
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Tanh(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=RMSProp(learning_rate=0.01)))

        # Adam
        # ★Avg.loss=0.114, Avg.accuracy=0.967, Max.accuracy=0.994, Argmax=21 | Avg.test_loss=0.304, Avg.test_accuracy=0.925, Max.test_accuracy=0.975, Argmax=75
        # ※白黒反転時：★Avg.loss=0.004, Avg.accuracy=0.998, Max.accuracy=1.000, Argmax=12 | Avg.test_loss=0.299, Avg.test_accuracy=0.959, Max.test_accuracy=0.965, Argmax=3
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Tanh(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=Adam(learning_rate=0.01, decay_rate1=0.9, decay_rate2=0.9999)))

        # NAG
        # ★Avg.loss=0.024, Avg.accuracy=0.996, Max.accuracy=1.000, Argmax=9 | Avg.test_loss=0.185, Avg.test_accuracy=0.965, Max.test_accuracy=0.975, Argmax=12
        # ※白黒反転時：★Avg.loss=0.021, Avg.accuracy=0.996, Max.accuracy=1.000, Argmax=8 | Avg.test_loss=0.209, Avg.test_accuracy=0.968, Max.test_accuracy=0.980, Argmax=35
        # self.nn = DNN(input_size=784,
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
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=Momentum(learning_rate=0.01, decay_rate=0.9)))

        # AdaGrad
        # 　⇒Momentumよりもさらに早く最大正解率100%に到達した。
        # ★Avg.loss=0.103, Avg.accuracy=0.964, Max.accuracy=1.000, Argmax=75 | Avg.test_loss=0.232, Avg.test_accuracy=0.924, Max.test_accuracy=0.960, Argmax=23
        # ※白黒反転時：★Avg.loss=0.001, Avg.accuracy=1.000, Max.accuracy=1.000, Argmax=4 | Avg.test_loss=0.120, Avg.test_accuracy=0.965, Max.test_accuracy=0.970, Argmax=0
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=AdaGrad(learning_rate=0.01)))

        # 同上、AdaDelta　⇒AdaGradと同程度に早く最大正解率100%に到達した。
        # ★Avg.loss=0.021, Avg.accuracy=0.991, Max.accuracy=1.000, Argmax=17 | Avg.test_loss=0.244, Avg.test_accuracy=0.958, Max.test_accuracy=0.970, Argmax=25
        # ※白黒反転時：★Avg.loss=0.001, Avg.accuracy=1.000, Max.accuracy=1.000, Argmax=4 | Avg.test_loss=0.179, Avg.test_accuracy=0.974, Max.test_accuracy=0.975, Argmax=8
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=AdaDelta(decay_rate=0.9)))

        # RMSProp
        # 　⇒20％程度で、学習が進まない。
        # ★Avg.loss=12.889, Avg.accuracy=0.200, Max.accuracy=0.205, Argmax=3 | Avg.test_loss=12.915, Avg.test_accuracy=0.199, Max.test_accuracy=0.220, Argmax=0
        # ※白黒反転時：★Avg.loss=12.886, Avg.accuracy=0.201, Max.accuracy=0.205, Argmax=5 | Avg.test_loss=12.928, Avg.test_accuracy=0.198, Max.test_accuracy=0.220, Argmax=3
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=RMSProp(learning_rate=0.01)))

        # 同上、Adam　⇒Momentumと同程度に早く最大正解率に到達した。最大正解率への到達エポック回数はAdaGradの方が良かった。
        # ★Avg.loss=1.609, Avg.accuracy=0.204, Max.accuracy=0.205, Argmax=0 | Avg.test_loss=1.611, Avg.test_accuracy=0.184, Max.test_accuracy=0.220, Argmax=74
        # ※白黒反転時：★Avg.loss=0.005, Avg.accuracy=0.999, Max.accuracy=1.000, Argmax=7 | Avg.test_loss=0.237, Avg.test_accuracy=0.966, Max.test_accuracy=0.975, Argmax=27
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               learner=MiniBatch(epoch_num=100, mini_batch_size=10, optimizer=Adam(learning_rate=0.01, decay_rate1=0.9, decay_rate2=0.9999)))

        # 同上、Nesterov　⇒Momentum、Adamと同程度に早く最大正解率に到達した。最大正解率への到達エポック回数はAdaGradの方が良かった。
        # ★Avg.loss=0.039, Avg.accuracy=0.986, Max.accuracy=1.000, Argmax=15 | Avg.test_loss=0.333, Avg.test_accuracy=0.944, Max.test_accuracy=0.970, Argmax=16
        # ※白黒反転時：★Avg.loss=0.039, Avg.accuracy=0.986, Max.accuracy=1.000, Argmax=8 | Avg.test_loss=0.186, Avg.test_accuracy=0.955, Max.test_accuracy=0.970, Argmax=26
        # self.nn = DNN(input_size=784,
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
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[100, 5],
        #               hidden_actfunc=Sigmoid(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               init_weight_stddev=0.01,
        #               learner=KFoldCrossValidation(kfold_num=100, optimizer=SGD(learning_rate=0.01)))

        # 3層、Sigmoid、100分割　⇒3層にするとかえって正解率が落ちた。しかも正解率が毎回ぶれる。
        # ★kfold_num=100: Avg.Loss=1.613, Avg.Accuracy=0.202, Max.Accuracy=0.500, Argmax=12
        # self.nn = DNN(input_size=784,
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
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Tanh(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               init_weight_stddev=0.01,
        #               learner=KFoldCrossValidation(kfold_num=100, optimizer=SGD(learning_rate=0.01)))

        # Momentum ⇒SGDと比べて、かなり早く学習が進み、第3エポックで最大正解率に到達した。
        # ★kfold_num=100: Avg.Loss=0.025, Avg.Accuracy=0.996, Max.Accuracy=1.000, Argmax=2
        # ※白黒反転時：★kfold_num=100: Avg.Loss=0.017, Avg.Accuracy=0.998, Max.Accuracy=1.000, Argmax=2
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Tanh(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               init_weight_stddev=0.01,
        #               learner=KFoldCrossValidation(kfold_num=100, optimizer=Momentum(learning_rate=0.01, decay_rate=0.9)))

        # AdaGrad ⇒すでに第1エポックで最大正解率に達した。最も早い。★★★
        # ★kfold_num=100: Avg.Loss=0.031, Avg.Accuracy=0.988, Max.Accuracy=1.000, Argmax=2
        # ※白黒反転時：★kfold_num=100: Avg.Loss=0.003, Avg.Accuracy=1.000, Max.Accuracy=1.000, Argmax=0
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Tanh(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               init_weight_stddev=0.01,
        #               learner=KFoldCrossValidation(kfold_num=100, optimizer=AdaGrad(learning_rate=0.01)))

        # AdaDelta ⇒AdaGradと同様、すでに第1エポックで最大正解率に達した。最も早い。★★★
        # ★kfold_num=100: Avg.Loss=0.012, Avg.Accuracy=0.995, Max.Accuracy=1.000, Argmax=2
        # ※白黒反転時：★kfold_num=100: Avg.Loss=0.001, Avg.Accuracy=1.000, Max.Accuracy=1.000, Argmax=0
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Tanh(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               init_weight_stddev=0.01,
        #               learner=KFoldCrossValidation(kfold_num=100, optimizer=AdaDelta(decay_rate=0.9)))

        # RMSProp ⇒最大でも50%。
        # ★kfold_num=100: Avg.Loss=13.076, Avg.Accuracy=0.188, Max.Accuracy=0.600, Argmax=70
        # ※白黒反転時：★kfold_num=100: Avg.Loss=13.233, Avg.Accuracy=0.179, Max.Accuracy=0.500, Argmax=18
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Tanh(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               init_weight_stddev=0.01,
        #               learner=KFoldCrossValidation(kfold_num=100, optimizer=RMSProp(learning_rate=0.01)))

        # Adam  ⇒AdaGrad、AdaDeltaと同様、すでに第1エポックで最大正解率に達した。最も早い。★★★
        # ★kfold_num=100: Avg.Loss=0.146, Avg.Accuracy=0.960, Max.Accuracy=1.000, Argmax=2
        # ※白黒反転時：★kfold_num=100: Avg.Loss=0.009, Avg.Accuracy=0.995, Max.Accuracy=1.000, Argmax=0
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=Tanh(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               init_weight_stddev=0.01,
        #               learner=KFoldCrossValidation(kfold_num=100, optimizer=Adam(learning_rate=0.01, decay_rate1=0.9, decay_rate2=0.9999)))

        # NAG ⇒Momentumと同様、第3エポックで最大正解率に到達した。
        # ★kfold_num=100: Avg.Loss=0.023, Avg.Accuracy=0.996, Max.Accuracy=1.000, Argmax=2
        # ※白黒反転時：★kfold_num=100: Avg.Loss=0.016, Avg.Accuracy=0.997, Max.Accuracy=1.000, Argmax=2
        # self.nn = DNN(input_size=784,
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
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               init_weight_stddev=0.01,
        #               learner=KFoldCrossValidation(kfold_num=100, optimizer=SGD(learning_rate=0.01)))

        # Momentum ⇒第3エポックで最大正解率に達した。tanhでのSGDと比べてかなり早く学習が進んだ。
        # ★kfold_num=100: Avg.Loss=0.047, Avg.Accuracy=0.986, Max.Accuracy=1.000, Argmax=2
        # ※白黒反転時：★kfold_num=100: Avg.Loss=0.030, Avg.Accuracy=0.986, Max.Accuracy=1.000, Argmax=2
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               init_weight_stddev=0.01,
        #               learner=KFoldCrossValidation(kfold_num=100, optimizer=Momentum(learning_rate=0.01, decay_rate=0.9)))

        # AdaGrad ⇒tanhでの場合と同様。すでに第1エポックで最大正解率に達した。最も早い。★★★
        # ★kfold_num=100: Avg.Loss=0.046, Avg.Accuracy=0.986, Max.Accuracy=1.000, Argmax=2
        # ※白黒反転時：★kfold_num=100: Avg.Loss=0.005, Avg.Accuracy=0.999, Max.Accuracy=1.000, Argmax=0
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               init_weight_stddev=0.01,
        #               learner=KFoldCrossValidation(kfold_num=100, optimizer=AdaGrad(learning_rate=0.01)))

        # AdaDelta ⇒AdaGradと同程度で、第3エポックで最大正解率に到達した。
        # ★kfold_num=100: Avg.Loss=0.031, Avg.Accuracy=0.990, Max.Accuracy=1.000, Argmax=2
        # ※白黒反転時：★kfold_num=100: Avg.Loss=0.002, Avg.Accuracy=0.998, Max.Accuracy=1.000, Argmax=2
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               init_weight_stddev=0.01,
        #               learner=KFoldCrossValidation(kfold_num=100, optimizer=AdaDelta(decay_rate=0.9)))

        # RMSProp ⇒最大でも50%。ぶれが大きい。
        # ★kfold_num=100: Avg.Loss=12.943, Avg.Accuracy=0.197, Max.Accuracy=0.600, Argmax=70
        # ※白黒反転時：★kfold_num=100: Avg.Loss=12.346, Avg.Accuracy=0.234, Max.Accuracy=0.500, Argmax=10
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               init_weight_stddev=0.01,
        #               learner=KFoldCrossValidation(kfold_num=100, optimizer=RMSProp(learning_rate=0.01)))

        # Adam ⇒AdaDeltaと同様、第2エポックで最大正解率に到達した。
        # ★kfold_num=100: Avg.Loss=1.613, Avg.Accuracy=0.109, Max.Accuracy=0.400, Argmax=7
        # ※白黒反転時：★kfold_num=100: Avg.Loss=0.001, Avg.Accuracy=0.999, Max.Accuracy=1.000, Argmax=1
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               init_weight_stddev=0.01,
        #               learner=KFoldCrossValidation(kfold_num=100, optimizer=Adam(learning_rate=0.01, decay_rate1=0.9, decay_rate2=0.9999)))

        # NAG ⇒AdaDeltaと同様、第3エポックで最大正解率に到達した。
        # ★kfold_num=100: Avg.Loss=0.031, Avg.Accuracy=0.987, Max.Accuracy=1.000, Argmax=2
        # ※白黒反転時：★kfold_num=100: Avg.Loss=0.030, Avg.Accuracy=0.986, Max.Accuracy=1.000, Argmax=2
        # self.nn = DNN(input_size=784,
        #               layer_size_list=[100, 100, 5],
        #               hidden_actfunc=ReLU(),
        #               output_actfunc=SoftmaxWithLoss(),
        #               loss_func=CrossEntropyError(),
        #               init_weight_stddev=0.01,
        #               learner=KFoldCrossValidation(kfold_num=100, optimizer=NAG(learning_rate=0.01, decay_rate=0.9)))

        # ★★★では、Day4までの講義内容を実装した中で最も性能が良かったモデルについて、重みの初期値を変えてみたらどうなるか？
        # 5層に増やした／init_weight_change=Trueを指定。
        # （元の実験）★kfold_num = 100: Avg.Loss = 0.001, Avg.Accuracy = 1.000, Max.Accuracy = 1.000, Argmax = 0
        # （初期値変更版）★kfold_num=100: Avg.Loss=0.000, Avg.Accuracy=1.000, Max.Accuracy=1.000, Argmax=0
        self.nn = DNN(input_size=784,
                      layer_size_list=[100, 100, 100, 100, 5],
                      hidden_actfunc=Tanh(),
                      output_actfunc=SoftmaxWithLoss(),
                      loss_func=CrossEntropyError(),
                      init_weight_stddev=0.01,
                      learner=KFoldCrossValidation(kfold_num=100, optimizer=AdaDelta(decay_rate=0.9)),
                      init_weight_change=True
                      )

    def fit(self, train_data, train_label):
        self.nn.fit(train_data=train_data, train_label=train_label)
        self.nn.save()

    def predict(self, model_path, test_data, test_label):
        y, loss, accuracy = self.nn.predict_with_learned_model(model_path, test_data, test_label)
        return loss, accuracy

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
