import numpy as np
from DNN import DNN


# 2018/07/21提出時点での最適組み合わせ：
# ・レイヤー数：3層
# ・隠れ層の活性化関数：ReLU
# ・出力層の活性化関数：ソフトマックス関数
# ・学習ロジック：k分割交差検証（1個抜き交差検証）
#
# 以下固定
# ・損失関数：クロスエントロピー誤差関数
# ・初期重み標準偏差：0.01
# ・学習率：0.01
class NNExecutor:
    def __init__(self):
        # TODO  DNNとCNNで切り替える実装にすること。コンストラクタの引数で渡してくるとか。
        # ↓2層NN。エポック数を100回、ミニバッチサイズを20にすると97％を超過した。
        # self.nn = DNN(input_size=784, layer_size_list=[100, 5])

        # ↓3層NN。学習が進まない。正解率20％。
        # ⇒ReLUに切り替えると97％を超過した。
        # ★fit start
        # ★epoch[0]終了 loss=1.609, accuracy=0.209
        # ・・・
        # ★epoch[88]終了 loss=0.018, accuracy=0.998
        # ★epoch[89]終了 loss=0.017, accuracy=0.998
        # ★epoch[90]終了 loss=0.016, accuracy=0.998
        # ★epoch[91]終了 loss=0.016, accuracy=0.998
        # ★epoch[92]終了 loss=0.015, accuracy=0.998
        # ★epoch[93]終了 loss=0.015, accuracy=0.998
        # ★epoch[94]終了 loss=0.014, accuracy=0.998
        # ★epoch[95]終了 loss=0.014, accuracy=0.998
        # ★epoch[96]終了 loss=0.014, accuracy=0.998
        # ★epoch[97]終了 loss=0.013, accuracy=0.998
        # ★epoch[98]終了 loss=0.013, accuracy=0.998
        # ★epoch[99]終了 loss=0.012, accuracy=0.998
        # ★Avg. loss=0.540, Avg. accuracy=0.836, argmax(accuracy)=88
        self.nn = DNN(input_size=784, layer_size_list=[100, 100, 5])

        # ↓4層NN。学習が進まない。正解率20％。
        # ⇒ReLUに切り替えて実施したが、かえって3層のときよりも正解率が落ちた（33％などにかなり落ちた）。
        # ★fit start
        # ★epoch[0]終了 loss=1.609, accuracy=0.284
        # ・・・
        # ★epoch[97]終了 loss=1.609, accuracy=0.825
        # ★epoch[98]終了 loss=1.609, accuracy=0.738
        # ★epoch[99]終了 loss=1.609, accuracy=0.741
        # ★Avg. loss=1.609, Avg. accuracy=0.332, argmax(accuracy)=97
        # self.nn = DNN(input_size=784, layer_size_list=[100, 100, 100, 5])

        # ↓5層NN。かえって学習が進まなくなった。20％。
        # ★fit start
        # ★epoch[0]終了 loss=1.609, accuracy=0.200
        # ★epoch[1]終了 loss=1.609, accuracy=0.200
        # ・・・
        # ★epoch[97]終了 loss=1.609, accuracy=0.200
        # ★epoch[98]終了 loss=1.609, accuracy=0.200
        # ★epoch[99]終了 loss=1.609, accuracy=0.200
        # ★Avg. loss=1.609, Avg. accuracy=0.200, argmax(accuracy)=0
        # self.nn = DNN(input_size=784, layer_size_list=[100, 100, 100, 100, 5])

    def fit(self, train_data, train_label):
        # 2層でもエポック数が10では20％~30％程度。
        # self.nn.fit(train_data=train_data, train_label=train_label, init_weight_stddev=0.01, epoch_num=10, mini_batch_size=100, learning_rate=0.01)

        # 2層でエポック数100にすると学習が進んだ。
        # ミニバッチサイズが大きすぎるかもしれない。
        # TODO 初期重み標準偏差、学習率固定としているが、DNN生成時に指定できる方がよいのでは？
        # TODO ハイパーパラメータは全部DNNで指定する方向で検討する。
        self.nn.fit(train_data=train_data, train_label=train_label, init_weight_stddev=0.01, epoch_num=100, mini_batch_size=20, learning_rate=0.01)

        # 学習済みモデルの保存。
        self.nn.save()

    def predict(self, model_path, test_data, test_label):
        y, loss, accuracy = self.nn.predict_with_learned_model(model_path, test_data, test_label)
        return loss, accuracy
