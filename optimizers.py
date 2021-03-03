import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, nn):
        for i, layer in enumerate(nn.layers):
            if (layer.__class__.__name__ == "ReLU") | (layer.__class__.__name__ == "MaxPool") | (layer.__class__.__name__ == "SoftmaxWithLoss"):
                continue
            # CNN対応版。
            layer.W -= self.learning_rate * layer.dLdW
            layer.B -= self.learning_rate * layer.dLdB
            # TODO 以下DNN版なので不要。
            # layer.affine.W -= self.learning_rate * layer.affine.dLdW
            # layer.affine.B -= self.learning_rate * layer.affine.dLdB


# TODO CNN対応版にすること。
class Momentum:
    def __init__(self, learning_rate=0.01, decay_rate=0.9):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.v_W = None
        self.v_B = None

    def update(self, nn):
        if self.v_W is None:  # このときself.v_BもNoneであること。
            self.v_W = []
            self.v_B = []
            for i, layer in enumerate(nn.layers):
                self.v_W.append(np.zeros_like(layer.affine.W))
                self.v_B.append(np.zeros_like(layer.affine.B))

        for i, layer in enumerate(nn.layers):
            self.v_W[i] = self.decay_rate * self.v_W[i] - self.learning_rate * layer.affine.dLdW
            self.v_B[i] = self.decay_rate * self.v_B[i] - self.learning_rate * layer.affine.dLdB
            layer.affine.W += self.v_W[i]
            layer.affine.B += self.v_B[i]

class AdaGrad:
    def __init__(self, learning_rate=0.01, decay_rate=0.9):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.h_W = None
        self.h_B = None
        self.eps = 1.0e-7

    def update(self, nn):
        if self.h_W is None:  # このときself.h_BもNoneであること。
            self.h_W = []
            self.h_B = []
            for i, layer in enumerate(nn.layers):
                self.h_W.append(np.zeros_like(layer.affine.W))
                self.h_B.append(np.zeros_like(layer.affine.B))

        for i, layer in enumerate(nn.layers):
            self.h_W[i] += layer.affine.dLdW * layer.affine.dLdW
            self.h_B[i] += layer.affine.dLdB * layer.affine.dLdB
            layer.affine.W -= self.learning_rate * layer.affine.dLdW / np.sqrt(self.h_W[i] + self.eps)
            layer.affine.B -= self.learning_rate * layer.affine.dLdB / np.sqrt(self.h_B[i] + self.eps)

class AdaDelta:
    def __init__(self, decay_rate=0.9):
        self.decay_rate = decay_rate
        self.Exp_gt2_W = None
        self.Exp_gt2_B = None
        self.Exp_dW2 = None
        self.Exp_dB2 = None
        self.eps = 1.0e-6  # 論文：arXiv:1212.5701v1での最適値を採用。

    def update(self, nn):
        if self.Exp_gt2_W is None:  # このときself.Exp_gt2_B等の更新変数もNoneであること。
            self.Exp_gt2_W = []
            self.Exp_gt2_B = []
            self.Exp_dx2_W = []
            self.Exp_dx2_B = []
            for i, layer in enumerate(nn.layers):
                self.Exp_gt2_W.append(np.zeros_like(layer.affine.W))
                self.Exp_gt2_B.append(np.zeros_like(layer.affine.B))
                self.Exp_dx2_W.append(np.zeros_like(layer.affine.dLdW))
                self.Exp_dx2_B.append(np.zeros_like(layer.affine.dLdB))

        for i, layer in enumerate(nn.layers):
            # Accumulate Gradient
            self.Exp_gt2_W[i] = self.decay_rate * self.Exp_gt2_W[i] + (1.0 - self.decay_rate) * layer.affine.dLdW * layer.affine.dLdW
            self.Exp_gt2_B[i] = self.decay_rate * self.Exp_gt2_B[i] + (1.0 - self.decay_rate) * layer.affine.dLdB * layer.affine.dLdB

            # Compute Update
            dW = (-1.0) * np.sqrt(self.Exp_dx2_W[i] + self.eps) / np.sqrt(self.Exp_gt2_W[i] + self.eps) * layer.affine.dLdW
            dB = (-1.0) * np.sqrt(self.Exp_dx2_B[i] + self.eps) / np.sqrt(self.Exp_gt2_B[i] + self.eps) * layer.affine.dLdB

            # Accumulate Updates
            self.Exp_dx2_W[i] = self.decay_rate * self.Exp_dx2_W[i] + (1.0 - self.decay_rate) * dW * dW
            self.Exp_dx2_B[i] = self.decay_rate * self.Exp_dx2_B[i] + (1.0 - self.decay_rate) * dB * dB

            # Apply Updates
            layer.affine.W += dW
            layer.affine.B += dB

# # 以下正式版
# class AdaDelta:
#     def __init__(self, decay_rate=0.95):
#         self.h_W = None
#         self.h_B = None
#         self.r_W = None
#         self.decay_rate = decay_rate
#         self.eps = 1e-6  # 論文：arXiv:1212.5701v1での最適値を採用。
#
#     def update(self, nn):
#         if self.h_W is None:
#             self.h_W = []
#             self.h_B = []
#             self.r_W = []
#             self.r_B = []
#             for i, layer in enumerate(nn.layers):
#                 self.h_W.append(np.zeros_like(layer.affine.W))
#                 self.h_B.append(np.zeros_like(layer.affine.B))
#                 self.r_W.append(np.zeros_like(layer.affine.W))
#                 self.r_B.append(np.zeros_like(layer.affine.B))
#
#         for i, layer in enumerate(nn.layers):
#             # 1ステップ前における更新量の移動平均のルートを求める
#             rms_param_W = np.sqrt(self.r_W[i] + self.eps)
#             rms_param_B = np.sqrt(self.r_B[i] + self.eps)
#
#             # 勾配の2乗の移動平均を求める
#             self.h_W[i] = self.decay_rate * self.h_W[i] + (1 - self.decay_rate) * layer.affine.dLdW * layer.affine.dLdW
#             self.h_B[i] = self.decay_rate * self.h_B[i] + (1 - self.decay_rate) * layer.affine.dLdB * layer.affine.dLdB
#
#             # 勾配の2乗の移動平均のルートを求める
#             rms_grad_W = np.sqrt(self.h_W[i] + self.eps)
#             rms_grad_B = np.sqrt(self.h_B[i] + self.eps)
#
#             # 更新量の算出
#             dp_W = - rms_param_W / rms_grad_W * layer.affine.dLdW
#             dp_B = - rms_param_B / rms_grad_B * layer.affine.dLdB
#
#             # 重みの更新
#             layer.affine.W += dp_W
#             layer.affine.B += dp_B
#
#             # 次ステップのために、更新量の移動平均を求める
#             self.r_W[i] = self.decay_rate * self.r_W[i] + (1 - self.decay_rate) * dp_W * dp_W
#             self.r_B[i] = self.decay_rate * self.r_B[i] + (1 - self.decay_rate) * dp_B * dp_B


class RMSProp:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.h_W = None
        self.h_B = None
        self.eps = 1.0e-6

    def update(self, nn):
        if self.h_W is None:  # このときself.h_BもNoneであること。
            self.h_W = []
            self.h_B = []
            for i, layer in enumerate(nn.layers):
                self.h_W.append(np.zeros_like(layer.affine.W))
                self.h_B.append(np.zeros_like(layer.affine.B))

        for i, layer in enumerate(nn.layers):
            self.h_W[i] = self.h_W[i] + layer.affine.dLdW * layer.affine.dLdW
            layer.affine.W -= self.learning_rate / (np.sqrt(self.h_W[i]) + self.eps) * layer.affine.dLdW
            layer.affine.B -= self.learning_rate / (np.sqrt(self.h_B[i]) + self.eps) * layer.affine.dLdB

class Adam:
    def __init__(self, learning_rate=0.01, decay_rate1=0.9, decay_rate2=0.9999):
        self.learning_rate = learning_rate
        self.decay_rate1 = decay_rate1
        self.decay_rate2 = decay_rate2
        self.m_W = None
        self.m_B = None
        self.v_W = None
        self.v_B = None
        self.eps = 1.0e-6
        self.iter = 0

    def update(self, nn):
        self.iter += 1

        if self.m_W is None:  # このときself.m_BもNoneであること。
            self.m_W = []
            self.m_B = []
            self.v_W = []
            self.v_B = []
            for i, layer in enumerate(nn.layers):
                self.m_W.append(np.zeros_like(layer.affine.W))
                self.m_B.append(np.zeros_like(layer.affine.B))
                self.v_W.append(np.zeros_like(layer.affine.W))
                self.v_B.append(np.zeros_like(layer.affine.B))

        for i, layer in enumerate(nn.layers):
            self.m_W[i] = self.decay_rate1 * self.m_W[i] + (1.0 - self.decay_rate1) * layer.affine.dLdW
            self.m_B[i] = self.decay_rate1 * self.m_B[i] + (1.0 - self.decay_rate1) * layer.affine.dLdB

            self.v_W[i] = self.decay_rate2 * self.v_W[i] + (1.0 - self.decay_rate2) * layer.affine.dLdW * layer.affine.dLdW
            self.v_B[i] = self.decay_rate2 * self.v_B[i] + (1.0 - self.decay_rate2) * layer.affine.dLdB * layer.affine.dLdB

            mt_hat_W = self.m_W[i] / (1.0 - self.decay_rate1**self.iter)
            vt_hat_W = self.v_W[i] / (1.0 - self.decay_rate2**self.iter)

            mt_hat_B = self.m_B[i] / (1.0 - self.decay_rate1**self.iter)
            vt_hat_B = self.v_B[i] / (1.0 - self.decay_rate2**self.iter)

            layer.affine.W -= self.learning_rate / (np.sqrt(vt_hat_W) + self.eps) * mt_hat_W
            layer.affine.B -= self.learning_rate / (np.sqrt(vt_hat_B) + self.eps) * mt_hat_B

# Nesterov Accelerated Gradient
class NAG:
    def __init__(self, learning_rate=0.01, decay_rate=0.9):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.v_W = None
        self.v_B = None

    def update(self, nn):
        if self.v_W is None:  # このときself.v_BもNoneであること。
            self.v_W = []
            self.v_B = []
            for i, layer in enumerate(nn.layers):
                self.v_W.append(np.zeros_like(layer.affine.W))
                self.v_B.append(np.zeros_like(layer.affine.B))

        for i, layer in enumerate(nn.layers):
            layer.affine.W += self.decay_rate**2 * self.v_W[i] - (1 + self.decay_rate) * self.learning_rate * layer.affine.dLdW
            layer.affine.B += self.decay_rate**2 * self.v_B[i] - (1 + self.decay_rate) * self.learning_rate * layer.affine.dLdB
            self.v_W[i] = self.decay_rate * self.v_W[i] - self.learning_rate * layer.affine.dLdW
            self.v_B[i] = self.decay_rate * self.v_B[i] - self.learning_rate * layer.affine.dLdB


