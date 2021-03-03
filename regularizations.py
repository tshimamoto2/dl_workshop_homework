import numpy as np

##############################
# 正則化
##############################
class L2:
    def __init__(self, lmda=0.0):
        self.lmda = lmda

    def update_dLdW(self, layers):
        for layer in layers:
            layer.affine.dLdW += self.lmda * layer.affine.W

    def regular(self, layers):
        regular = 0
        for layer in layers:
            regular += 0.5 * self.lmda * np.sum(layer.affine.W ** 2)
        return regular

# TODO 実装して使用してみたが学習が進まない。何らかのバグがありそうなので一旦削除。
# class MaxNorm:
#     def __init__(self, lmda=0.0):
#         self.lmda = lmda
#
#     def update_dLdW(self, layers):
#         for layer in layers:
#             # |W|のWでの微分は、W>=0なら1、W<0なら-1なので、numpyのsignを使った。
#             layer.affine.dLdW += self.lmda * np.sign(layer.affine.W)
#
#     def regular(self, layers):
#         regular = 0
#         for layer in layers:
#             # L∞ノルム（最大値ノルム、max normとも）は、ベクトル（or行列）の中で絶対値が最大の要素の絶対値。
#             regular += self.lmda * np.max(np.abs(layer.affine.W))
#         return regular
