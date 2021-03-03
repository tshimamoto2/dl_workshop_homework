import numpy as np
from layers import NormalWeight

##################################################
# CNNを実装するにあたり、下記前提を置いた。
# 【前提】
#     ・入力データxは4階のテンソルであること。
#     ・入力データxの第1次元はバッチ軸であること。よって、x.shape[0]は一度に処理されるミニバッチサイズである。
#     ・入力データxの第2次元はチャネル軸であること。よって、x.shape[1]は一度に処理されるチャネル数である。
#         - ニューラルネットワークの最初の入力データの場合は、画像のチャネルそのものを指す。すなわち、グレースケール形式なら1、RGB形式なら3である。
#         - また、ニューラルネットワーク途中で本Convレイヤーが呼び出された場合は、前レイヤーでのフィルター数（変数ではFN）となる。
#     ・入力データxの第3次元は画像の高さ軸である。よって、x.shape[2]は画像の高さ[pixel]である。
#     ・入力データxの第4次元は画像の幅軸である。よって、x.shape[3]は画像の幅[pixel]である。
#     ・畳み込みフィルターWは4階のテンソルであること。
#     ・畳み込みフィルターWの第1次元はフィルター軸であること。よって、x.shape[0]は一度に処理されるフィルター個数である。
#     ・畳み込みフィルターWの第2次元はチャネル軸であること。よって、x.shape[1]は一度に処理されるフィルター個数である。
#     ・畳み込みフィルターWの第3次元は、入力データxの画像の高さに対応する、フィルターの高さ軸である。よって、x.shape[2]はフィルター高さ[pixel]である。
#     ・畳み込みフィルターWの第4次元は、入力データxの画像の幅に対応する、フィルターの幅軸である。よって、x.shape[3]はフィルター幅[pixel]である。
#     ・入力、フィルタ、出力ともに正方行列を前提とする。畳み込みのstrideは高さ軸方向、幅軸方向とも同じ値とすること。
#          - 入力データxの画像データは正方行列とする。よってx.shape[2]==x.shape[3]であること。ただし変数としてはIH、IWなどと分けておく。
#          - フィルタについても高さと幅が等しい正方行列であること。ただし変数としてはFH、FWなどと分けておく。
#          - 出力の画像サイズも正方行列になること。ただし変数としてはOH、OWなどと分けておく。
#     ・ハイパーパラメータとして指定するパディング値は、画像の高さ方向と幅方向の両方に同時に適用されること（高さ方向にも幅方向にも同数だけパディングされること）
##################################################
# 畳み込み層
##################################################
# 畳み込み層は、実装の難易度を下げるため、以下の前提を置いた。
class Conv:
    def __init__(self, FN=1, FH=2, FW=2, padding=0, stride=1, weight=NormalWeight(0.01)):
        self.FN = FN  # フィルター数。畳み込みの出力のチャンネル数になる。
        self.FH = FH  # フィルター高さ[ピクセル]
        self.FW = FW  # フィルター幅[ピクセル]
        self.padding = padding  # パディングサイズ[ピクセル]
        self.stride = stride  # フィルター（ウィンドウ）の1回の走査あたりの移動サイズ[ピクセル]
        self.weight = weight  # 重みの初期値オブジェクト
        self.OH = None
        self.OW = None

        self.x = None
        self.x2d = None
        self.f2d = None

        self.W = None  # フィルターの重み
        self.B = None  # フィルターのバイアス
        self.dLdW = None  # 損失関数の、本畳み込み層の重みによる偏微分値
        self.dLdB = None  # 損失関数の、本畳み込み層のバイアスによる偏微分値

    def has_weight(self):
        return True

    def forward(self, x, t, is_learning=False):
        # 誤差逆伝播のために保持。
        self.x = x

        # 入力データの各次元のサイズを取得。
        batch_size, channel_size, IH, IW = x.shape
        OH = out_size(IH, self.FH, self.padding, self.stride)
        OW = out_size(IW, self.FW, self.padding, self.stride)

        # フィルターの重みとバイアスを生成。
        if self.W is None:
            np.random.seed(2000)
            self.W = np.random.randn(self.FN, channel_size, self.FH, self.FW) * self.weight.get_stddev()
            self.B = np.zeros((self.FN, OH, OW))

        # 入力データとフィルターとの畳み込み演算。2次元化したデータは逆伝播のために保持。
        self.x2d = im2col_HP(x, self.FH, self.FW, self.padding, self.stride)
        self.f2d = self.W.reshape(self.FN, -1)
        out = np.dot(self.x2d, self.f2d.T)

        # 4階テンソルに戻す。
        out = out.reshape(batch_size, OH, OW, -1).transpose(0, 3, 1, 2)
        out += self.B
        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape

        # 出力を2次元化。ただしim2colの仕様に合わせて、チャネルが列方向に並ぶように変形。
        dout2d = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        # dLdB
        self.dLdB = np.sum(dout, axis=0)

        # dLdW
        dLdW2d = np.dot(self.x2d.T, dout2d)
        self.dLdW = dLdW2d.transpose(1, 0).reshape(FN, C, FH, FW)

        # dLdX
        dx2d = np.dot(dout2d, self.f2d)
        dLdX = col2im_HP(dx2d, self.x.shape, FH, FW, self.padding, self.stride)

        return dLdX

class MaxPool:
    def __init__(self, FH=1, FW=1, padding=0, stride=1):
        self.B = None
        self.C = None
        self.IH = None
        self.IW = None
        self.FH = FH
        self.FW = FW
        self.padding = padding
        self.stride = stride
        self.OH = None
        self.OW = None

        self.x = None
        self.x2d = None
        self.out = None
        self.arg_max = None

    def has_weight(self):
        return False

    def forward(self, x, t, is_learning_False):
        self.x = x
        self.B, self.C, self.IH, self.IW = self.x.shape
        self.OH = out_size(self.IH, self.FH, self.padding, self.stride)
        self.OW = out_size(self.IW, self.FW, self.padding, self.stride)

        # 入力値（4次元）の2次元化。
        x2d = im2col(x, self.FH, self.FW, self.padding, self.stride)

        # 列の幅を指定して、チャネル軸も縦1列に並べ替える。
        # よって、第1次元：バッチ軸（B個）、第2次元：高さ軸（OH個）ー幅軸（OW個）ーチャネル軸（C個）となる。
        x2d = x2d.reshape(-1, self.FH * self.FW)

        # 逆伝播のために、入力xをフィルターで走査してみて「切り取られた区画（ウィンドウ）の中のどの位置が」最大値だったのかを保持しておく。
        # （例）入力xがIHxIW=3x3で、フィルターサイズがPHxPW=2x2、padding=0、stride=1の場合、出力Oは2x2となる。
        #   例えば第3ウィンドウには左上から右下の順に[x21,x23,x31,x32]が含まれる。
        #   x2d、out、argmaxの関係例を以下に示す。
        #     第1ウィンドウx2d[0]（出力out11に対応）: [x11, x12, x21, x22] →最大値がx11だとすると、argmax=0
        #     第2ウィンドウx2d[1]（出力out12に対応）: [x12, x13, x22, x23] →最大値がx23だとすると、argmax=3
        #     第3ウィンドウx2d[2]（出力out21に対応）: [x21, x22, x31, x32] →最大値がx22だとすると、argmax=1
        #     第4ウィンドウx2d[3]（出力out22に対応）: [x22, x23, x32, x33] →最大値がx32だとすると、argmax=2
        self.arg_max = np.argmax(x2d, axis=1)

        # 第2次元（次元のインデックスは1。FH*FW個の要素がある）中の要素の最大値を取る。
        out2d = np.max(x2d, axis=1)

        # 4次元に戻す。
        #   上記次元構造（2次元）：バッチ軸（B個）／高さ軸（OH個）ー幅軸（OW個）ーチャネル軸（C個）
        #   ⇒4次元化：バッチ軸（B個）／高さ軸（OH個）／幅軸（OW個）／チャネル軸（C個）
        #   ⇒フィルタ軸を先に持ってくる：バッチ軸（B個）／チャネル軸（C個）／高さ軸（OH個）／幅軸（OW個）／
        out = out2d.reshape(self.B, self.OH, self.OW, self.C)
        out = out.transpose(0, 3, 1, 2)
        return out

    def backward(self, dout):
        # doutのチャンネル数軸を4番目に移動させる
        dout = dout.transpose(0, 2, 3, 1)

        # プーリング適応領域の要素数(プーリング適応領域の高さ × プーリング適応領域の幅)
        pool_size = self.FH * self.FW

        # 勾配を入れる配列を初期化する
        dx2d = np.zeros((dout.size, pool_size))

        # 順伝播計算時に最大値となった場所に、doutを配置する。
        # 逆に言うと、順伝播時に信号が流れなかった場所には逆伝播しない。
        # （例）入力データが最初の2行2列フィルターに当たるウィンドウ[x00, x01, x10, x11]のうち、
        # 　　　最大値がx10だとすると、x10の場所(argmaxが2)が1で、その他は0として信号が順伝播してきたとイメージするとよい。
        dx2d[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()

        # 勾配を4階テンソルに変換。
        dLdX = col2im_HP(dx2d, self.x.shape, self.FH, self.FW, self.padding, self.stride, is_maxpooling=True)

        return dLdX

##############################
# 以下ユーティリティ。特に畳み込み層、プーリング層で使用するメソッド。
##############################
##############################
# 出力サイズ算出
##############################
def out_size(in_size, fil_size, padding, stride):
    return (in_size + 2 * padding - fil_size) // stride + 1

##############################
# 4階テンソルの2次元化【高速版；参考資料の改変版】
# （参考）斎藤康毅『ゼロから作るDeep Learning』（オライリー・ジャパン）
# ただし、上記参考資料とは異なり、中間データとして利用する6階のテンソルの次元構造の一部を、
# 出力（の高さ軸、幅軸）→フィルタ（の高さ軸、幅軸）の順にした。
##############################
def im2col(x, FH, FW, padding=0, stride=1):
    B, C, H, W = x.shape
    OH = out_size(H, FH, padding, stride)
    OW = out_size(W, FW, padding, stride)

    # 画像データの高さ方向と幅方向を0パディング。
    # 第1, 2次元：ヘッドもテイルもパディングしない。第3, 4次元：ヘッドとテイル両方にパディング。
    img = np.pad(x, [(0, 0), (0, 0), (padding, padding), (padding, padding)], 'constant')

    # 一旦6次元配列を準備。ただし、参考資料のコードと違って、出力側のループを先に持ってくる。こちらの方がロジックが分かりやすい。
    x6d = np.zeros((B, C, OH, OW, FH, FW))

    # 入力データの移し替え。
    for i in range(OH):
        il = i * stride
        ir = il + FH
        for j in range(OW):
            jl = j * stride
            jr = jl + FW
            x6d[:, :, i, j, :, :] = img[:, :, il:ir, jl:jr]

    # チャネル次元を最後に持って行く。チャネル次元は列方向に並べたいため。
    x2d = x6d.transpose(0, 2, 3, 4, 5, 1).reshape(B * OH * OW, -1)

    return x2d

##############################
# 4階テンソルの2次元化【高速版】
# （参考）斎藤康毅『ゼロから作るDeep Learning』（オライリー・ジャパン）のロジックそのまま。
# 入力xから、stride個分飛ばしながらスライスして値を取ってくる方法。
##############################
def im2col_HP(x, FH, FW, padding=0, stride=1):
    B, C, H, W = x.shape
    OH = out_size(H, FH, padding, stride)
    OW = out_size(W, FW, padding, stride)

    # 画像データの高さ方向と幅方向を0パディング。
    # 第1, 2次元：ヘッドもテイルもパディングしない。第3, 4次元：ヘッドとテイル両方にパディング。
    img = np.pad(x, [(0, 0), (0, 0), (padding, padding), (padding, padding)], 'constant')

    # 一旦6次元配列を準備。
    x6d = np.zeros((B, C, FH, FW, OH, OW))

    # 入力データの移し替え。
    for h in range(FH):
        hmax = h + stride * OH
        for w in range(FW):
            wmax = w + stride * OW
            x6d[:, :, h, w, :, :] = img[:, :, h:hmax:stride, w:wmax:stride]

    x2d = x6d.transpose(0, 4, 5, 1, 2, 3).reshape(B * OH * OW, -1)
    return x2d

##############################
# 4次元化処理【高速版】
# （参考）斎藤康毅『ゼロから作るDeep Learning』（オライリー・ジャパン）
# ただし、参考資料とは異なり、畳み込み層で使う場合と、最大値プーリング層で使う場合とで処理を分けることができるようにしている。
##############################
def col2im_HP(col, input_shape, FH, FW, padding=0, stride=1, is_maxpooling=False):
    B, C, IH, IW = input_shape
    OH = out_size(IH, FH, padding, stride)
    OW = out_size(IH, FW, padding, stride)

    # 配列の形を変えて、軸を入れ替える
    col = col.reshape(B, OH, OW, C, FH, FW).transpose(0, 3, 4, 5, 1, 2)

    # 配列の初期化。pad分を大きくとっておく。stride分も大きくとっておく。
    img = np.zeros((B, C, IH + 2 * padding + stride - 1, IW + 2 * padding + stride - 1))

    # 配列を並び替える
    for y in range(FH):
        y_max = y + stride * OH
        for x in range(FW):
            x_max = x + stride * OW
            if is_maxpooling:
                img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
            else:
                img[:, :, y:y_max:stride, x:x_max:stride] = col[:, :, y, x, :, :]

    return img[:, :, padding:IH + padding, padding:IW + padding]  # pad分は除いておく(pad分を除いて真ん中だけを取り出す)

##############################
# TODO 以下、初期実装時に作成したが廃止したユーティリティメソッド。
##############################
# TODO メソッド化してはみたが、実際に使用する側のメソッド内で実装できるため廃止。
# def convolute(x, f, padding, stride):
#     batch_size, channel_size, IH, IW = x.shape
#     FN, FC, FW, FH = f.shape
#     OH = out_size(IH, FH, padding, stride)
#     OW = out_size(IW, FW, padding, stride)
#
#     # 入力とフィルターそれぞれの2次元化。
#     x2d = im2col(x, FH, FW, padding, stride)
#     f2d = im2col_filter(f)
#
#     # 行列積を取る。
#     # 次元構造は、バッチ軸（B個）／フィルタ軸（FN個）×高さ軸（OH個）×幅軸（OW個）の2次元構造である。
#     # print("x2d.shape=", x2d.shape)
#     # print("f2d.shape=", f2d.shape)
#     out2d = np.dot(x2d, f2d.T)
#
#     # 4次元に戻す。
#     #   上記次元構造（2次元）：バッチ軸（B個）／高さ軸（OH個）ー幅軸（OW個）ーフィルタ軸（FN個）
#     #   ⇒4次元化：バッチ軸（B個）／高さ軸（OH個）／幅軸（OW個）／フィルタ軸（FN個）
#     #   ⇒フィルタ軸を先に持ってくる：バッチ軸（B個）／フィルタ軸（FN個）／高さ軸（OH個）／幅軸（OW個）
#     # 　※フィルタ数が新たなチャネル数になることに注意。
#     out = out2d.reshape(batch_size, OH, OW, FN)
#     out = out.transpose(0, 3, 1, 2)
#     return out

# TODO メソッド化してみたが、メモリ消費が過大になるため使用しない。廃止。
# # ------------------------------
# # 同一データ（行、列）挿入メソッド。
# # 与えられたテンソルの与えられた軸の各データの直後に、そのデータのコピーを挿入する。
# # 実際には、引数のxは4次元テンソルを想定し、画像データの高さ軸（第3次元目；次元インデックスは2）、
# # または幅軸（第3次元目；次元インデックスは2）方向の処理を想定している。
# # （例）
# # [
# #     [1,2,3],
# #     [4,5,6],
# #     [7,8,9],
# # ]
# # ↓　軸として第1次元（行方向）、insert_numとして2を指定した場合、以下となる。
# # [
# #     [1,2,3],　元の1行目をコピー。
# #     [1,2,3],　ペースト1回目。
# #     [1,2,3],　ペースト2回目。
# #     [4,5,6],　元の2行目をコピー。
# #     [4,5,6],　ペースト1回目。
# #     [4,5,6],　ペースト2回目。
# #     [7,8,9],　元の3行目をコピー。
# #     [7,8,9],　ペースト1回目。
# #     [7,8,9],　ペースト2回目。
# # ]
# # ------------------------------
# def insert_same_values(x, insert_num=0, axis=2, inner_only=False):
#     if inner_only:
#         org_shape = x.shape[axis] - 1
#     else:
#         org_shape = x.shape[axis]
#
#     for i in range(org_shape):
#         pos = 1 + i * (insert_num + 1)
#         for j in range(insert_num):
#             if axis == 2:
#                 values = x[:,:,pos-1,:]
#             elif axis == 3:
#                 values = x[:,:,:,pos-1]
#             else:
#                 # TODO エラーにするべき。
#                 pass
#             x = np.insert(x, pos, values, axis=axis)
#     return x

# TODO 多重ループを使用していて低速なため廃止。
# ##############################
# # 4階テンソルの2次元化。for文を利用した独自ロジック。低速のため廃止。
# ##############################
# def im2col_ORIGINAL(x, FH,FW, padding, stride):
#     # TODO debug
#     # print("x.shape=", x.shape)
#     # print(x)
#     # print()
#
#     # 入力値行列およびフィルター行列から各次元の要素数を取得する。
#     # ただし、バッチ数B、チャネル数Cはそれぞれ、入力値行列とフィルター行列とで一致していなければならない。
#     B, C, IH, IW = x.shape
#     # TODO debug
#     OH = int(np.floor((IH + 2 * padding - FH) / stride)) + 1
#     OW = int(np.floor((IW + 2 * padding - FW) / stride)) + 1
#     # print("IH,IW={0},{1}".format(IH, IW))
#     # print("FH,FW={0},{1}".format(FH, FW))
#     # print("OH,OW={0},{1}".format(OH, OW))
#     # print()
#
#     # パディングした行列を作る。
#     x_pad_H = IH + 2 * padding
#     x_pad_W = IW + 2 * padding
#     x_pad = np.zeros((B, C, x_pad_H, x_pad_W))
#     for b in range(B):
#         for c in range(C):
#             for h in range(x_pad_H):
#                 if (padding <= h) & (h < padding + IH):  # except padding area
#                     for w in range(x_pad_W):
#                         if (padding <= w) & (w < padding + IW):  # except padding area
#                             x_pad[b][c][h][w] = x[b][c][h - padding][w - padding]
#     # print("x_pad_H, x_pad_W={0},{1}".format(x_pad_H, x_pad_W))
#     # print(x_pad)
#     # print()
#
#     # 2次元化データを作る。
#     x2d_h = B * OH * OW
#     x2d_w = C * FH * FW
#     x2d = np.zeros((x2d_h, x2d_w))
#     # print("x2d_h(=B*OH*OW), x2d_w(=C*FH*FW) = {0},{1}".format(x2d_h, x2d_w))
#     # print()
#
#     for b in range(B):
#         batch_offset = b * OH * OW
#         for c in range(C):
#             channel_offset = c * FH * FW
#             row = batch_offset
#             for i in range(OH):
#                 for j in range(OW):
#                     h0 = stride * i  # 高さ方向の開始位置。
#                     w0 = stride * j  # 幅方向の開始位置。
#                     mat = x_pad[b][c][h0:(h0 + FH), w0:(w0 + FW)]  # filterと同じサイズの行列を抜き出す。
#                     arr = mat.reshape(1, -1)  # 1行、FH*FW列の行列に変形（実体は2次元配列だが、内容としては1次元配列）。
#                     for col in range(arr.shape[1]):
#                         x2d[row][channel_offset + col] = arr[0][col]  # 1行しかないので第1次元は0を指定。
#                     row += 1  # x_pad内のウィンドウを1ストライド動かすたびにカウントアップする。
#
#     return x2d

# TODO 以下独自ロジックにはバグがあるためdeprecated.
# ------------------------------
# 以下エラー内容：
#   File "C:\study\dldev\2_notebook\convolutions.py", line 165, in backward
#     dLdX = col2im(dcol, self.x.shape[0], self.x.shape[1], self.x.shape[2], self.x.shape[3], self.FH, self.FW, self.padding, self.stride)
#   File "C:\study\dldev\2_notebook\convolutions.py", line 336, in col2im
#     x_pad[b][c][h_start + i][w_start + j] = mat3[i][j]
# IndexError: index 27 is out of bounds for axis 0 with size 27
# ------------------------------
# def col2im(x2d, B, C, IH, IW, FH, FW, padding, stride):
#     OH = int(np.floor((IH + 2 * padding - FH) / stride)) + 1
#     OW = int(np.floor((IW + 2 * padding - FW) / stride)) + 1
#
#     # まず行方向をバッチ軸として分ける。（第1次元：バッチ軸。次元のインデックスは0）
#     # 次に各バッチをウィンドウの走査個数ずつ（OH*OW個ある）に分ける。（第2次元：ウィンドウの走査軸。次元のインデックスは1）
#     # さらに各ウィンドウをチャネルの個数ずつに分ける。（第3次元：チャネル軸。次元のインデックスは2）
#     # 残った軸は自動的に決まり、各チャネル内の各ウィンドウ内の走査軸となる。（第4次元：ウィンドウ内の走査軸。次元のインデックスは3）
#     # 最後に、第2次元と第3次元を転置することによって、元の画像データの次元構造に戻る。
#     mat = x2d.reshape(B, OH * OW, C, FH * FW).transpose(0, 2, 1, 3)
#     x_pad = np.zeros((B, C, IH + 2 * padding, IW + 2 * padding))
#     x = np.zeros((B, C, IH, IW))
#     for b in range(B):
#         for c in range(C):
#             h_start = 0
#             w_start = 0
#             mat2 = mat[b][c]
#             for row in range(mat2.shape[0]):
#                 mat3 = mat2[row].reshape(FH, FW)
#                 for i in range(FH):
#                     for j in range(FW):
#                         x_pad[b][c][h_start+i][w_start+j] = mat3[i][j]
#                 if row == (OW - 1):
#                     h_start += stride
#                     w_start = 0
#                 else:
#                     w_start += stride
#             x[b][c] = x_pad[b][c][padding:padding+IH, padding:padding+IW]
#
#     return x

# # ------------------------------
# # 行列の内部0パディング
# # 損失関数をL、畳み込みの結果をOとすると、dL/dWおよびdL/dXを求めるために、dL/dOに内部パディングを入れる必要がある。そのための関数。
# # ------------------------------
# def inner_padding(x, padding_num=0, axis=[2,3]):
#     for i in axis:
#         x = insert_zero(x, padding_num=padding_num, axis=i, inner_only=True)
#     return x
#
# def insert_zero(x, padding_num=0, axis=0, inner_only=False):
#     ZERO = 0
#     if inner_only:
#         org_shape = x.shape[axis] - 1
#     else:
#         org_shape = x.shape[axis]
#
#     for i in range(org_shape):
#         pos = 1 + i * (padding_num + 1)
#         for j in range(padding_num):
#             x = np.insert(x, pos, ZERO, axis=axis)
#     return x
#
# # ------------------------------
# # 180度フリップ。指定軸（array-like）で反転する。
# # ------------------------------
# def flip180(a, axis=[2,3]):
#     for i in axis:
#         a = np.flip(a, axis=i)
#     return a
#

# TODO フィルターの2次元化は1行で済むのでメソッド化の必要なし。
# # ------------------------------
# # フィルターの2次元化。
# # ------------------------------
# def im2col_filter(f):
#     FN, C, FH, FW = f.shape
#     # フィルター軸に沿ってFN個に分割することによって、自動的に（フィルター軸、チャネルの個数×高さの個数×幅の個数）に直列化する。
#     # なぜなら、そもそも、フィルター軸を1つ固定すると、（チャネル軸、高さ軸、幅軸）で分類されているため、
#     # 直列化する順番としては、横軸方向に並ぶ⇒高さ軸方向に並ぶ⇒さらにチャネル軸方向に並ぶ、となるため。
#     f2d = f.reshape(FN, -1)
#
#     # TODO 以下のようにゴリゴリ作らなくても、上記のように第1次元でreshape掛ければ終わりなので以下不要ロジック。
#     # 行方向がフィルター個数の方向、列方向がチャネルの方向（およびフィルター内データ1次元化の方向）とする。
#     # f2d = np.zeros((FN, FH * FW * C))
#     # for fn in range(FN):
#     #     for c in range(C):
#     #         channel_offset = c * FH * FW
#     #         for i in range(FH):
#     #             for j in range(FW):
#     #                 h0 = stride * i  # 高さ方向の開始位置。
#     #                 w0 = stride * j  # 幅方向の開始位置。
#     #                 # mat = f[fn][c][h0:(h0 + FH), w0:(w0 + FW)]  # filterと同じサイズの行列を抜き出す。# TODO 実体はfilterなので不要。以下ロジックで終わり。
#     #                 mat = f[fn][c]
#     #                 arr = mat.reshape(1, -1)  # 1行、FH*FW列の行列に変形（実体は2次元配列だが、内容としては1次元配列）。
#     #                 for col in range(arr.shape[1]):
#     #                     f2d[fn][channel_offset + col] = arr[0][col]  # 1行しかないので第1次元は0を指定。
#     return f2d
