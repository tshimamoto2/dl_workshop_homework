import numpy as np
import os
import re
import glob
from PIL import Image
from sklearn.preprocessing import LabelBinarizer
from collections import OrderedDict
import pickle

class ImageLoader:
    def __init__(self):
        pass

    def makedataset(self, dirpath):
        data = []
        label = []

        # 指定されたディレクトリ名の最後のスラッシュを削除してからフルパスを正規表現で組み立てる（globを使いたいから）。
        fpath_regex = "{0}/{1}".format(re.sub("/*$", "", dirpath), "*/**.png")
        flist = glob.glob(fpath_regex)
        for i, fpath in enumerate(flist):
            img = MyImage(fpath)
            data.append(img.data())
            label.append(img.label())

        # numpyでのdot演算のためにnumpy.array化。
        return np.array(data), np.array(label)


# from keras.preprocessing.image import load_img, img_to_array, array_to_img

class MyImage:
    kanadict = OrderedDict(a=0, i=1, u=2, e=3, o=4)

    def __init__(self, fpath):
        self.img = Image.open(fpath)
        self.fpath = fpath
        # self.img = load_img(fpath, target_size=(28, 28))

    def data(self):
        # numpyでのdot演算のためにnumpy.array化。
        # return np.array(self.img.getdata())
        array = np.array(self.img)

        # 画素を直列化（1次元配列化）
        array = array.reshape(1, -1)

        # 画素データを正規化。
        # TODO 画素の最大値を使用するべきか？
        array = array.astype(np.float32)

        # TODO 以下実験してみると、白黒反転すると学習が早く進んだケースがあった。念のため全部試してみる。
        # ↓（実験1）単純に255で割るバージョン
        # array = array / 255.0
        # ↓（実験2）白黒反転バージョン（黒を示す値0が1.0に、白を示す値255が0.0になる）。
        array = (255.0 - array) / 255.0

        return np.array(array[0])

    def label(self):
        # ディレクトリ名を除いてから、画像ファイル名の接頭辞を取り出す。
        label = os.path.split(self.fpath)[1].split("_")[0]
        # カナ辞書で変換。
        label = int(MyImage.kanadict[label])
        # one-hotベクトル化。
        label = LabelBinarizer().fit(list(MyImage.kanadict.values())).transform([label])

        # numpyでのdot演算のためにnumpy.array化。
        return np.array(label[0])
