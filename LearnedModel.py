import pickle

class LearnedModel:
    def __init__(self):
        # わざわざNoneにする必要はないが、保存対象パラメータの一覧の意味であえて定義しておく。
        self.input_size = None
        self.layer_size_list = None
        self.init_weight_stddev = None
        self.epoch_num = None
        self.mini_batch_size = None
        self.learning_rate = None
        self.W = None
        self.B = None

    def save(self, fpath):
        pkl_file = open(fpath, "wb")
        lm = {}
        lm["input_size"] = self.input_size
        lm["layer_size_list"] = self.layer_size_list
        lm["init_weight_stddev"] = self.init_weight_stddev
        lm["epoch_num"] = self.epoch_num
        lm["mini_batch_size"] = self.mini_batch_size
        lm["learning_rate"] = self.learning_rate
        lm["W"] = self.W
        lm["B"] = self.B
        pickle.dump(lm, pkl_file)
        pkl_file.close()

    def load(self, fpath):
        # print("★LearedModel load start")
        pkl_file = open(fpath, "rb")
        lm = pickle.load(pkl_file)
        # print("lm=", lm)
        self.input_size = lm["input_size"]
        self.layer_size_list = lm["layer_size_list"]
        self.init_weight_stddev = lm["init_weight_stddev"]
        self.epoch_num = lm["epoch_num"]
        self.mini_batch_size = lm["mini_batch_size"]
        self.learning_rate = lm["learning_rate"]
        self.W = lm["W"]
        self.B = lm["B"]
        pkl_file.close()
        # print("★LearedModel load end")
        return self
