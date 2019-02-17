from keras.utils import Sequence


class BatchGenerator(Sequence):
    def __init__(self):
        super(BatchGenerator, self).__init__()

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def on_epoch_end(self):
        pass