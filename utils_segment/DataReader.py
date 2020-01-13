import numpy as np
import cv2
import os
import random
import Config
import utils_segment.Utility as Utility

BGR = 0
GRAY = 1
BINARY = 2


class DirectoryScanner:

    def __init__(self, scanner_type=BGR):
        self.train_xs = []
        self.train_ys = []
        self.train_nums = 0
        self.test_xs = []
        self.test_ys = []
        self.test_nums = 0
        self.scanner_type = scanner_type
        self.channels = 3 if scanner_type == BGR else 1

    def vec_from_file(self, file):
        w = Config.char_width
        h = Config.char_height
        img = cv2.imread(file)
        if img is None:
            return None
        img = cv2.resize(img,
                         dsize=(w, h),
                         interpolation=cv2.INTER_CUBIC)

        if self.scanner_type == GRAY:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif self.scanner_type == BINARY:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        x = np.array(img)
        x = np.reshape(x, [h * w * self.channels])
        return x

    def read_data(self, input_dir, target, limit=-1):
        print('Reading data from {}...'.format(input_dir))
        files = os.listdir(input_dir)
        num = len(files)
        if limit != -1:
            if num > limit:
                num = limit
        inputs = []
        targets = []
        for i in range(num):
            file = files[i]
            x = self.vec_from_file(os.path.join(input_dir, file))
            if x is not None:
                inputs.append(x)
                targets.append(target)
            if (i + 1) % 1000 == 0:
                print("Done {} images.".format(i + 1))
        print('There are {} images loaded'.format(len(inputs)))
        return inputs, targets

    def load_train_data(self, pos_dir, neg_dir, limit):
        pos_xs, pos_ys = self.read_data(pos_dir, [0, 1], limit // 2)
        neg_xs, neg_ys = self.read_data(neg_dir, [1, 0], limit // 2)
        self.train_xs = Utility.list_to_numpy(pos_xs + neg_xs)
        self.train_ys = Utility.list_to_numpy(pos_ys + neg_ys)
        self.train_nums = self.train_xs.shape[0]

    def load_test_data(self, pos_dir, neg_dir, limit):
        pos_xs, pos_ys = self.read_data(pos_dir, [0, 1], limit // 2)
        neg_xs, neg_ys = self.read_data(neg_dir, [1, 0], limit // 2)
        self.test_xs = Utility.list_to_numpy(pos_xs + neg_xs)
        self.test_ys = Utility.list_to_numpy(pos_ys + neg_ys)
        self.test_nums = self.test_nums.shape[0]

    def next_train_batch(self, batch_size):
        random_indices = random.sample(list(range(self.train_nums)), batch_size)
        batch_x = self.train_xs[random_indices]
        batch_y = self.train_ys[random_indices]
        return batch_x, batch_y

    def next_test_batch(self, batch_size):
        random_indices = random.sample(list(range(self.test_nums)), batch_size)
        batch_x = self.test_xs[random_indices]
        batch_y = self.test_ys[random_indices]
        return batch_x, batch_y


if __name__ == '__main__':
    neg_dir = r'E:\DVG.Data\dvg\PlateLocalization\PlateCharacters\0'
    pos_dir = r'E:\DVG.Data\dvg\PlateLocalization\PlateCharacters\1'
    scanner = DirectoryScanner(scanner_type=BINARY_INV)
    scanner.load_train_data(pos_dir, neg_dir, limit=40000)