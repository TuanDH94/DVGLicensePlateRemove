import cv2
import numpy as np

digit_flag = np.array([120, 250, 144, 50, 190, 30, 240, 250, 10])


def create_signature(size):
    batches = size // len(digit_flag)
    remain_size = size - batches * len(digit_flag)
    list = []
    for i in range(batches):
        list.append(digit_flag)
    list.append(np.zeros(remain_size))
    return np.concatenate(list)


def sign(path):
    im = np.array(cv2.imread(path))
    im = np.transpose(im, [2, 0, 1])
    signature = create_signature(im.shape[2])
    signature = np.array([signature, signature, signature])
    im = np.insert(im, 0, signature, axis=1)
    im = np.transpose(im, [1, 2, 0])
    cv2.imwrite(path, im)


def unsign(path):
    im = np.array(cv2.imread(path))
    im = np.transpose(im, [2, 0, 1])
    signature = create_signature(im.shape[2])
    signature = np.array([signature, signature, signature])
    im = np.insert(im, 0, signature, axis=1)
    im = np.transpose(im, [1, 2, 0])
    cv2.imwrite(path, im)


def check(path):
    im = np.array(cv2.imread(path))
    im = np.transpose(im, [2, 0, 1])
    im_signature = im[0][0]
    signature = create_signature(im.shape[2])
    distance = np.average(np.abs(signature - im_signature))
    print('signature distance = {}'.format(distance))
    if distance < 20:
        return True
    else:
        return False