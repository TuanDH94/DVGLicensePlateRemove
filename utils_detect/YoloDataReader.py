from keras.utils import Sequence
import os
import sys
import numpy as np
import cv2 as cv
from models.keras_yolo import preprocess_true_boxes
import xml.etree.ElementTree as ET

np.random.seed(100)

class YoloDataReader:
    def __init__(self, images_path,
                 annotations_path,
                 anchors,
                 class_name,
                 num_of_batchs,
                 shuffle=True):
        self.image_X = None
        self.box_Y = None
        self.num_of_batchs = num_of_batchs
        self.images_path = images_path
        self.annotations_path = annotations_path
        self.shuffle = shuffle
        self.class_name = class_name
        self.anchors = anchors
        self.init()
        self.on_epoch_end()

    def __getitem__(self, index):
        batch_size = int(np.floor(len(self.list_ids) / self.num_of_batchs))
        indexes = self.indexes[index * batch_size:(index + 1) * batch_size]
        if index == self.num_of_batchs - 1:
            indexes = self.indexes[index * batch_size:]
        # Find list of IDs
        list_ids_temp = [self.list_ids[k] for k in indexes]
        X = self._generate_X(list_ids_temp)
        box, detect_mask, true_mask = self._generate_box(indexes)
        return X, box, detect_mask, true_mask

    def __len__(self):
        return self.num_of_batchs

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def init(self):
        list_paths = os.listdir(self.annotations_path)
        self.list_ids = [path.replace('.xml', '') for path in list_paths]

        # init box
        size = len(self.list_ids)
        max_box = 5
        self.box_Y = np.zeros((size, max_box, 5), dtype=np.float32)

        for i, id in enumerate(self.list_ids):
            file = id +'.xml'
            # progress bar
            sys.stdout.write('\r')
            percentage = 1. * (i + 1) / size
            progress = int(percentage * 20)
            bar_arg = [progress * '=', ' ' * (19 - progress), percentage * 100]
            bar_arg += [file]
            sys.stdout.write('[{}>{}]{:.0f}%  {}'.format(*bar_arg))
            sys.stdout.flush()

            annotation_path = os.path.join(self.annotations_path , file)
            w, h, voc_boxs = self.parse_annotations(annotation_path)

            # fix some error data
            if w == 0 or h == 0:
                image_path = os.path.join(self.images_path, id + '.jpg')
                image = cv.imread(image_path)
                w, h = image.shape[1], image.shape[0]

            # convert voc to yolo format
            for j, voc_box in enumerate(voc_boxs):
                x_center = 0.5 * (voc_box[2] + voc_box[0]) / w
                y_center = 0.5 * (voc_box[3] + voc_box[1]) / h
                box_width = (voc_box[2] - voc_box[0]) / w
                box_height = (voc_box[3] - voc_box[1]) / h
                self.box_Y[i][j] = np.array([x_center, y_center, box_width, box_height, voc_box[4]])

        self.detectors_mask, self.matching_true_boxes = self.get_detector_mask(self.box_Y, self.anchors)

    def _generate_X(self, list_ids):
        dsize = (416, 416)
        images_X = np.empty((len(list_ids), *dsize, 3))
        size = len(list_ids)
        for i, id in enumerate(list_ids):
            # print progressbar
            sys.stdout.write('\r')
            percentage = 1. * (i + 1) / size
            progress = int(percentage * 20)
            bar_arg = [progress * '=', ' ' * (19 - progress), percentage * 100]
            bar_arg += [id]
            sys.stdout.write('[{}>{}]{:.0f}%  {}'.format(*bar_arg))
            sys.stdout.flush()

            image_X = self.generate_image(os.path.join(self.images_path, id + '.jpg'))
            images_X[i] = image_X
        return images_X

    def _generate_box(self, list_indices):
        size = len(list_indices)
        box_Y_by_ids = np.empty((size, *self.box_Y.shape[1:]))
        detectors_mask_by_ids = np.empty((size, *self.detectors_mask.shape[1:]))
        matching_true_boxes_by_ids = np.empty((size, *self.matching_true_boxes.shape[1:]))
        for i, index in enumerate(list_indices):
            box_Y_by_ids[i] = self.box_Y[index]
            detectors_mask_by_ids[i] = self.detectors_mask[index]
            matching_true_boxes_by_ids[i] = self.matching_true_boxes[index]
        return box_Y_by_ids, detectors_mask_by_ids, matching_true_boxes_by_ids

    @staticmethod
    def generate_image(image_path):
        image = cv.imread(image_path)
        image = cv.resize(image, (416, 416))
        image = np.array(image, dtype=np.float32)
        image = image / 255.
        return image

    @staticmethod
    def get_detector_mask(boxes, anchors):
        detectors_mask = [0 for i in range(len(boxes))]
        matching_true_boxes = [0 for i in range(len(boxes))]
        for i, box in enumerate(boxes):
            detectors_mask[i], matching_true_boxes[i] = preprocess_true_boxes(box, anchors, [416, 416])
        detectors_mask, matching_true_boxes = np.array(detectors_mask), np.array(matching_true_boxes)
        return detectors_mask, matching_true_boxes

    def parse_annotations(self, annotation_path):
        in_file = open(annotation_path)
        tree = ET.parse(in_file)
        root = tree.getroot()
        imsize = root.find('size')
        w = int(imsize.find('width').text)
        h = int(imsize.find('height').text)
        boxzs = list()
        for obj in root.iter('object'):
            name = obj.find('name').text
            if name not in self.class_name:
                continue
            name = self.class_name.index(name)
            xmlbox = obj.find('bndbox')
            xn = int(float(xmlbox.find('xmin').text))
            xx = int(float(xmlbox.find('xmax').text))
            yn = int(float(xmlbox.find('ymin').text))
            yx = int(float(xmlbox.find('ymax').text))
            boxz = np.array([xn, yn, xx, yx, name])
            boxzs.append(boxz)

        return w, h, boxzs

if __name__ == '__main__':
    num_of_batchs = 5
    YOLO_ANCHORS = np.array(
        ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
         (7.88282, 3.52778), (9.77052, 9.16828)))
    data_reader = YoloDataReader(
        images_path='E:\ImageData\colordata\yolo_img',
        annotations_path='E:\ImageData\colordata\yolo_annotations',
        anchors=YOLO_ANCHORS,
        class_name=['1'],
        num_of_batchs=num_of_batchs
    )
    for i in range(num_of_batchs):
        X, box, detect_mask, true_mask = data_reader.__getitem__(i)
        data_reader.on_epoch_end()