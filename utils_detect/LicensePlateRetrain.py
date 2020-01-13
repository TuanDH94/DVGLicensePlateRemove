import argparse
import Config
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import PIL as pil
from PIL import Image
import cv2 as cv
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from models.keras_yolo import (preprocess_true_boxes, yolo_body,
                               yolo_eval, yolo_head, yolo_loss)
from keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator
from utils.draw_boxes import draw_boxes
import gc
import xml.etree.ElementTree as ET
import glob
from keras.utils.training_utils import multi_gpu_model

import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
from sys import getsizeof

YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))


class LicensePlateTrainer:
    def __init__(self):
        self.config = Config.Config.getInstance()
        self.pick = None
        self.images = None
        self.boxes = None
        self.model = None

    def pipeline(self):
        self.load_data()
        self.process_data()
        self.train(is_validate=True)

    def train(self, is_validate=True):
        anchors = self.get_anchors(self.config.yoloAnchors)
        detectors_mask, matching_true_boxes = self.get_detector_mask(self.boxes, anchors)

        image_data_train, image_data_val, \
        boxes_train, boxes_val, \
        detectors_mask_train, detectors_mask_val, \
        matching_true_boxes_train, matching_true_boxes_val, \
            = train_test_split(self.images, self.boxes, detectors_mask, matching_true_boxes, test_size=0.15,
                               random_state=0)

        model_body, model = self.create_model(anchors, self.pick, load_pretrained=True, freeze_body=True)

        # multtiple gpu
        model = multi_gpu_model(model, gpus=2)
        model.compile(
            optimizer='adam', loss={
                'yolo_loss': lambda y_true, y_pred: y_pred
            })  # This is a hack to use the custom loss function in the last layer.

        logging = TensorBoard()
        checkpoint = ModelCheckpoint(self.config.plateDetectWeight,
                                     monitor='val_loss',
                                     save_weights_only=True, save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

        model.fit([image_data_train, boxes_train, detectors_mask_train, matching_true_boxes_train],
                  np.zeros(len(image_data_train)),
                  validation_data=([image_data_val, boxes_val, detectors_mask_val, matching_true_boxes_val],
                                   np.zeros(len(image_data_val))),
                  batch_size=8,
                  epochs=30,
                  callbacks=[logging, checkpoint, early_stopping])
        if is_validate:
            self.draw(model_body, self.pick, anchors, image_data_val)

    def draw(self, model_body, class_names, anchors, image_data,
             out_path="output_images", out_crop="output_crop", save_all=True):
        # load best weight
        model_body.load_weights(self.config.yoloWeights)

        yolo_outputs = yolo_head(model_body.output, anchors, len(class_names))
        input_image_shape = K.placeholder(shape=(2,))
        boxes, scores, classes = yolo_eval(
            yolo_outputs, input_image_shape, score_threshold=0.80, iou_threshold=0.0)
        sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

        if not os.path.exists(out_path):
            os.makedirs(out_path)
        for i in range(len(image_data)):
            out_boxes, out_scores, out_classes = sess.run(
                [boxes, scores, classes],
                feed_dict={
                    model_body.input: image_data[i],
                    input_image_shape: [image_data.shape[2], image_data.shape[3]],
                    K.learning_phase(): 0
                })
            print('Found {} boxes for image.'.format(len(out_boxes)))
            print(out_boxes)

            # Plot image with predicted boxes.
            image_with_boxes, origin_image, crop_boxs = draw_boxes(image_data[i][0], out_boxes, out_classes,
                                                                   class_names, out_scores)
            for ib, crop_box in enumerate(crop_boxs):
                origin_image.crop(crop_box).save(os.path.join(out_crop, str(i) + '-' + str(ib) + '.png'))
            # Save the image:
            if save_all or (len(out_boxes) > 0):
                cv.imwrite(os.path.join(out_path, str(i) + '.jpg'), image_with_boxes)

    def process_data(self):
        self.images = np.array(self.images)
        print(getsizeof(self.images))
        #  # Box preprocessing.
        self.boxes = np.array(self.boxes)
        print(getsizeof(self.boxes))
        old_boxes = self.boxes.copy()
        new_boxes = list()
        for i, box in enumerate(old_boxes):
            orig_size = box[0:2]
            boxzs = box[2]
            boxzs_new = list()
            for boxz in boxzs:
                # Get box parameters as x_center, y_center, box_width, box_height, class.
                x_center = 0.5 * (boxz[2] + boxz[0]) / orig_size[0]
                y_center = 0.5 * (boxz[3] + boxz[1]) / orig_size[1]
                box_width = (boxz[2] - boxz[0]) / orig_size[0]
                box_height = (boxz[3] - boxz[1]) / orig_size[1]

                boxz_new = [x_center, y_center, box_width, box_height, boxz[4]]
                boxzs_new.append(boxz_new)
            boxzs_new = np.array(boxzs_new)
            new_boxes.append(boxzs_new)
        self.boxes = np.array(new_boxes)
        del old_boxes
        gc.collect()

        # # Padding
        # find the max number of boxes
        max_boxes = 0
        for boxz in self.boxes:
            if boxz.shape[0] > max_boxes:
                max_boxes = boxz.shape[0]

        # add zero pad for training
        for i, boxz in enumerate(self.boxes):
            if boxz.shape[0] == 0:
                # boxes_new[i] = np.zeros( (max_boxes, 5), dtype=np.float32)
                print('error empty box ' + str(i))
                # replace with previous
                self.boxes[i] = self.boxes[i - 1].copy()
                self.images[i] = self.images[i - 1].copy()
            elif boxz.shape[0] < max_boxes:
                zero_padding = np.zeros((max_boxes - boxz.shape[0], 5), dtype=np.float32)
                self.boxes[i] = np.vstack((boxz, zero_padding))
        self.boxes = np.array(self.boxes)

    def load_data(self):
        self.pick = self.get_classes()
        self.boxes = list()
        self.images = list()
        cur_dir = os.getcwd()
        os.chdir(self.config.plateAnnotations)
        annotations = os.listdir('.')
        annotations = glob.glob(str(annotations) + '*.xml')
        size = len(annotations)

        os.chdir(self.config.plateImages)
        all_imgs = os.listdir('.')
        dsize = (416, 416)
        self.images = np.empty((len(annotations), *dsize, 3))
        for i, file in enumerate(annotations):
            # progress bar
            sys.stdout.write('\r')
            percentage = 1. * (i + 1) / size
            progress = int(percentage * 20)
            bar_arg = [progress * '=', ' ' * (19 - progress), percentage * 100]
            bar_arg += [file]
            sys.stdout.write('[{}>{}]{:.0f}%  {}'.format(*bar_arg))
            sys.stdout.flush()

            annotation_path = os.path.join(self.config.plateAnnotations, file)
            box = self.parse_annotations(annotation_path)
            if box is not None:
                image = file.replace('xml', 'jpg')
                if image in all_imgs:
                    cv_img = cv.imread(image)
                    cv_img = cv.resize(cv_img, (416, 416))
                    # image = Image.open(image)
                    # image = image.resize((416,416))
                    image = np.array(cv_img, dtype=np.float32)
                    image = image / 255.
                    cv_img = None
                    del cv_img
                    self.images[i] = image
                    self.boxes.append(box)
        gc.collect()

    def parse_annotations(self, annotation_path):
        in_file = open(annotation_path)
        tree = ET.parse(in_file)
        root = tree.getroot()
        imsize = root.find('size')
        w = int(imsize.find('width').text)
        h = int(imsize.find('height').text)
        if w == 0 or h == 0:
            return None
        boxzs = list()
        for obj in root.iter('object'):
            name = obj.find('name').text
            if name not in self.pick:
                continue
            name = self.pick.index(name)
            xmlbox = obj.find('bndbox')
            xn = int(float(xmlbox.find('xmin').text))
            xx = int(float(xmlbox.find('xmax').text))
            yn = int(float(xmlbox.find('ymin').text))
            yx = int(float(xmlbox.find('ymax').text))
            boxz = np.array([xn, yn, xx, yx, name])
            boxzs.append(boxz)
        box = np.array([w, h, boxzs])

        return box

    def create_model(self, anchors, class_names, load_pretrained=True, freeze_body=True):
        detectors_mask_shape = (13, 13, 5, 1)
        matching_boxes_shape = (13, 13, 5, 5)

        # Create model input layers.
        image_input = Input(shape=(416, 416, 3))
        boxes_input = Input(shape=(None, 5))
        detectors_mask_input = Input(shape=detectors_mask_shape)
        matching_boxes_input = Input(shape=matching_boxes_shape)

        # Create model body.
        yolo_model = yolo_body(image_input, len(anchors), len(class_names))
        topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)

        if load_pretrained:
            # Save topless yolo:
            if not os.path.exists(self.config.yoloTopless):
                print("CREATING TOPLESS WEIGHTS FILE")
                model_body = load_model(self.config.yoloH5Weights)
                model_body = Model(model_body.inputs, model_body.layers[-2].output)
                model_body.save_weights(self.config.yoloTopless)
            topless_yolo.load_weights(self.config.yoloTopless)

        if freeze_body:
            for layer in topless_yolo.layers:
                layer.trainable = False
        final_layer = Conv2D(len(anchors) * (5 + len(class_names)), (1, 1), activation='linear')(topless_yolo.output)

        model_body = Model(image_input, final_layer)
        # with tf.device('/cpu:0'):
        # TODO: Replace Lambda with custom Keras layer for loss.
        model_loss = Lambda(
            yolo_loss,
            output_shape=(1,),
            name='yolo_loss',
            arguments={'anchors': anchors,
                       'num_classes': len(class_names)})([
            model_body.output, boxes_input,
            detectors_mask_input, matching_boxes_input
        ])

        model = Model(
            [model_body.input, boxes_input, detectors_mask_input,
             matching_boxes_input], model_loss)

        return model_body, model

    def get_detector_mask(self, boxes, anchors):
        detectors_mask = [0 for i in range(len(boxes))]
        matching_true_boxes = [0 for i in range(len(boxes))]
        for i, box in enumerate(boxes):
            detectors_mask[i], matching_true_boxes[i] = preprocess_true_boxes(box, anchors, [416, 416])

        return np.array(detectors_mask), np.array(matching_true_boxes)

    def get_classes(self):
        with open(self.config.plateLabels) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def get_anchors(self, anchors_path):
        if os.path.isfile(self.config.yoloAnchors):
            with open(anchors_path) as f:
                anchors = f.readline()
                anchors = [float(x) for x in anchors.split(',')]
                return np.array(anchors).reshape(-1, 2)
        else:
            Warning("Could not open anchors file, using default.")
            return YOLO_ANCHORS


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description="Training license plate")
    argparser.add_argument(
        '-m',
        '--mode',
        default='training')
    argparser = argparser.parse_args()
    Config.Config(argparser.mode)
    trainer = LicensePlateTrainer()
    trainer.pipeline()
