import Config

import argparse
import Config
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import PIL as pil
import cv2 as cv
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from models.keras_yolo import (preprocess_true_boxes, yolo_body,
                               yolo_eval, yolo_head, yolo_loss)
from utils.draw_boxes import draw_boxes
import gc
import xml.etree.ElementTree as ET
import glob
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from multiprocessing import Pool

# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))

SCORE_THRESSHOLD = 0.75
IOU_THRESSHOLD = 0.0
class LicensePlateLocalizor:
    def __init__(self):
        self.model = None
        self.model_body = None
        self.config = Config.Config.getInstance()
        self.load_model()

    def load_model(self):
        class_names = self.get_classes()
        anchors = self.get_anchors()
        self.graph = tf.Graph()
        with self.graph.as_default() as graph_default:
            self.sess = K.get_session()
            if self.model is None or self.model_body is None:
                self.create_model(anchors, class_names)
                self.model_body.load_weights(self.config.plateDetectWeight)
            self.yolo_outputs = yolo_head(self.model_body.output, anchors, len(class_names))
            self.input_image_shape = K.placeholder(shape=(2,))
            self.boxes, self.scores, self.classes = yolo_eval(
                self.yolo_outputs, self.input_image_shape, score_threshold=SCORE_THRESSHOLD, iou_threshold=IOU_THRESSHOLD)

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
                self.model_body = load_model(self.config.yoloH5Weights)
                self.model_body = Model(self.model_body.inputs, self.model_body.layers[-2].output)
                self.model_body.save_weights(self.config.yoloTopless)
            topless_yolo.load_weights(self.config.yoloTopless)

        if freeze_body:
            for layer in topless_yolo.layers:
                layer.trainable = False
        final_layer = Conv2D(len(anchors) * (5 + len(class_names)), (1, 1), activation='linear')(topless_yolo.output)

        self.model_body = Model(image_input, final_layer)
        # with tf.device('/cpu:0'):
        # TODO: Replace Lambda with custom Keras layer for loss.
        model_loss = Lambda(
            yolo_loss,
            output_shape=(1,),
            name='yolo_loss',
            arguments={'anchors': anchors,
                       'num_classes': len(class_names)})([
            self.model_body.output, boxes_input,
            detectors_mask_input, matching_boxes_input
        ])

        self.model = Model(
            [self.model_body.input, boxes_input, detectors_mask_input,
             matching_boxes_input], model_loss)

        return

    def detect(self, image):
        # Image preprocessing.
        img = image.copy()
        width, height = image.shape[1], image.shape[0]
        img = cv.resize(img, ((416, 416)), cv.INTER_CUBIC)
        img = np.array(img, dtype=np.float)
        img = img / 255.

        img = np.array(np.expand_dims(img, axis=0))
        with self.graph.as_default() as graph_default:
            out_boxes, out_scores, out_classes = self.sess.run(
                [self.boxes, self.scores, self.classes],
                feed_dict={
                    self.model_body.input: img,
                    self.input_image_shape: [img.shape[1], img.shape[2]],
                    K.learning_phase(): 0
                })
        true_out_boxes = list()
        for out_box in out_boxes:
            bottom, left, top, right = out_box / 416.
            # correct size after resize image in processing
            top, left, bottom, right = top*height, left*width, bottom*height, right*width
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(height, np.floor(bottom + 0.5).astype('int32'))
            right = min(width, np.floor(right + 0.5).astype('int32'))

            top, left, bottom, right = self.extend_boxes(top, left, bottom, right, width, height)

            true_out_box = top, left, bottom, right
            true_out_boxes.append(true_out_box)

        return true_out_boxes, out_scores, out_classes

    def show_locate(self, image):
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        true_out_boxes, out_scores, out_classes = self.detect(image)
        final_images = list()
        for box in true_out_boxes:
            top, left, bottom, right = box
            final_image = image[bottom:top, left:right, :]
            final_image = cv.cvtColor(final_image, cv.COLOR_BGR2RGB)
            cv.imshow('Plate image', final_image)
            cv.waitKey(0)
        cv.destroyAllWindows()

    def extend_boxes(self, top, left, bottom, right, width, height):
        width_extend = np.float32(width) / 20.
        height_extend = np.float32(height) / 20.
        top = min(height, np.floor(top + height_extend).astype('int32'))
        left = max(0, np.floor(left - width_extend).astype('int32'))
        bottom = max(0, np.floor(bottom - height_extend).astype('int32'))
        right = min(width, np.floor(right + width_extend).astype('int32'))

        return top, left, bottom, right

    def get_classes(self):

        with open(self.config.plateLabels) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def get_anchors(self):
        if os.path.isfile(self.config.yoloAnchors):
            with open(self.config.yoloAnchors) as f:
                anchors = f.readline()
                anchors = [float(x) for x in anchors.split(',')]
                return np.array(anchors).reshape(-1, 2)
        else:
            Warning("Could not open anchors file, using default.")
            return YOLO_ANCHORS


if __name__ == '__main__':
    Config.Config('local')
    removal = LicensePlateLocalizor()
    removal.load_model()
    images_path = 'E:\ImageData\colordata\yolo_img'
    output = 'E:\Source\LicensePlateRecognition\data\output_test'
    files = os.listdir(images_path)
    count = 0
    for file in files:
        image_path = os.path.join(images_path, file)
        # fix opencv read none image
        if '-' in image_path:
            new_image_path = image_path.replace('-', '_')
            os.rename(image_path, new_image_path)
            image = cv.imread(new_image_path)
        else:
            image = cv.imread(image_path)
        if image is None:
            continue
        removal.show_locate(image)
        # if count > 3000:
        #     break
    print()
