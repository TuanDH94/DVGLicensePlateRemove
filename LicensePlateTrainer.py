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
from utils_detect.YoloDataReader import YoloDataReader
import math

YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))

BATCH_SIZE = 8
EPOCHS = 20


class LicensePlateTrainer:
    def __init__(self):
        self.config = Config.Config.getInstance()
        self.num_of_batchs = 3

        self.class_name = self.get_classes()

        self.train_reader = YoloDataReader(
            self.config.plateImages,
            self.config.plateAnnotations,
            YOLO_ANCHORS,
            self.class_name,
            3
        )

        self.valid_reader = YoloDataReader(
            self.config.plateImages,
            self.config.plateAnnotations,
            YOLO_ANCHORS,
            self.class_name,
            3
        )

    def training(self, drawing=True, training=True):
        self.model_body, self.model, = \
            self.create_model(YOLO_ANCHORS, self.class_name, load_pretrained=True, freeze_body=True)

        # self.model = multi_gpu_model(self.model, gpus=2)

        self.model.compile(
            optimizer='adam', loss={
                'yolo_loss': lambda y_true, y_pred: y_pred
            })  #

        for i in range(self.num_of_batchs):
            image_data_train, boxes_train, detectors_mask_train, matching_true_boxes_train \
                = self.train_reader.__getitem__(i)
            image_data_val, boxes_val, detectors_mask_val, matching_true_boxes_val \
                = image_data_train, boxes_train, detectors_mask_train, matching_true_boxes_train
            image_data_drawing = image_data_val[:200]
            logging = TensorBoard()
            checkpoint = ModelCheckpoint(self.config.plateDetectWeight,
                                         monitor='val_loss',
                                         save_weights_only=True, save_best_only=True)
            early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

            if training:
                self.model.fit([image_data_train, boxes_train, detectors_mask_train, matching_true_boxes_train],
                               np.zeros(len(image_data_train)),
                               validation_data=([image_data_val, boxes_val, detectors_mask_val, matching_true_boxes_val],
                                                np.zeros(len(image_data_val))),
                               batch_size=BATCH_SIZE,
                               epochs=EPOCHS,
                               callbacks=[logging, checkpoint, early_stopping])

            if i == self.num_of_batchs - 1:
                if drawing:
                    image_data_drawing = image_data_val[:200]
                    self.draw(self.model, self.class_name, YOLO_ANCHORS, image_data_drawing)

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

    def draw(self, model_body, class_names, anchors, image_data,
             out_path="output_images", out_crop="output_crop", save_all=True):
        # load best weight
        model_body.load_weights(self.config.plateDetectWeight)

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

    def get_classes(self):
        with open(self.config.plateLabels) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names


if __name__ == '__main__':
    Config.Config('local')
    trainer = LicensePlateTrainer()
    trainer.training(training=False, drawing=True)


    # test load weight and draw
    # image_data_train, boxes_train, detectors_mask_train, matching_true_boxes_train \
    #     = trainer.train_reader.__getitem__(0)
    # image_data_drawing = image_data_train[:200]
    # trainer.draw(trainer.model, trainer.class_name, YOLO_ANCHORS, image_data_drawing)