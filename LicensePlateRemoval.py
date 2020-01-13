from LicensePlateLocalizor import LicensePlateLocalizor
from LicensePlateSegmenter import LicensePlateSegmenter
import cv2 as cv
import Config
import logging

logging.basicConfig(filename='app.log', filemode='w+', format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.DEBUG)
import time

class LicensePlateRemoval:
    def __init__(self):
        self.plate_localizor = None
        self.plate_segmenter = None
        self.init_model()

    def init_model(self):
        self.plate_localizor = LicensePlateLocalizor()
        self.plate_segmenter = LicensePlateSegmenter(tracking=False)

    def image_remove(self, image):
        # image = cv.imread(image_path)
        # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        start = time.time()
        out_boxes, out_scores, out_classes = self.plate_localizor.detect(image)
        end = time.time()
        detect_time = end - start
        logging.info('License Plate Detetect on: ' + str(detect_time) + 's')
        start = time.time()
        if len(out_boxes) < 1:
            return image, False
        for box in out_boxes:
            top, left, bottom, right = box
            plate_image = image[bottom:top, left:right, :]
            removed_plate_image, result = self.plate_segmenter.seg_plate(plate_image)
            image[bottom:top, left:right, :] = removed_plate_image
        end = time.time()
        removal_time = end - start
        logging.info('License Plate Removal on: ' + str(removal_time) + 's')
        logging.info('Total time: ' + str(removal_time + detect_time) + 's')

        return image, True




import os
import numpy as np
np.random.seed(1234)
import shutil
if __name__ == '__main__':
    image_path = 'E:\\ImageData\\color_data_detected_splited\\train'
    output_path = 'E:\\Source\\LicensePlateRecognition\\data\\output_test_7'
    image_sampling_path = 'E:\\Source\\LicensePlateRecognition\\data\\test'

    plate_localizor = LicensePlateRemoval()

    # threshold = 0.45
    # plate_localizor.set_threshold(threshold)
    images = list()
    list_dir = os.listdir(image_path)
    for dir in list_dir:
        full_dir_path = os.path.join(image_path, dir)
        list_file = os.listdir(full_dir_path)
        image_sampling = np.random.choice(list_file, 100)
        images += [os.path.join(full_dir_path, img) for img in image_sampling]
        print()

    # for image in images:
    #     shutil.copyfile(image, image_sampling_path)
    #
    for i, image in enumerate(images):
        # new_file_path = file_path.replace('-', '_')
        # os.rename(file_path, new_file_path)
        im = cv.imread(str(image))
        print('Remove image: ' + str(i))
        if im is None:
            continue
        try:
            out_im, result = plate_localizor.image_remove(im)
            if result:
                cv.imwrite(output_path + '\\detected\\' + str(i) +'.jpg', out_im)
            else:
                cv.imwrite(output_path + '\\not_detected\\' + str(i) +'.jpg', out_im)

        except Exception as e:
            print(e)