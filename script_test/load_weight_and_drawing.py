import Config
from LicensePlateTrainer import LicensePlateTrainer
import numpy as np

Config.Config('local')
trainer = LicensePlateTrainer()
YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))

image_data_train, boxes_train, detectors_mask_train, matching_true_boxes_train \
    = trainer.train_reader.__getitem__(0)
image_data_drawing = image_data_train[:200]
trainer.draw(trainer.model_body, trainer.class_name, YOLO_ANCHORS, image_data_drawing)