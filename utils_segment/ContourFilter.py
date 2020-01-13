import numpy as np
import cv2
import utils_segment.config as config


def filter_by_box_size(contours, image):
    if len(contours) == 0:
        return contours

    #calculate centeroids
    centeroid = np.array([0, 0])
    x_ys = []

    for contour in contours:
        min_rect = cv2.minAreaRect(contour)
        x_y, w_h , a = min_rect
        x_ys.append(x_y)
        centeroid = centeroid + np.array(x_y)
    centeroid /= len(contours)

    max_distance = 0
    min_distance = 10000
    mean_distance = 0
    distances = [];
    for x_y in x_ys:
        distance = np.linalg.norm(centeroid - x_y)
        distances.append(distance)
        if(max_distance < distance):
            max_distance = distance
        if(min_distance > distance):
            min_distance = distance
    # distances = np.array(distances)
    distances = sorted(distances)
    if(len(distances) > 0):
        mean_distance = distances[int(len(distances) / 2)]
    else:
        return []

    out_contours = []
    temp_contours = []
    for i in range(len(contours)):
        # im_clone = cv2.drawContours(np.copy(image), contours, i, (0, 255, 0), 3)
        # cv2.imshow('img_vs_contours', im_clone)

        distance = np.linalg.norm(centeroid - x_ys[i])
        distance_threshold = 3 * mean_distance - 2 * min_distance
        if distance <= distance_threshold:
            temp_contours.append(contours[i])

    if(len(temp_contours) <= 0):
        return []
    # calculate mean
    mean = 0
    heights = []
    for contour in temp_contours:
        min_rect = cv2.minAreaRect(contour)
        x_y, w_h , a = min_rect
        mean += max(w_h)
        heights.append(max(w_h))
    mean /= len(temp_contours)

    for i in range(len(temp_contours)):
        # im_clone = cv2.drawContours(np.copy(image), temp_contours, i, (0, 255, 0), 3)
        # cv2.imshow('img_vs_contours', im_clone)

        score = abs(mean - heights[i])/heights[i]
        if score <= 0.1:
            out_contours.append(temp_contours[i])

    return out_contours

def filter_by_box_size_bk(contours):
    if len(contours) == 0:
        return contours

    #calculate centeroids
    centeroid = np.array([0, 0])

    # calculate mean
    mean = 0
    heights = []

    x_ys = []

    for contour in contours:
        min_rect = cv2.minAreaRect(contour)
        x_y, w_h , a = min_rect
        mean += max(w_h)
        heights.append(max(w_h))
        x_ys.append(x_y)
        centeroid = centeroid + np.array(x_y)
    mean /= len(contours)
    centeroid /= len(contours)

    max_distance = 0
    min_distance = 10000
    for x_y in x_ys:
        distance = np.linalg.norm(centeroid - x_y)
        if(max_distance < distance):
            max_distance = distance
        if(min_distance > distance):
            min_distance = distance

    out_contours = []
    for i in range(len(contours)):
        score = abs(mean - heights[i])/heights[i]
        distance = np.linalg.norm(centeroid - x_ys[i])
        if score < 0.1 and (max_distance <= config.max_distance_centeroid_threshold or
                                (max_distance > config.max_distance_centeroid_threshold
                                 and (distance - min_distance) < config.ratio_delta_distance_with_min_max_threshold * (max_distance - min_distance))):
            out_contours.append(contours[i])

    return out_contours



def filter_by_position(contours):
    return 0
