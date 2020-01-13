import utils_segment.config as config
import cv2
import numpy as np
import math
import sys
import urllib.request as ur
from scipy.spatial import distance
import urllib
from skimage import io


def expand_rect(original, expand_x_pixels, expand_y_pixels, max_x, max_y):
    half_x = round(expand_x_pixels/2.0)
    half_y = round(expand_y_pixels/2.0)
    ex = original[0] - half_x
    ey = original[1] - half_y
    ew = original[2] + expand_x_pixels
    eh = original[3] + expand_y_pixels

    ex = min(max(ex, 0), max_x)
    ey = min(max(ey, 0), max_y)

    if ex + ew > max_x:
        ew = max_x - ex
    if ey + eh > max_y:
        eh = max_y - ey
    return ex, ey, ew, eh


def equalize_brightness(original_img):
    # Divide the image by its morphologically closed counterpart
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    closed = cv2.morphologyEx(original_img, cv2.MORPH_CLOSE, kernel)
    img = np.float32(original_img)
    img = cv2.divide(img, closed, scale=1, dtype=cv2.CV_32FC1)
    img = cv2.normalize(img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    img = np.uint8(img)
    return img


def fill_mask(img, mask, color):
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            m = mask[row, col]
            if m > 0:
                for z in range(3):
                    prev_val = img[row, col, z]
                    img[row, col, z] = int(color[z]) | prev_val


def draw_x(img, rect, color, thickness):
    x, y, w, h = rect
    cv2.line(img, (x, y), (x + w, y + h), color, thickness)
    cv2.line(img, (x, y + h), (x + w, y), color, thickness)


def distance_between_points(p1, p2):
    a_squared = (p2[0] - p1[0])**2
    b_squared = (p2[1] - p1[1])**2
    return np.sqrt(a_squared + b_squared)


def angle_between_points(p1, p2):
    delta_y = p2[1] - p1[1]
    delta_x = p2[0] - p1[0]
    return math.atan2(delta_y, delta_x) * (180/math.pi)


def get_size_maintaining_aspect(input, max_width, max_height):
    aspect = input.shape[1] / input.shape[0]
    if max_width > aspect * max_height:
        return max_height * aspect, max_height
    else:
        return max_width, max_width/aspect


def find_closest_point(polygon_points, num_points, position):
    closest_point_index = 0
    smallest_distance = sys.maxsize
    for i in range(num_points):
        pos = int(polygon_points[i][0]), int(polygon_points[i][1])
        distance = distance_between_points(pos, position)
        if distance < smallest_distance:
            smallest_distance = distance
            closest_point_index = i
    return int(polygon_points[closest_point_index][0]), int(polygon_points[closest_point_index][1])


def sort_polygon_points(polygon_points, surrounding_image):
    points = list()
    points.append(find_closest_point(polygon_points, 4, (0, 0)))
    points.append(find_closest_point(polygon_points, 4, (surrounding_image[0], 0)))
    points.append(find_closest_point(polygon_points, 4, (surrounding_image[0], surrounding_image[1])))
    points.append(find_closest_point(polygon_points, 4, (0, surrounding_image[1])))
    return points


def get_contour_area_percent_inside_mask(mask, contours, hierarchy, contour_index):
    inner_area = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(inner_area,
                     contours,
                     contour_index,
                     (255, 255, 255),
                     cv2.FILLED,
                     8,
                     hierarchy,
                     2)
    starting_pixels = cv2.countNonZero(inner_area)
    inner_area = cv2.bitwise_and(inner_area, mask)
    ending_pixels = cv2.countNonZero(inner_area)
    return ending_pixels/starting_pixels


def list_to_numpy(x):
    y = np.dstack(x)
    y = np.transpose(y, [0, 2, 1])
    y = y[0]
    return y


def produce_thresholds(im_gray):
    # _, threshold_1 = cv2.threshold(im_gray, 50, 255, cv2.THRESH_BINARY_INV)
    _, threshold_2 = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
    # _, threshold_3 = cv2.threshold(im_gray, 127, 255, cv2.THRESH_BINARY_INV)
    threshold_4 = cv2.adaptiveThreshold(im_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    threshold_5 = cv2.adaptiveThreshold(im_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    threshold_6 = cv2.Canny(im_gray, 150, 200)
    # _, threshold_7 = cv2.threshold(im_gray, 205, 255, cv2.THRESH_BINARY_INV)
    # _, threshold_8 = cv2.threshold(im_gray, 165, 255, cv2.THRESH_BINARY_INV)
    # _, threshold_9 = cv2.threshold(im_gray, 20, 255, cv2.THRESH_BINARY_INV)
    # _, threshold_10 = cv2.threshold(im_gray, 30, 255, cv2.THRESH_BINARY_INV)
    # _, threshold_11 = cv2.threshold(im_gray, 40, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow('threshold_1', threshold_1)
    # cv2.imshow('threshold_2', threshold_2)
    # cv2.imshow('threshold_3', threshold_3)
    # cv2.imshow('threshold_4', threshold_4)
    # cv2.imshow('threshold_5', threshold_5)
    # cv2.imshow('threshold_6', threshold_6)
    # cv2.imshow('threshold_7', threshold_7)
    # cv2.imshow('threshold_8', threshold_8)
    # cv2.imshow('threshold_9', threshold_9)
    # cv2.imshow('threshold_10', threshold_10)
    # cv2.imshow('threshold_11', threshold_11)
    # return threshold_1, threshold_2, threshold_3, threshold_4, threshold_5, threshold_6, threshold_7, threshold_8, threshold_9, threshold_10, threshold_11
    return threshold_2, threshold_4, threshold_5, threshold_6

def produce_thresholds_bk(im_gray):
    _, threshold_1 = cv2.threshold(im_gray, 38, 255, cv2.THRESH_BINARY_INV)
    _, threshold_2 = cv2.threshold(im_gray, 68, 255, cv2.THRESH_BINARY_INV)
    _, threshold_3 = cv2.threshold(im_gray, 98, 255, cv2.THRESH_BINARY_INV)
    _, threshold_4 = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY_INV)
    _, threshold_5 = cv2.threshold(im_gray, 158, 255, cv2.THRESH_BINARY_INV)
    # _, threshold_6 = cv2.threshold(im_gray, 188, 255, cv2.THRESH_BINARY_INV)
    # _, threshold_7 = cv2.threshold(im_gray, 218, 255, cv2.THRESH_BINARY_INV)
    threshold_6 = cv2.adaptiveThreshold(im_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 23, 12)
    threshold_7 = cv2.adaptiveThreshold(im_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 23, 12)
    # threshold_8 = cv2.Canny(im_gray, 100, 150)
    # cv2.imshow('threshold_1', threshold_1)
    # cv2.imshow('threshold_2', threshold_2)
    # cv2.imshow('threshold_3', threshold_3)
    # cv2.imshow('threshold_4', threshold_4)
    # cv2.imshow('threshold_5', threshold_5)
    # cv2.imshow('threshold_6', threshold_6)
    # cv2.imshow('threshold_7', threshold_7)
    # cv2.imshow('threshold_8', threshold_8)
    # cv2.imshow('threshold_9', threshold_9)
    # cv2.imshow('threshold_10', threshold_10)
    # cv2.imshow('threshold_11', threshold_11)
    # cv2.imshow('threshold_12', threshold_12)
    return threshold_1, threshold_2, threshold_3, threshold_4, threshold_5, threshold_6, threshold_7


def inpaint(image, contours, tracking):
    sum_contour = []
    for i in range(len(contours)):
        for j in range(len(contours[i])):
            sum_contour.append(contours[i][j])
    height, width = image.shape[0], image.shape[1]
    mask = np.zeros([height, width, 1], dtype=np.uint8)
    if len(contours) > 0:
        sum_contour = np.stack(sum_contour)
        min_rect = cv2.minAreaRect(sum_contour)
        box = cv2.boxPoints(min_rect)
        box = np.int32([box])

        # cv2.fillPoly(mask, box, (255, 255, 255))
        # dst = cv2.inpaint(image, mask, 10, cv2.INPAINT_TELEA)
        # return dst

        # if check_edge_size_rules(box, width, height, tracking) and check_edge_rules(box, tracking):
        if check_edge_rules(box, tracking):
            cv2.fillPoly(mask, box, (255, 255, 255))
            dst = cv2.inpaint(image, mask, 10, cv2.INPAINT_TELEA)
            result = {}
            result['success'] = True
            result['points'] = box[0].tolist()
            result['msg'] = 'Found'
            result['found'] = True
            return dst, result
        else:
            result = {}
            result['success'] = True
            result['msg'] = 'Not found'
            result['found'] = False
            return image, result
    else:
        result = {}
        result['success'] = True
        result['msg'] = 'Not found'
        result['found'] = False
        return image, result

def check_edge_size_rules(box, width, height, tracking):
    for i in range(len(box)) :
        if len(box[i]) > 3 :
            node = box[i][1];
            node_neighbor_first = box[i][0];
            node_neighbor_second = box[i][2];
            distance_first = distance.euclidean(node_neighbor_first, node);
            distance_second = distance.euclidean(node, node_neighbor_second);
            distance_min = min(distance_first, distance_second);
            distance_max = max(distance_first, distance_second);
            ratio_distance_min_vs_min_size = distance_min / min(width, height)
            ratio_distance_max_vs_max_size = distance_max / max(width, height)
            if tracking:
                print('license plate license\'s width / image\'s width = {}'.format(ratio_distance_min_vs_min_size))
                print('license plate license\'s length / image\'s length = {}'.format(ratio_distance_max_vs_max_size))
            if(ratio_distance_min_vs_min_size > config.ratio_edge_with_image_size_max or ratio_distance_max_vs_max_size > config.ratio_edge_with_image_size_max) :
                return False
    return True

def check_edge_rules(box, tracking):
    for i in range(len(box)) :
        if len(box[i]) > 3 :
            node = box[i][1];
            node_neighbor_first = box[i][0];
            node_neighbor_second = box[i][2];
            distance_first = distance.euclidean(node_neighbor_first, node);
            distance_second = distance.euclidean(node, node_neighbor_second);
            ratio = distance_first / distance_second;
            if ratio > 1 :
                ratio = 1 / ratio
            if tracking:
                print('license plate width / length = {}'.format(ratio))
            if ratio >= config.edge_ratio_min :
                return True
    return False

def url_to_image(url):
    try:
        ur.urlopen(url)
        image = io.imread(url)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image
    except Exception:
        return None


if __name__ == '__main__':
    im_gray = cv2.imread(r'test\must_detect\201801150fdcaa72-b0a-8a17.jpg', 0)
    thresholds = produce_thresholds(im_gray)
    cv2.imshow('im_1', cv2.resize(thresholds[4], dsize=(28, 44), interpolation=cv2.INTER_CUBIC))
    cv2.imshow('im_2', cv2.resize(thresholds[3], dsize=(28, 44), interpolation=cv2.INTER_CUBIC))
    cv2.waitKey(0)