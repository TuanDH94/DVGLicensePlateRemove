import numpy as np
import cv2
import utils_segment.Utility as Utility


def order_points(pts):
    rect = np.zeros((4, 2), dtype='float32')
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_points_transform(image, pts):
    rect = order_points(pts)
    tl, tr, br, bl = rect
    width_a = Utility.distance_between_points(br, bl)
    width_b = Utility.distance_between_points(tr, tl)
    max_width = max(int(width_a), int(width_b))

    height_a = Utility.distance_between_points(tr, br)
    height_b = Utility.distance_between_points(tl, bl)
    max_height = max(int(height_a), int(height_b))

    dst = np.array([[0, 0],
                    [max_width - 1, 0],
                    [max_width - 1, max_height - 1],
                    [0, max_height - 1]], dtype='float32')
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped