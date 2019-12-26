#!usr/bin/env python3
# -*- coding:utf-8 -*-

"""base function define"""

__author__ = "huzhenhong@2019-11-19"

import os
import cv2 as cv
import numpy as np

def cv_show(win_name, show_img):
    """
    opencv 显示图片
    :param win_name: 窗口名称
    :param show_img: 待显示图片
    :return: 无
    """
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, show_img)

def cv_resize(src_img, width=None, height=None, inter=cv.INTER_AREA):
    """
    调整图像尺寸
    :param src_img: 原图
    :param width: 调整后的宽
    :param height: 调整后的高
    :param inter: 插值方法
    :return: 调整后的图像
    """
    if width is None and height is None:
        return src_img  # 不予变换

    h, w = src_img.shape[:2]

    if width is None:
        rate = float(h) / height
        return cv.resize(src_img, (int(w / rate + 0.5), height), interpolation=inter)
    elif height is None:
        rate = float(w) / width
        return cv.resize(src_img, (width, int(h / rate + 0.5)), interpolation=inter)
    else:
        return cv.resize(src_img, (width, height), interpolation=inter)

def cv_rotate(src_img, angle, scale = 1):
    """

    :param src_img:
    :param angle:
    :return:
    """
    rows, cols, channels = src_img.shape
    M = cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), int(angle), scale)
    rotated_img = cv.warpAffine(src_img, M, (cols, rows))
    return rotated_img

def cv_get_roi_by_contour(contour, src_img, border=10, horizontally=True):
    """

    :param contour:
    :param src_img:
    :param border:
    :return:
    """
    # 最小外接矩形
    min_bounding_rect = cv.minAreaRect(contour)
    width, height = min_bounding_rect[1]
    center = min_bounding_rect[0]
    angle = min_bounding_rect[2]
    # draw_img = cv.drawContours(cv.cvtColor(src_img, cv.COLOR_GRAY2BGR), [np.int32(cv.boxPoints(min_bounding_rect))], 0, (0, 0, 255), 1)
    # cv_show('contours', draw_img)

    # 获取四个顶点的坐标
    src_points = np.float32(cv.boxPoints(min_bounding_rect))

    if horizontally is True:
        if width >= height:

            # 扣外面一圈
            src_points[0] = src_points[0] + [-border, border]
            src_points[1] = src_points[1] + [-border, -border]
            src_points[2] = src_points[2] + [border, -border]
            src_points[3] = src_points[3] + [border, border]


            dst_points = np.array([[0, height - 1],
                                   [0, 0],
                                   [width - 1, 0],
                                   [width - 1, height - 1]],
                                  dtype='float32')
        else:
            width, height = height, width

            # 扣外面一圈
            src_points[0] = src_points[0] + [border, border]
            src_points[1] = src_points[1] + [-border, border]
            src_points[2] = src_points[2] + [-border, -border]
            src_points[3] = src_points[3] + [border, -border]

            dst_points = np.array([[width - 1, height - 1],
                                   [0, height - 1],
                                   [0, 0],
                                   [width - 1, 0]],
                                  dtype='float32')
    else:
        if width >= height:
            width, height = height, width
            # 扣外面一圈
            src_points[0] = src_points[0] + [border, border]
            src_points[1] = src_points[1] + [-border, border]
            src_points[2] = src_points[2] + [-border, -border]
            src_points[3] = src_points[3] + [border, -border]

            dst_points = np.array([[width - 1, height - 1],
                                   [0, height - 1],
                                   [0, 0],
                                   [width - 1, 0]],
                                  dtype='float32')
        else:
            # 扣外面一圈
            src_points[0] = src_points[0] + [-border, border]
            src_points[1] = src_points[1] + [-border, -border]
            src_points[2] = src_points[2] + [border, -border]
            src_points[3] = src_points[3] + [border, border]


            dst_points = np.array([[0, height - 1],
                                   [0, 0],
                                   [width - 1, 0],
                                   [width - 1, height - 1]],
                                  dtype='float32')

    # 计算透视变换矩阵
    M = cv.getPerspectiveTransform(src_points, dst_points)
    warped_img = cv.warpPerspective(src_img, M, (int(width), int(height)))

    return warped_img

def rename_imgs(directory_name, renamed_directory_name, img_name="", img_nums=1000):
    """
    对指定目录下的所有图片按img_name进行重命名
    :param directory_name:
    :param renamed_directory_name:
    :param img_name:
    :param img_nums:
    :return:
    """
    list_of_file = os.listdir(directory_name)
    index = 1
    for filename in list_of_file:
        img = cv.imread(directory_name + '/'+ filename)
        if img is not  None:
            # os.remove(directory_name + "/" + filename)
            if 0 == len(img_name):
                # 命名从数字1开始递增
                cv.imwrite(renamed_directory_name + '/'+ str(index) + ".jpg", img)
                index+=1
            else:
                # 命名从img_name开始递增
                cv.imwrite(renamed_directory_name + img_name + '/'+ str(index) + ".jpg", img)
                index += 1

        if index > img_nums:
            break


def read_directory_imgs(directory_name, nums=100):
    """
    按顺序读取指定目录下的nums张图片
    :param directory_name:
    :param nums:
    :return:
    """
    list_of_img = []
    count = 0
    for filename in os.listdir(directory_name):
        count += 1
        if count > nums:
            break

        img = cv.imread(directory_name + "/" + filename, cv.IMREAD_GRAYSCALE)
        list_of_img.append(img)

    return  list_of_img

