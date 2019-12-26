#!usr/bin/env python3
# -*- coding:utf-8 -*-

"""提取数字区域"""

__author__ = "huzhenhong@2019-11-26"

import cv2 as cv
import numpy as np
import base_function as bf


def get_nums_by_color(src_img):
    """

    :param src_img:
    :return:
    """
    # 提取黄色区域
    hsv_img = cv.cvtColor(src_img, cv.COLOR_BGR2HSV)
    lower_yellow = np.array([26, 43, 46])
    upper_yellow = np.array([34, 255, 255])
    mask = cv.inRange(hsv_img, lower_yellow, upper_yellow)
    inRange_img = cv.bitwise_and(src_img, src_img, mask=mask)
    bf.cv_show('inRange_img', np.hstack((src_img, inRange_img)))

    # 图片预处理
    gray_img = cv.cvtColor(inRange_img, cv.COLOR_BGR2GRAY)
    _, binary_img = cv.threshold(gray_img, 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY)
    morph_rect = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    morph_open = cv.morphologyEx(binary_img, cv.MORPH_OPEN, morph_rect, iterations=1)
    morph_close = cv.morphologyEx(morph_open, cv.MORPH_CLOSE, morph_rect, iterations=3)
    bf.cv_show('binary_img && morph_open && morph_close', np.hstack((binary_img, morph_open, morph_close)))

    # 查找轮廓
    _, contours, hierachy = cv.findContours(morph_close, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) < 5:
        return

    # 获取面积最大的五个轮廓
    valid_contours = sorted(contours, key=lambda c: cv.contourArea(c))[:5]
    # 对轮廓按质心横坐标进行排序
    sorted_contours =sorted(valid_contours, key=lambda c: int(cv.moments(c)['m10'] / cv.moments(c)['m00']))
    # 计算每个轮廓的面积
    contours_area = [cv.contourArea(c) for c in sorted_contours]
    # 找到面积最小的，记录其索引极为小数点位置
    decimal_point_index = np.argmin(contours_area)
    # 移除小数点
    sorted_contours.pop(decimal_point_index)
    decimal_point_index = 4 - decimal_point_index
    print(decimal_point_index)

    # 绘制数字轮廓
    draw_img = cv.drawContours(src_img.copy(), sorted_contours, -1, (0, 0, 255), 1)
    bf.cv_show('nums_contour', draw_img)
    # 提取数字ROI
    num_imgs = []
    for i, c in enumerate(sorted_contours):
        num_img = bf.cv_get_roi_by_contour(c, binary_img, horizontally=False)
        num_imgs.append(bf.cv_resize(num_img, width=64, height=128))
        # bf.cv_show('num_img_' + str(i), num_img)

    return num_imgs, decimal_point_index


def get_nums_by_contour(src_img):
    """

    :param src_img:
    :return:
    """
    # 获取仪表图片
    meter_img = get_meter_roi(src_img)
    bf.cv_show('meter_img', meter_img)

    # 获取数字区域
    nums_roi = get_nums_roi(meter_img)

    return get_num(nums_roi)

def get_meter_roi(src_img):
    """

    :param src_img:
    :return:
    """
    bf.cv_show('src_img', src_img)

    # 灰度图像
    gray_img = cv.cvtColor(src_img, cv.COLOR_RGB2GRAY)

    # 高斯滤波
    gauss_img = cv.GaussianBlur(gray_img, (5, 5), 0)

    # # 自适应均衡化
    # clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # clahe_img = clahe.apply(gauss_img)
    bf.cv_show('gray_img && gauss_img', np.hstack((gray_img, gauss_img)))

    # 二值化
    _, binary_img = cv.threshold(gauss_img, 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY_INV)

    # 形态学处理
    morph_rect = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
    morph_open_4 = cv.morphologyEx(binary_img, cv.MORPH_OPEN, morph_rect, iterations=4)
    bf.cv_show('binary_img && morph_open_4' , np.hstack((binary_img, morph_open_4)))

    # 轮廓查找
    ref_img, contours, hierachy = cv.findContours(morph_open_4, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    sorted_contours = sorted(contours, key=lambda c : cv.contourArea(c), reverse=True)
    for c in sorted_contours:
        print(cv.contourArea(c))

    draw_img = cv.drawContours(src_img.copy(), sorted_contours, 0, (0, 0, 255), 1)
    bf.cv_show('contours', draw_img)

    # 扣取仪表
    meter_img = bf.cv_get_roi_by_contour(sorted_contours[0], binary_img, -20)
    # 图片取反
    meter_img = cv.bitwise_not(meter_img)

    return meter_img

def get_nums_roi(meter_img):
    """

    :param src_img:
    :return:
    """
    # 连接数字
    morph_rect_15x5 = cv.getStructuringElement(cv.MORPH_RECT, (15, 5))
    morph_close_4 = cv.morphologyEx(meter_img, cv.MORPH_CLOSE, morph_rect_15x5, iterations=4)
    bf.cv_show('morph_close_4', morph_close_4)

    # 轮廓查找
    ref_img, contours, hierachy = cv.findContours(morph_close_4, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=lambda c: cv.contourArea(c), reverse=True)
    if 0 == len(sorted_contours):
        return

    draw_img = cv.drawContours(cv.cvtColor(meter_img, cv.COLOR_GRAY2BGR), sorted_contours, 0, (0, 0, 255), 1)
    bf.cv_show('nums_contours', draw_img)

    nums_roi = bf.cv_get_roi_by_contour(sorted_contours[0], meter_img)

    return nums_roi


def get_num(nums_roi):
    """

    :param nums_roi:
    :return:
    """
    num_imgs = []

    # 轮廓查找，面积从大到小排列
    ref_img, contours, hierachy = cv.findContours(nums_roi, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # 获取面积最大的五个轮廓
    valid_contours = sorted(contours, key=lambda c: cv.contourArea(c))[:5]
    # 对轮廓按质心横坐标进行排序
    sorted_contours =sorted(valid_contours, key=lambda c: int(cv.moments(c)['m10'] / cv.moments(c)['m00']))
    # 计算每个轮廓的面积
    contours_area = [cv.contourArea(c) for c in sorted_contours]
    # 找到面积最小的，记录其索引极为小数点位置
    decimal_point_index = np.argmin(contours_area)
    # 移除小数点
    sorted_contours.pop(decimal_point_index)
    decimal_point_index = 4 - decimal_point_index
    print(decimal_point_index)

    # 绘制数字轮廓
    draw_img = cv.drawContours(cv.cvtColor(nums_roi, cv.COLOR_GRAY2BGR), sorted_contours, -1, (0, 0, 255), 1)
    bf.cv_show('nums_contour', draw_img)
    # 提取数字ROI
    num_imgs = []
    for i, c in enumerate(sorted_contours):
        num_img = bf.cv_get_roi_by_contour(c, nums_roi, horizontally=False)
        num_imgs.append(bf.cv_resize(num_img, width=64, height=128))
        # bf.cv_show('num_img_' + str(i), num_img)

    return num_imgs, decimal_point_index


