
import cv2 as cv
import numpy as np
import base_function as bf

def get_nums(src_img):
    """

    :param src_img:
    :return:
    """
    # 灰度图像
    gray_img = cv.cvtColor(src_img, cv.COLOR_RGB2GRAY)

    # 高斯滤波
    gauss_img = cv.GaussianBlur(gray_img, (5, 5), 0)

    # # 自适应均衡化
    # clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # clahe_img = clahe.apply(gauss_img)
    bf.cv_show('gray_img && gauss_img', np.hstack((gray_img, gauss_img)))

    # 二值化
    _, binary_img = cv.threshold(gauss_img, 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY)

    # 形态学处理
    morph_rect = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    morph_open = cv.morphologyEx(binary_img, cv.MORPH_OPEN, morph_rect, iterations=2)
    morph_close = cv.morphologyEx(morph_open, cv.MORPH_CLOSE, morph_rect, iterations=2)
    bf.cv_show('binary_img && morph_close && morph_close' , np.hstack((binary_img, morph_open, morph_close)))

    # 连接数字
    morph_rect_3x12 = cv.getStructuringElement(cv.MORPH_RECT, (25, 5))
    morph_close_1 = cv.morphologyEx(morph_close, cv.MORPH_CLOSE, morph_rect_3x12, iterations=3)
    bf.cv_show('morph_close_1', morph_close_1)

    # 轮廓查找
    ref_img, contours, hierachy = cv.findContours(morph_close_1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    sorted_contours = sorted(contours, key=lambda c: cv.contourArea(c), reverse=True)
    # 获取数字区域
    draw_img = cv.drawContours(src_img.copy(), sorted_contours, 0, (0, 0, 255), 1)
    bf.cv_show('contours', draw_img)

    min_bounding_rect = cv.minAreaRect(sorted_contours[0])
    center = min_bounding_rect[0]
    width, height = min_bounding_rect[1]
    angle = min_bounding_rect[2]

    # 获取四个顶点的坐标
    src_points = np.float32(cv.boxPoints(min_bounding_rect))
    # min_bounding_rect_img = cv.circle(src_img.copy(), tuple(src_points.astype('int32')), 2, (0, 255, 0), 1)
    for point in src_points:
        min_bounding_rect_img = cv.circle(src_img, tuple(point.astype('int32')), 2, (0, 255, 0), 1)

        bf.cv_show('min_bounding_rect_img', min_bounding_rect_img)

    # for c in sorted_contours:
    #     print(cv.contourArea(c))
    #
    # # valid_contours = [c for c in contours if 500 < cv.contourArea(c) < 10000 ]
    # valid_contours = sorted_contours[:4]
    # draw_img = cv.drawContours(src_img.copy(), valid_contours, -1, (0, 0, 255), 1)
    # bf.cv_show('contours', draw_img)
    #
    # boudings = [cv.boundingRect(c) for c in valid_contours]
    # sorted_bouding = sorted(boudings, key=lambda b:b[0])
    #
    # for index, bouding in enumerate(sorted_bouding):
    #     # 垂直外接矩形
    #     x, y, w, h = bouding
    #
    #     src_points = np.array([[x, y],
    #                            [x + w, y],
    #                            [x + w, y + h],
    #                            [x, y + h]],
    #                             dtype='float32')
    #
    #
    #     dst_points = np.array([[0, 0],
    #                            [w - 1, 0],
    #                            [w - 1, h - 1],
    #                            [0, h - 1]],
    #                             dtype='float32')
    #
    #
    #     # 计算透视变换矩阵
    #     M = cv.getPerspectiveTransform(src_points, dst_points)
    #     num_img = cv.warpPerspective(morph_close, M, (int(w), int(h)))
    #     resize_img = bf.cv_resize(num_img, width=64, height=128)
    #     cv.imwrite('num_img_' + str(index) + '.jpg', resize_img)
    #     bf.cv_show('num_img_' + str(index) + '.jpg', resize_img)
    #     print(resize_img.shape)


src_img = cv.imread('warped_img.jpg')
if src_img is not None:
    bf.cv_show('src_img', src_img)
    get_nums(src_img)

    cv.waitKey(0)
    cv.destroyAllWindows()