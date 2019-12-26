# !usr/bin/python
# -*- coding:utf-8 -*-

"""A load svm train function"""

__author__ = "huzhenhong-2019-12-25"

import os
import numpy as np
import cv2 as cv
import base_function as bf
from sklearn import metrics

# rand1 = np.array([[155,48],[159,50],
#                   [164,53],[168,56]
#                   ,[172,60]])
# rand2 = np.array([[152,53],[156,55],
#                   [160,56],[172,64],
#                   [176,65]])
# #lable
# lable = np.array([[0],[0],[0],[0],[0],
#                   [1],[1],[1],[1],[1]])
# #data
# data = np.vstack((rand1,rand2)) #合并到一起
# data = np.array(data,dtype='float32')
#
# #训练
# svm = cv.ml.SVM_create()
# # ml 机器学习模块 SCM_create() 创建
# svm.setType(cv.ml.SVM_C_SVC) # svm type
# svm.setKernel(cv.ml.SVM_LINEAR) # line #线性分类器
# svm.setC(0.01)
# # 进行训练
# result= svm.train(data,cv.ml.ROW_SAMPLE,lable)
# print(1)

def get_all_renamed_imgs(directory):
    """

    :param directory:
    :return:
    """
    list_of_imgs = []
    # 遍历所有图片
    # list_of_directory = os.listdir(directory)
    # for folder in list_of_directory:
    for i in range(10):
        # if "renamed" == folder:
        #     continue

        directory_name = "./DigaitalTrainData/" + str(i)
        renamed_directory_name = "./DigaitalTrainData/renamed/" + str(i)
        if os.path.exists(renamed_directory_name) is not True:
            os.makedirs(renamed_directory_name)

        # 对图像进行重命名
        bf.rename_imgs(directory_name,
                       renamed_directory_name=renamed_directory_name,
                       img_nums=400)

        # 加载重命名后的所有图像
        list_of_imgs.append(bf.read_directory_imgs(renamed_directory_name, 400))

    return list_of_imgs


list_of_imgs = get_all_renamed_imgs("./DigaitalTrainData/")
# 切分为训练集和测试集
if list_of_imgs is not None:
    train_imgs= [imgs[:300] for imgs in list_of_imgs]
    test_imgs = [imgs[300:] for imgs in list_of_imgs]

    train_data = np.array(train_imgs, dtype='float32').reshape(3000, -1)
    train_label = np.repeat(np.arange(10), 300).reshape(3000, -1)

    test_data = np.array(test_imgs, dtype='float32').reshape(1000, -1)
    test_label = np.repeat(np.arange(10), 100).reshape(1000, -1)

    # svm 训练
    svm = cv.ml.SVM_create()
    svm.setKernel(cv.ml.SVM_LINEAR)
    svm.setType(cv.ml.SVM_C_SVC)
    svm.setC(2.67)
    svm.setGamma(5.383)

    result = svm.train(train_data, cv.ml.ROW_SAMPLE, train_label)
    svm.save('svm_data.dat')
    print('done')

    svm2 = cv.ml.SVM_load('svm_data.dat')
    _, y_pred = svm2.predict(test_data)

    print(metrics.accuracy_score(test_label, y_pred))