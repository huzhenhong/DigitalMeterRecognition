
import cv2 as cv
import numpy as np
import base_function as bf

output = []

# 加载模板
template_imgs = []
for i in range(10):
    img = cv.imread('template/' + str(i) + '.jpg', cv.IMREAD_GRAYSCALE)
    template_imgs.append(img)

# 加载待识别图像
recognition_imgs = []
for i in range(4):
    img = cv.imread('num_img_' + str(i+1) + '.jpg', cv.IMREAD_GRAYSCALE)
    recognition_imgs.append(img)

# # 模板匹配
# methods = [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED, cv.TM_CCOEFF, cv.TM_CCOEFF_NORMED, cv.TM_CCORR, cv.TM_CCORR_NORMED]
#
# for method in methods:
#     for i, recognition_img in enumerate(recognition_imgs):
#         scores = []
#         for template_img in template_imgs:
#             res= cv.matchTemplate(recognition_img, template_img, method)
#             minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(res)
#
#             if cv.TM_SQDIFF == method or cv.TM_SQDIFF_NORMED ==method:
#                 scores.append(minVal)
#                 output.append(np.argmin(scores))
#                 bf.cv_show(str(method) + 'recognition_template_' + str(i),
#                            np.hstack((recognition_img, template_imgs[np.argmin(scores)])))
#             else:
#                 scores.append(maxVal)
#                 output.append(np.argmax(scores))
#                 bf.cv_show(str(method) + 'recognition_template_' + str(i),
#                            np.hstack((recognition_img, template_imgs[np.argmax(scores)])))

# 作差法
for i, recognition_img in enumerate(recognition_imgs):
    scores = []
    for template_img in template_imgs:
        res = cv.bitwise_and(recognition_img, template_img)
        cv.countNonZero(res)
        scores.append(cv.countNonZero(res))
    output.append(np.argmax(scores))
    bf.cv_show('recognition_template_' + str(i), np.hstack((recognition_img, template_imgs[np.argmax(scores)])))



print(output)
cv.waitKey()
cv.destroyAllWindows()


