import matplotlib.pyplot as plt
import numpy as np
import operator
import math
import random
from time import *


# 根据直线方程求交点
def cal_point(rect):
    # 四条直线:ax+by+c=0
    # 第一条直线:a1=cos(n), b1=sin(n),c1=-min(C1)
    # 第二条直线:a2= -sin(n),b2=cos(n),c2=-min(C2)
    # 第三条直线:a3=cos(n),b3=sin(n),c3=-max(C1)
    # 第四条直线:a4=-sin(n),b4=cos(n),c4=-max(C2)
    points = []
    n = rect[0] * math.pi / 180

    # 第一条直线和第二条直线交点left_bottom
    x = (-rect[5] * math.sin(n) + rect[3] * math.cos(n))
    y = rect[3] * math.sin(n) + rect[5] * math.cos(n)
    left_bottom = [x, y]

    # 第一条直线和第四条直线交点left_top
    x = (-rect[4] * math.sin(n) + rect[3] * math.cos(n))
    y = rect[3] * math.sin(n) + rect[4] * math.cos(n)
    left_top = [x, y]

    # 第三条直线和第二条直线交点right_bottom
    x = -rect[5] * math.sin(n) + rect[2] * math.cos(n)
    y = rect[2] * math.sin(n) + rect[5] * math.cos(n)
    right_bottom = [x, y]

    # 第三条直线和第四条直线交点right_top
    x = (-rect[4] * math.sin(n) + rect[2] * math.cos(n))
    y = rect[2] * math.sin(n) + rect[4] * math.cos(n)
    right_top = [x, y]

    points.append(right_top)
    points.append(right_bottom)
    points.append(left_top)
    points.append(left_bottom)

    return points


# 计算接近程度损失函数
def Loss_Closeness(C1, C2, c1_max, c1_min, c2_max, c2_min):
    score = 0  # 接近程度评分
    dis_range = 0.05  # 点到直线的距离阈值

    D1 = (c1_max - C1) if np.linalg.norm(c1_max - C1) < np.linalg.norm(C1 - c1_min) else (C1 - c1_min)
    D2 = (c2_max - C2) if np.linalg.norm(c2_max - C2) < np.linalg.norm(C2 - c2_min) else (C2 - c2_min)

    for i in range(len(D1)):
        d = max(dis_range, min(float(D1[i]), float(D2[i])))
        score = score + 1 / d
    return score


# 计算接近程度评级标准的矩形四边方向向量
def rotate_vector(raw_data):
    raw_data = np.array(raw_data)
    vectors = []

    for theta in range(0, 91, 3):
        n = theta * math.pi / 180  # 将角度转为弧度
        p1 = math.cos(n)
        p2 = math.sin(n)
        e_1 = [p1, p2]
        e_2 = [-p2, p1]

        # 转换成列矩阵操作
        e_1 = np.mat(e_1).T
        e_2 = np.mat(e_2).T

        # 计算点的投影,保留三位小数
        C1 = np.around(np.dot(raw_data, e_1), 3)
        C2 = np.around(np.dot(raw_data, e_2), 3)

        c1_max = C1.max()
        c1_min = C1.min()
        c2_max = C2.max()
        c2_min = C2.min()

        # 计算接近程度评分
        score = Loss_Closeness(C1, C2, c1_max, c1_min, c2_max, c2_min)

        # 获取方向矩形参数
        vectors.append([theta, score, c1_max, c1_min, c2_max, c2_min])

    return vectors


# 获取以接近程度为评价标准的最优结果
def cal_result(raw_data):
    vector_list = rotate_vector(raw_data)  # 获取方向矩形
    vector_list.sort(key=operator.itemgetter(1), reverse=True)  # 按所选评价标准降序

    edge_point = cal_point(vector_list[0])

    return edge_point


