import pcl
import pcl.pcl_visualization
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from time import *

from Undergraduate_Design import Optimal_Rect



# 求两点间距离
def cal_distance(p1, p2):
    distance = math.pow((p1[0] - p2[0]), 2) + math.pow((p1[1] - p2[1]), 2)
    distance = math.sqrt(distance)
    return distance


# 选定数据范围,保留x方向[-20,20],ｙ方向[-20,20],z方向[-3,-0.8]的点
def mask_points_by_range(points, limit_range=[-20, -20, -3, 20, 20, -0.8]):
    mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) & \
           (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4]) & \
           (points[:, 2] >= limit_range[2]) & (points[:, 2] <= limit_range[5])
    return mask


# 以数组的形式载入点云数据
def load_points(path):
    raw_points = pcl.load(path)
    cloud_points = raw_points.to_array()
    cloud_points = np.around(cloud_points, 2)  # 保留两位小数
    return cloud_points


# 利用PCL库可视化原始点云图
def visualize_point_clouds(path):
    cloud = pcl.load(path)
    viewer = pcl.pcl_visualization.PCLVisualizering('cloud')  # 创建viewer对象
    viewer.AddPointCloud(cloud)
    viewer.SpinOnce()


# 三维数据降采样
def voxel_filter(point_cloud, leaf_size):
    # 获取数据边界,建立体素
    x_max, y_max, z_max = np.max(point_cloud, axis=0)
    x_min, y_min, z_min = np.min(point_cloud, axis=0)
    D_x = (x_max - x_min) / leaf_size
    D_y = (y_max - y_min) / leaf_size
    # D_z = (z_max - z_min) / leaf_size
    # 获取点在voxel_grid中的位置
    h = []
    for i in range(len(point_cloud)):
        h_x = np.floor((point_cloud[i][0] - x_min) / leaf_size)
        h_y = np.floor((point_cloud[i][1] - y_min) / leaf_size)
        h_z = np.floor((point_cloud[i][2] - z_min) / leaf_size)
        H = h_x + h_y * D_x + h_z * D_x * D_y
        h.append(H)

    # 对所有点根据其所在格子位置进行排序
    h = np.array(h)
    voxel_index = np.argsort(h)  # 数据从小到大排序,返回索引
    h_sort = h[voxel_index]  # 将数据所在网格索引从小到到排列

    # random
    filtered_points = []
    index_begin = 0
    for i in range(len(voxel_index) - 1):
        # 找出所有在同一个体素内的点
        if h_sort[i] == h_sort[i + 1]:
            continue
        else:
            point_index = voxel_index[index_begin:(i + 1)]
            random_index = np.random.choice(point_index)
            random_choice = point_cloud[random_index]  # 随机选取一个点作为保留点
            filtered_points.append(random_choice)
            index_begin = i
    filtered_points = np.array(filtered_points)
    return filtered_points


# 二维数据降采样
def voxel_filter_2D(point_cloud, leaf_size):
    # 获取数据边界,建立体素
    x_max, y_max = np.max(point_cloud, axis=0)
    x_min, y_min = np.min(point_cloud, axis=0)
    D_x = (x_max - x_min) / leaf_size
    D_y = (y_max - y_min) / leaf_size

    # 获取点在voxel_grid中的位置
    h = []
    for i in range(len(point_cloud)):
        h_x = np.floor((point_cloud[i][0] - x_min) / leaf_size)
        h_y = np.floor((point_cloud[i][1] - y_min) / leaf_size)
        H = h_x + h_y * D_x
        h.append(H)

    # 对所有点根据其所在格子位置进行排序
    h = np.array(h)
    voxel_index = np.argsort(h)  # 数据从小到大排序,返回索引
    h_sort = h[voxel_index]  # 将数据所在网格索引从小到到排列

    # random
    filtered_points = []
    index_begin = 0
    count = 1
    for i in range(len(voxel_index) - 1):
        # 找出所有在同一个体素内的点
        if h_sort[i] == h_sort[i + 1]:
            count = count + 1
            continue
        else:
            if count > 1:
                point_index = voxel_index[index_begin:(i + 1)]
                random_index = np.random.choice(point_index)
                random_choice = point_cloud[random_index]  # 随机选取一个点作为保留点
                filtered_points.append(random_choice)
            index_begin = i
            count = 1

    filtered_points = np.array(filtered_points)
    return filtered_points


# 将样本集按照labels中的标签值重新排序，得到按照类簇排列好的输出结果
def labels_to_original(labels, data):
    number_label = []
    for m in np.unique(labels):
        if m != -1:
            number_label.append(m)

    result = []
    for piece in range(len(number_label)):
        result.append([])

    for j in range(len(labels)):
        if labels[j] != -1:
            index = number_label.index(labels[j])
            result[index].append([data[j][0], data[j][1]])
    return result


if __name__ == '__main__':
    begin = time()

    file_path = '/home/yyh/yyh_self/pycharm/Point_Cloud_Processed/61.pcd'

    # 载入数据
    Point_Data = load_points(file_path)
    print('原始数据总数：', len(Point_Data))

    # 降采样
    Point_Data = voxel_filter(Point_Data, 0.1)
    print('降采样数据总数: ', len(Point_Data))

    # 数据剪切
    mask = mask_points_by_range(Point_Data, [-15, -20, -2.8, 15, 20, -0.5])
    Point_Data = Point_Data[mask]
    print('限定范围后数据总数: ', len(Point_Data))

    # 将数据投影到二维
    Data_2D = Point_Data[:, 0:len(Point_Data[0]) - 1]

    # DBSCAN聚类
    db = DBSCAN(eps=0.55, min_samples=5)
    db.fit_predict(Data_2D)

    # 获取分类标签
    cluster_labels = db.labels_  # 所有标签(包含噪点: -1)
    print('分类核心点: ', len(db.core_sample_indices_))
    # 将点数太少的类别归类为噪点
    for i in np.unique(cluster_labels):
        if i == -1:
            print('噪点: ', np.count_nonzero(cluster_labels == i))
        if i != -1:
            if np.count_nonzero(cluster_labels == i) <= 50:
                cluster_labels[cluster_labels == i] = -1

    # 分类数目(排除噪点)
    n_clusters = len(np.unique(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print('分类数目:', n_clusters)

    # 将数据按分类结果排序(排除噪点)
    cluster_result = labels_to_original(cluster_labels, Data_2D)

    # 绘制最终结果图
    fig_3 = plt.figure(3)
    axis_3 = fig_3.add_subplot(1, 1, 1)

    for k in cluster_result:
        k = np.asarray(k)
        axis_3.scatter(k[:, 0], k[:, 1], s=1, marker='.')

        # 方法二:接近程度标准
        filters = voxel_filter_2D(k, 0.07)
        boundary = Optimal_Rect.cal_result(filters)

        # 添加车辆长,宽,长宽比限制
        dis_1 = cal_distance(boundary[0], boundary[1])
        dis_2 = cal_distance(boundary[0], boundary[2])
        max_dis = max(dis_1, dis_2)
        min_dis = min(dis_1, dis_2)
        if (max_dis <= 5.7) and (max_dis >= 1.7) and (min_dis <= 2.8) and (min_dis >= 1.1):
            if max_dis / min_dis >= 1.1:
                axis_3.plot([boundary[2][0], boundary[0][0]], [boundary[2][1], boundary[0][1]], c='b')
                axis_3.plot([boundary[2][0], boundary[3][0]], [boundary[2][1], boundary[3][1]], c='b')
                axis_3.plot([boundary[3][0], boundary[1][0]], [boundary[3][1], boundary[1][1]], c='b')
                axis_3.plot([boundary[1][0], boundary[0][0]], [boundary[1][1], boundary[0][1]], c='b')

    axis_3.set_xlabel('X / m')
    axis_3.set_ylabel('Y / m')
    axis_3.set_title('Closeness with Length/Width restriction')

    end = time()
    print('总程序运行时间:', end-begin, '秒')

    plt.show()




