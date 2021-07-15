# 原理可参考https://zhuanlan.zhihu.com/p/30033898

import os
import cv2
import sys
import math
import config
import collections
import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
from scipy.linalg import lstsq
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares

# 特征提取。输入：图像文件名列表；输出：所有图像的特征点列表、特征描述符列表、颜色列表
def extract_features(image_names):
    # 创建特征检测器
    sift = cv2.xfeatures2d.SIFT_create(0, 3, 0.04, 10)
    key_points_for_all = []
    descriptor_for_all = []
    colors_for_all = []
    for image_name in image_names:
        image = cv2.imread(image_name)
        if image is None:
            continue
        # 用sift算法提取每张图像中的特征点及其描述符
        key_points, descriptor = sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)
        if len(key_points) <= 10:   # 如果提取的特征点不超过10个，则丢弃该图像？
            continue
        key_points_for_all.append(key_points)
        descriptor_for_all.append(descriptor)
        # 提取每一个特征点的颜色
        colors = np.zeros((len(key_points), 3))
        for i, key_point in enumerate(key_points):
            p = key_point.pt
            colors[i] = image[int(p[1])][int(p[0])]         
        colors_for_all.append(colors)
    return np.array(key_points_for_all), np.array(descriptor_for_all), np.array(colors_for_all)

# 两个图像进行特征匹配。输入：两个图像的特征描述符列表；输出：匹配结果
def match_features(query, train):
    # 创建特征匹配器
    bf = cv2.BFMatcher(cv2.NORM_L2)
    # 用KNN近邻算法进行匹配
    knn_matches = bf.knnMatch(query, train, k=2)
    matches = []
    #Apply Lowe's SIFT matching ratio test(MRT)，值得一提的是，这里的匹配没有
    #标准形式，可以根据需求进行改动。
    for m, n in knn_matches:
        if m.distance < config.MRT * n.distance:
            matches.append(m)
    return np.array(matches)

# 对一个特征描述列表中的所有特征，进行两两匹配。输入：特征描述符列表；输出：匹配结果列表（每一项是前后两个图像的特征点匹配结果）
def match_all_features(descriptor_for_all):
    matches_for_all = []
    # 按照图像文件列表次序，相邻的图像进行两两特征点匹配
    for i in range(len(descriptor_for_all) - 1):
        matches = match_features(descriptor_for_all[i], descriptor_for_all[i + 1])
        matches_for_all.append(matches)
    return np.array(matches_for_all)
        
#########################################################################################################################################

# 从2张匹配过的图像计算相机外参。输入：内参，图1和图2的匹配过的点清单。输出：旋转、平移、mask(用于指明点清单中的异常点，可用于过滤)
def find_transform(K, p1, p2):  
    focal_length = 0.5 * (K[0, 0] + K[1, 1])    # 焦距
    principle_point = (K[0, 2], K[1, 2])        # 原点
    # 计算本质矩阵 Essential Matrix
    E,mask = cv2.findEssentialMat(p1, p2, focal_length, principle_point, cv2.RANSAC, 0.999, 1.0)
    cameraMatrix = np.array([[focal_length, 0, principle_point[0]], [0, focal_length, principle_point[1]], [0, 0, 1]])
    # 分解本质矩阵，得到R和t
    pass_count, R, T, mask = cv2.recoverPose(E, p1, p2, cameraMatrix, mask)
    return R, T, mask

# 输入：图1特征点清单，图2特征点清单，图1和图2特征点匹配结果。输出：匹配过的特征点在图1和图2中的点清单
def get_matched_points(p1, p2, matches):
    src_pts = np.asarray([p1[m.queryIdx].pt for m in matches])
    dst_pts = np.asarray([p2[m.trainIdx].pt for m in matches])
    return src_pts, dst_pts

def get_matched_colors(c1, c2, matches):
    color_src_pts = np.asarray([c1[m.queryIdx] for m in matches])
    color_dst_pts = np.asarray([c2[m.trainIdx] for m in matches])    
    return color_src_pts, color_dst_pts

# 选择重合的点。输入：特征点清单、mask指明哪些特征点有效；输出：根据mask去掉无效点之后的特征点清单
def maskout_points(p1, mask):   
    p1_copy = []
    for i in range(len(mask)):
        if mask[i] > 0:
            p1_copy.append(p1[i])
    return np.array(p1_copy)
    
# 初始化3D重建的各项参数，仅利用图像1和2
# 输入：相机内参，各图像特征点列表及其颜色列表，两两图像特征点匹配结果列表
# 输出：3D坐标清单（点云），？？，及其颜色，？？，？？
def init_structure(K, key_points_for_all, colors_for_all, matches_for_all):  
    # 从图1和图2的特征匹配结果，得到匹配点列表p1和p2（这些匹配点在图1和图2中的点坐标），以及它们的颜色c1和c2
    p1, p2 = get_matched_points(key_points_for_all[0], key_points_for_all[1], matches_for_all[0])
    c1, c2 = get_matched_colors(colors_for_all[0], colors_for_all[1], matches_for_all[0])
    # 计算本质矩阵，并分解得到相机的外参：R，t
    if find_transform(K, p1, p2):
        R,T,mask = find_transform(K, p1, p2)
    else:
        R,T,mask = np.array([]), np.array([]), np.array([])
    
    # 清理一下，利用mask去掉匹配点中的无效的点
    p1 = maskout_points(p1, mask)
    p2 = maskout_points(p2, mask)
    colors = maskout_points(c1, mask)

    # 设置第一个相机的变换矩阵，即作为剩下摄像机矩阵变换的基准。把图1相机作为世界坐标系？
    R0 = np.eye(3, 3)
    T0 = np.zeros((3, 1))
    structure = reconstruct(K, R0, T0, R, T, p1, p2)

    rotations = [R0, R]
    motions = [T0, T]

    # 生成一个跟‘特征点列表’长度（图像数量）和宽度（每个图像的特征点数量）相同的、但值全部为-1的列表
    correspond_struct_idx = []
    for key_p in key_points_for_all:
        correspond_struct_idx.append(np.ones(len(key_p)) *- 1)
    correspond_struct_idx = np.array(correspond_struct_idx)
    # 给有效的'特征匹配对'编了序号idx，然后在correspond_struct_idx中标记了key_points_for_all中特征点与'特征匹配对'的查找关系
    # -1表示无效特征点，其它值表示'特征匹配对'的序号。也就是说，在correspond_struct_idx[i]和correspond_struct_idx[i+1]中有相同id值对应的两个特征点是匹配关系
    idx = 0
    matches = matches_for_all[0]
    for i, match in enumerate(matches):
        if mask[i] == 0:        # 跳过无效匹配点
            continue
        correspond_struct_idx[0][int(match.queryIdx)] = idx
        correspond_struct_idx[1][int(match.trainIdx)] = idx
        idx += 1                                               
    return structure, correspond_struct_idx, colors, rotations, motions
    
# 输出特征点的3D坐标（三维重建）。输入：相机内参外参（K、俩图像的R和t），匹配后的俩图像特征点清单。输出：特征点的3D坐标清单
def reconstruct(K, R1, T1, R2, T2, p1, p2):    
    proj1 = np.zeros((3, 4))
    proj2 = np.zeros((3, 4))
    proj1[0:3, 0:3] = np.float32(R1)
    proj1[:, 3] = np.float32(T1.T)
    proj2[0:3, 0:3] = np.float32(R2)
    proj2[:, 3] = np.float32(T2.T)
    fk = np.float32(K)
    proj1 = np.dot(fk, proj1)
    proj2 = np.dot(fk, proj2)
    # 用两个相机的位姿和特征点在两个相机坐标系下的坐标，输出三角化后的特征点的3D齐次坐标
    s = cv2.triangulatePoints(proj1, proj2, p1.T, p2.T)
    structure = []    
    for i in range(len(s[0])):
        col = s[:, i]
        col /= col[3]
        structure.append([col[0], col[1], col[2]])    
    return np.array(structure)

# 点云融合：把next_structure融入structure，把next_colors融入colors
def fusion_structure(matches, struct_indices, next_struct_indices, structure, next_structure, colors, next_colors):
    # 遍历每一对匹配点
    for i,match in enumerate(matches):  
        query_idx = match.queryIdx
        train_idx = match.trainIdx
        struct_idx = struct_indices[query_idx]  
        if struct_idx >= 0:
            next_struct_indices[train_idx] = struct_idx
            continue
        structure = np.append(structure, [next_structure[i]], axis = 0)
        colors = np.append(colors, [next_colors[i]], axis = 0)
        struct_indices[query_idx] = next_struct_indices[train_idx] = len(structure) - 1
    return struct_indices, next_struct_indices, structure, colors

#制作图像点以及空间点
def get_objpoints_and_imgpoints(matches, struct_indices, structure, key_points):    
    object_points = []
    image_points = []
    for match in matches:
        query_idx = match.queryIdx
        train_idx = match.trainIdx
        struct_idx = struct_indices[query_idx]  
        if struct_idx < 0: 
            continue
        object_points.append(structure[int(struct_idx)])
        image_points.append(key_points[train_idx].pt)
    return np.array(object_points), np.array(image_points)

########################
#bundle adjustment
########################

# 这部分中，函数get_3dpos是原方法中对某些点的调整，而get_3dpos2是根据笔者的需求进行的修正，即将原本需要修正的点全部删除。
# bundle adjustment请参见https://www.cnblogs.com/zealousness/archive/2018/12/21/10156733.html

def get_3dpos(pos, ob, r, t, K):
    dtype = np.float32
    def F(x):
        p,J = cv2.projectPoints(x.reshape(1, 1, 3), r, t, K, np.array([]))
        p = p.reshape(2)
        e = ob - p
        err = e    
                
        return err
    res = least_squares(F, pos)
    return res.x

def get_3dpos_v1(pos,ob,r,t,K):
    p,J = cv2.projectPoints(pos.reshape(1, 1, 3), r, t, K, np.array([]))
    p = p.reshape(2)
    e = ob - p
    if abs(e[0]) > config.x or abs(e[1]) > config.y:        
        return None
    return pos

def bundle_adjustment(rotations, motions, K, correspond_struct_idx, key_points_for_all, structure):
    
    for i in range(len(rotations)):
        r, _ = cv2.Rodrigues(rotations[i])
        rotations[i] = r
    for i in range(len(correspond_struct_idx)):
        point3d_ids = correspond_struct_idx[i]
        key_points = key_points_for_all[i]
        r = rotations[i]
        t = motions[i]
        for j in range(len(point3d_ids)):
            point3d_id = int(point3d_ids[j])
            if point3d_id < 0:
                continue
            new_point = get_3dpos_v1(structure[point3d_id], key_points[j].pt, r, t, K)
            structure[point3d_id] = new_point
    
    return structure

#######################
#作图
#######################

# 这里有两种方式作图，其中一个是matplotlib做的，但是第二个是基于mayavi做的，效果上看，fig_v1效果更好。fig_v2是mayavi加颜色的效果。

def fig(structure, colors):
    colors /= 255
    for i in range(len(colors)):
        colors[i, :] = colors[i, :][[2, 1, 0]]
    fig = plt.figure()
    fig.suptitle('3d')
    ax = fig.gca(projection = '3d')
    for i in range(len(structure)):
        ax.scatter(structure[i, 0], structure[i, 1], structure[i, 2], color = colors[i, :], s = 5)
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.view_init(elev = 135, azim = 90)
    plt.show()

def fig_v1(structure):

    mlab.points3d(structure[:, 0], structure[:, 1], structure[:, 2], mode = 'point', name = 'dinosaur')
    mlab.show()

def fig_v2(structure, colors):

    for i in range(len(structure)):
        
        mlab.points3d(structure[i][0], structure[i][1], structure[i][2], 
            mode = 'point', name = 'dinosaur', color = colors[i])

    mlab.show()
    
def main():
    imgdir = config.image_dir
    img_names = os.listdir(imgdir)
    img_names = sorted(img_names)
    for i in range(len(img_names)):
        img_names[i] = imgdir + img_names[i]
    # img_names = img_names[0:10]

    # 相机参数
    K = config.K
    
    # 生成每一个图像的特征点列表，及其描述符、颜色列表
    key_points_for_all, descriptor_for_all, colors_for_all = extract_features(img_names)
    
    # 对图像列表中，每前后两个图像进行特征点匹配，输出匹配结果（列表长度应该是图像数量-1，列表中每一项包含2个图像的匹配结果，即匹配点清单）
    matches_for_all = match_all_features(descriptor_for_all)

    # 先用图1和图2匹配的结果进行初始化，生成初始的：点云structure及其颜色colors，俩俩图像间的R/t对，3D点在特征点列表中的查找对应
    structure, correspond_struct_idx, colors, rotations, motions = init_structure(K, key_points_for_all, colors_for_all, matches_for_all)   

    # 循环一遍所有配对图像的特征匹配结果，刷新点云
    for i in range(1, len(matches_for_all)):
        # 得到xx坐标，用于计算相机位姿
        object_points, image_points = get_objpoints_and_imgpoints(matches_for_all[i], correspond_struct_idx[i], structure, key_points_for_all[i + 1])
        # solvePnPRansac函数要求点数大于7。对小于7的点集做一个重复填充操作，即用点集中的第一个点补满7个
        if len(image_points) < 7:
            while len(image_points) < 7:
                object_points = np.append(object_points, [object_points[0]], axis = 0)
                image_points = np.append(image_points, [image_points[0]], axis = 0)
        # 计算相机位姿PnP：从多个3D坐标及其2D投影，计算出相机的位姿
        _, r, T, _ = cv2.solvePnPRansac(object_points, image_points, K, np.array([]))   # 参数：3D坐标，2D坐标，相机内参，
        R, _ = cv2.Rodrigues(r)     # 通过罗德里格斯公式转换旋转向量和旋转矩阵
        rotations.append(R)
        motions.append(T)

        # 生成第i图和第i+1图匹配点的3D坐标
        p1, p2 = get_matched_points(key_points_for_all[i], key_points_for_all[i + 1], matches_for_all[i])
        c1, c2 = get_matched_colors(colors_for_all[i], colors_for_all[i + 1], matches_for_all[i])
        next_structure = reconstruct(K, rotations[i], motions[i], R, T, p1, p2)
        # 将刚算出的3D点融入点云：把next_structure融入structure。颜色也一样融合
        correspond_struct_idx[i], correspond_struct_idx[i + 1], structure, colors = fusion_structure(matches_for_all[i],correspond_struct_idx[i],correspond_struct_idx[i+1],structure,next_structure,colors,c1)

    # Bundle Adjustment 以最小化重投影误差
    structure = bundle_adjustment(rotations, motions, K, correspond_struct_idx, key_points_for_all, structure)

    # 由于经过bundle_adjustment的structure，会产生一些空的点（实际代表的意思是已被删除）。这里删除那些为空的点
    i = 0
    while i < len(structure):
        if math.isnan(structure[i][0]):
            structure = np.delete(structure, i, 0)
            colors = np.delete(colors, i, 0)
            i -= 1
        i += 1
        
    print(len(structure))
    print(len(motions))
    # np.save('structure.npy', structure)
    # np.save('colors.npy', colors)
    
    # fig(structure,colors)
    fig_v1(structure)
    # fig_v2(structure, colors)
   
if __name__ == '__main__':
    main()
