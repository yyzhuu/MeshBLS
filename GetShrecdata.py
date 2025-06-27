import trimesh
import os
import numpy as np
import torch
from cnn_model import Net1, Net2, Net3

def compute_curvature(centerTrainData, normalTrainData):
    """
    计算面曲率：通过顶点法线和面法线的点积来估算曲率。
    
    参数:
    centerTrainData (numpy.ndarray): 顶点法线数据，形状为 (num_faces, num_vertices, 3)
    normalTrainData (numpy.ndarray): 面法线数据，形状为 (num_faces, 3)
    
    返回:
    numpy.ndarray: 面曲率，形状为 (num_faces,)
    """
    
    # 确保面法线和顶点法线的面数一致
    assert normalTrainData.shape[0] == centerTrainData.shape[0], "Shape mismatch between normalTrainData and centerTrainData"

    
    face_curvature = []

    # 遍历每个面，计算该面对应的曲率
    for i in range(normalTrainData.shape[0]):  # 对每个面进行处理
        face_normal = normalTrainData[i]  # 获取该面的面法线 (3,) 
        vertex_normals = centerTrainData[i]  # 获取该面的所有顶点法线 (num_vertices, 3)
        vertex_normals = vertex_normals.reshape(-1, 3)
        face_normal = face_normal.reshape(-1, 3)


        # 计算每个顶点法线与面法线的点积
        dot_products = np.sum(vertex_normals * face_normal, axis=1)  # 计算顶点法线与面法线的点积
        
        # 曲率估算：点积的平均值表示顶点与面法线的一致性
        curvature_value = np.mean(dot_products)  # 使用点积的平均值估算曲率

        face_curvature.append(curvature_value)
    
    return np.array(face_curvature)

# 计算面角度的函数

def compute_face_angles(mesh):
    """
    计算每个面的角度（使用余弦定理计算三角形的角度）。
    
    参数:
    mesh (trimesh.Trimesh): mesh 对象
    
    返回:
    numpy.ndarray: 每个面的三个角度，形状为 (num_faces, 3)
    """
    # 获取每个面的顶点
    vertices = mesh.vertices[mesh.faces]
    angles = []
    
    for face in vertices:
        # 计算每条边的长度
        a = np.linalg.norm(face[1] - face[0])  # 边BC
        b = np.linalg.norm(face[2] - face[1])  # 边AC
        c = np.linalg.norm(face[0] - face[2])  # 边AB
        
        # 余弦定理计算每个角度
        angle1 = np.arccos((b**2 + c**2 - a**2) / (2*b*c))  # 角A
        angle2 = np.arccos((a**2 + c**2 - b**2) / (2*a*c))  # 角B
        angle3 = np.arccos((a**2 + b**2 - c**2) / (2*a*b))  # 角C
        
        # 将三个角度添加到结果列表中
        angles.append([angle1, angle2, angle3])

    # 将列表转换为 NumPy 数组
    angles = np.array(angles)
    
    # 返回每个面的三个角度，形状为 (num_faces, 3)
    return angles

def compute_dihedral_angles(mesh):
    face_normals = mesh.face_normals
    dihedral_angles = []

    # 遍历每个面，计算其与相邻面的夹角
    for i in range(len(mesh.faces)):
        # 获取当前面
        normal_i = face_normals[i]
        
        # 获取相邻面的法线
        if i + 1 < len(face_normals):
            normal_j = face_normals[i + 1]
            dihedral_angle = np.arccos(np.dot(normal_i, normal_j))  # 计算夹角
        else:
            dihedral_angle = 0  # 如果没有相邻面，设置默认值为0
        
        dihedral_angles.append(dihedral_angle)

    # 返回形状为 (num_faces, 1)
    return np.array(dihedral_angles).reshape(-1, 1)


# 计算面面积的函数
def compute_face_area(mesh):
    """
    计算每个面的面积（通过计算两个边的叉积的模长）。
    
    参数:
    mesh (trimesh.Trimesh): mesh 对象
    
    返回:
    numpy.ndarray: 每个面的面积，形状为 (num_faces,)
    """
    vertices = mesh.vertices  # 获取顶点坐标
    faces = mesh.faces  # 获取面的索引
    
    # 预分配数组以存储面积
    areas = np.zeros(len(faces))
    
    # 对每个面进行计算
    for i, face in enumerate(faces):
        v1 = vertices[face[1]] - vertices[face[0]]  # 计算边1
        v2 = vertices[face[2]] - vertices[face[0]]  # 计算边2
        cross_product = np.cross(v1, v2)  # 计算叉积
        area = 0.5 * np.linalg.norm(cross_product)  # 面积是叉积的模长的一半
        areas[i] = area  # 存储面积值
    
    return areas


def safe_expand(arr, target_shape, default_value=np.nan):
    """
    扩展数组的维度到目标形状，如果数组长度不足，用默认值填充。
    
    参数：
    arr (numpy.ndarray): 输入数组
    target_shape (tuple): 目标形状
    default_value (float): 用于填充缺失数据的值，默认为 NaN
    
    返回：
    numpy.ndarray: 填充后的数组
    """
    # 检查数组的当前形状
    current_shape = arr.shape
    
    # 如果当前形状的维度不一致，进行扩展
    if current_shape[1] < target_shape[1]:
        # 用默认值填充
        padding = np.full((current_shape[0], target_shape[1] - current_shape[1]), default_value)
        arr = np.concatenate((arr, padding), axis=1)
    
    return arr

def extract_features_from_obj(obj_file):
    mesh = trimesh.load_mesh(obj_file)
    
    # 提取面中心
    face_centers = mesh.triangles_center  # (num_faces, 3)

    # 提取法线
    face_normals = mesh.face_normals  # (num_faces, 3)

    # 计算曲率
    curvs = compute_curvature(face_centers, face_normals)  # 计算曲率
    curvs = np.expand_dims(curvs, axis=1)  # 扩展曲率维度
    curvs = np.repeat(curvs, 3, axis=1)   # 扩展为 (num_faces, 3)
    
    # 计算角度
    angles1 = compute_face_angles(mesh)  # (num_faces, 3)
  
    # 计算二面角
    angles2 = compute_dihedral_angles(mesh)  # 需要确保返回 (num_faces, 1)

    # 计算面积
    areas = compute_face_area(mesh)  # (num_faces,)
    areas = np.expand_dims(areas, axis=1)  # 转换为 (num_faces, 1)

    # 确保每个特征的维度都是 (num_faces, 3)
    face_centers = safe_expand(face_centers, (face_centers.shape[0], 3))
    face_normals = safe_expand(face_normals, (face_normals.shape[0], 3))
    curvs = safe_expand(curvs, (curvs.shape[0], 3))
    angles1 = safe_expand(angles1, (angles1.shape[0], 3))
    angles2 = safe_expand(angles2, (angles2.shape[0], 6))
    areas = safe_expand(areas, (areas.shape[0], 3))  # 填充到三维

    # 将所有特征拼接：确保每个数组的维度一致
    features = np.concatenate([face_centers, face_normals, curvs, angles1, angles2, areas], axis=1)
    
    return features

# 从目录中获取数据集
def get_dataset(dataset_dir):
    """
    从指定的目录加载数据集，提取 .obj 文件的特征，并将其存储到训练数据和测试数据中。
    
    参数:
    dataset_dir (str): 数据集目录路径
    
    返回:
    tuple: 训练数据、训练标签、测试数据和测试标签
    """
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    
    for category in os.listdir(dataset_dir):
        category_dir = os.path.join(dataset_dir, category)
        
        if os.path.isdir(category_dir):
            cache_dir = os.path.join(category_dir, 'cache')
            os.makedirs(cache_dir, exist_ok=True)

            # 训练数据 - 从 'train' 文件夹中读取 .obj 文件
            train_category_data = []
            train_dir = os.path.join(category_dir, 'train')
            if os.path.exists(train_dir):  
                for file in os.listdir(train_dir):
                    if file.endswith('.obj'):
                        obj_file = os.path.join(train_dir, file)
                        features = extract_features_from_obj(obj_file)
                        
                        if isinstance(features, np.ndarray) and features.shape[0] > 0:
                            train_category_data.append(features)
                        else:
                            print(f"Skipping {obj_file} due to invalid or empty features")

                if len(train_category_data) > 0:
                    train_data.append(np.stack(train_category_data))  # 将每个类别的数据堆叠起来
                    train_labels.append(np.full(len(train_category_data), fill_value=category))  # 将标签添加到训练标签中
                    npz_file = os.path.join(cache_dir, f"{category}_train_data.npz")
                    np.savez(npz_file, data=np.stack(train_category_data))  # 缓存训练数据

            # 测试数据 - 从 'test' 文件夹中读取 .obj 文件
            test_category_data = []
            test_dir = os.path.join(category_dir, 'test')
            if os.path.exists(test_dir):
                for file in os.listdir(test_dir):
                    if file.endswith('.obj'):
                        obj_file = os.path.join(test_dir, file)
                        features = extract_features_from_obj(obj_file)
                        
                        if isinstance(features, np.ndarray) and features.shape[0] > 0:
                            test_category_data.append(features)
                        else:
                            print(f"Skipping {obj_file} due to invalid or empty features")

                if len(test_category_data) > 0:
                    test_data.append(np.stack(test_category_data))  # 将测试数据堆叠起来
                    test_labels.append(np.full(len(test_category_data), fill_value=category))  # 将标签添加到测试标签中
                    npz_file = os.path.join(cache_dir, f"{category}_test_data.npz")
                    np.savez(npz_file, data=np.stack(test_category_data))  # 缓存测试数据

    # 将所有数据和标签合并为一个数组
    return np.concatenate(train_data), np.concatenate(train_labels), np.concatenate(test_data), np.concatenate(test_labels)


def getFeatures1(centerTrainData, centerTestData):
    # 将 numpy 数组转换为 torch 张量
    centerTrainData = torch.tensor(centerTrainData, dtype=torch.float32)
    centerTestData = torch.tensor(centerTestData, dtype=torch.float32)
    
    # 创建一个 Net1 实例，输入维度为3，分类数量为30
    model = Net1(dim_in=3, class_n=30)
    
    # 将数据输入到模型中得到特征
    centerTrain = model(centerTrainData)
    centerTest = model(centerTestData)
    
    return centerTrain, centerTest


def getFeatures2(angle2TrainData, angle2TestData):

    angle2TrainData = torch.tensor(angle2TrainData, dtype=torch.float32)
    angle2TestData = torch.tensor(angle2TestData, dtype=torch.float32)
    
    # 创建一个 Net2 实例，输入维度为3，分类数量为30
    model = Net2(dim_in=6, class_n=30)
    
    # 将数据输入到模型中得到特征
    angle2Train = model(angle2TrainData)
    angle2Test = model(angle2TestData)
    
    return angle2Train, angle2Test


def getFeatures3(areaTrainData, areaTestData):

    # 将 numpy 数组转换为 torch 张量
    areaTrainData = torch.tensor(areaTrainData, dtype=torch.float32)
    areaTestData = torch.tensor(areaTestData, dtype=torch.float32)
    
    # 创建一个 Net3 实例，输入维度为1，分类数量为30
    model = Net3(dim_in=1, class_n=30)
    
    # 将数据输入到模型中得到特征
    areaTrain = model(areaTrainData)
    areaTest = model(areaTestData)
    
    return areaTrain, areaTest

