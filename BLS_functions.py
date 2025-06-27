import torch
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def tansig(x):
    # 使用 PyTorch 版本的 tanh 函数
    return torch.tanh(x)

def pinv(A, reg):
    # 使用 PyTorch 来计算伪逆
    AtA = A.T @ A
    regularized_matrix = AtA + reg * torch.eye(A.shape[1], device=A.device)
    return torch.linalg.pinv(regularized_matrix) @ A.T

def softmax(x):
    # 使用 PyTorch 版本的 softmax
    return F.softmax(x, dim=1)

def show_accuracy(predictLabel, Label):
    # 使用 torch.argmax 来代替 np.argmax
    pred = torch.argmax(predictLabel, dim=1)
    true = torch.argmax(Label, dim=1)
    return torch.mean((pred == true).float())

def remove_singular_data(X):
    print(f"Original X shape: {X.shape}")
    
    # 获取唯一的行，返回三个值：唯一行、索引、出现次数
    unique_X, indices, counts = torch.unique(X, dim=0, return_inverse=True, return_counts=True)
    
    print(f"Unique X shape: {unique_X.shape}")
    print(f"Indices shape: {indices.shape}")
    print(f"Counts shape: {counts.shape}")
    
    # 进一步的操作，比如只保留出现次数大于 1 的行
    cleaned_X = unique_X[counts > 1]
    
    return cleaned_X



def one_hot_m(labels, num_classes=None):
    # 确保标签是 numpy 数组并调整形状为二维
    labels = np.array(labels).reshape(-1, 1)  # Reshape to 2D array because OneHotEncoder expects a 2D array
    
    # 使用 OneHotEncoder 进行 one-hot 编码
    enc = OneHotEncoder(sparse_output=False, categories='auto')  # sparse_output=False 返回一个密集矩阵
    one_hot_labels = enc.fit_transform(labels)
    
    # 将 one-hot 编码的结果转换为 PyTorch tensor
    one_hot_labels_tensor = torch.tensor(one_hot_labels, dtype=torch.float32)  # 使用 float32 数据类型

    return one_hot_labels_tensor