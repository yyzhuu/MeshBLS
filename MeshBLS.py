from BLS_functions import *
import numpy as np
from numpy import random
from scipy import linalg as LA
import time
import torch


center_c = 0.001
center_s = 0.8
center_e = 8000

curvs_c = 0.001
curvs_s = 0.8
curvs_e = 1000

angle1_c = 0.001
angle1_s = 0.8
angle1_e = 1000

angle2_c = 0.001
angle2_s = 0.8
angle2_e = 1000

area_c = 0.001
area_s = 0.8
area_e = 1000
# center_c = 0.001
# center_s = 1
# center_e = 1000
#
# curvs_c = 0.001
# curvs_s = 0.8
# curvs_e = 500
#
# angle1_c = 0.001
# angle1_s = 0.8
# angle1_e = 1000
#
# angle2_c = 0.001
# angle2_s = 1
# angle2_e = 1000
#
# area_c = 0.001
# area_s = 0.8
# area_e = 1000

fusion_c = 1  # Regularization coefficient


def Enlayer(Train, shape, e, s, train_y):
    # 使用 PyTorch 的 `torch.cat` 来拼接数据
    InOfEnhLayerWithBias = torch.cat([Train, 0.1 * torch.ones((Train.shape[0], 1), device=Train.device)], dim=1)
    
    if shape >= center_e:
        random.seed(67797325)
        weiOfEnhLayer = LA.orth(2 * random.randn(shape + 1, e)) - 1
    else:
        random.seed(67797325)
        weiOfEnhLayer = LA.orth(2 * random.randn(shape + 1, e).T - 1).T
        
    # 确保用 `torch.matmul` 替代 `np.dot`
    tempOfOutOfEnhLayer = torch.matmul(InOfEnhLayerWithBias, torch.tensor(weiOfEnhLayer, dtype=torch.float32))

    # 检查是否为空
    if tempOfOutOfEnhLayer.numel() > 0:
        # 在此之前，加入dim参数以确保max函数的正确使用
        parameterOfShrink = s / torch.max(tempOfOutOfEnhLayer, dim=0).values  # 选择dim=0对行进行归约
        OutOfEnhLayer = tansig(tempOfOutOfEnhLayer * parameterOfShrink)
    else:
        # 处理空矩阵的情况（可以根据实际需要进行相应的处理）
        print("Error: tempOfOutOfEnhLayer is empty, skipping further operations.")
        return None, None, None, None

    InputOfCLayer = torch.cat([Train, OutOfEnhLayer], dim=1)

    # 使用 `torch.pinverse` 来代替 `pinv` (如果需要)
    pinvOfInputC = pinv(InputOfCLayer, reg=1e-5) 
    CWeight = torch.matmul(pinvOfInputC, train_y)
    OutC = torch.matmul(InputOfCLayer, CWeight)

    return OutC, weiOfEnhLayer, parameterOfShrink, CWeight


def EnlayerTest(Test, weiOfEnhLayer, parameterOfShrink, CWeight):
    # 确保 weiOfEnhLayer 是 PyTorch 张量
    if isinstance(weiOfEnhLayer, np.ndarray):
        weiOfEnhLayer = torch.tensor(weiOfEnhLayer, dtype=torch.float32, device=Test.device)

    # 使用 `torch.cat` 拼接数据
    InOfEnhLayerWithBiasTest = torch.cat([Test, 0.1 * torch.ones((Test.shape[0], 1), device=Test.device)], dim=1)

    # 使用 `torch.matmul` 替代 `np.dot`
    tempOfOutOfEnhLayerTest = torch.matmul(InOfEnhLayerWithBiasTest, weiOfEnhLayer)

    # 激活函数
    OutOfEnhLayerTest = tansig(tempOfOutOfEnhLayerTest * parameterOfShrink)
    
    # 拼接测试数据
    InputOfCLayerTest = torch.cat([Test, OutOfEnhLayerTest], dim=1)

    # 使用 `torch.matmul` 替代 `np.dot`
    OutCTest = torch.matmul(InputOfCLayerTest, CWeight)
    return OutCTest




def MeshBLS(centerTrain, centerTest, normalTrain, normalTest, curvsTrain, curvsTest, angle1Train, angle1Test,
            angle2Train, angle2Test, areaTrain, areaTest, train_y, test_y):

    center_shape = centerTrain.shape[1]
    triangle_shape = curvsTrain.shape[1]
    angle1_shape = angle1Train.shape[1]
    angle2_shape = angle2Train.shape[1]
    area_shape = areaTrain.shape[1]

    time_start = time.time()
    #train_cleaned = remove_singular_data(centerTrain) 
    # 使用 `torch.matmul` 替代 `np.dot`
    OutC1, weiOfEnhLayer1, parameterOfShrink1, C1Weight = Enlayer(centerTrain, center_shape, center_e, center_s, train_y)
    OutC3, weiOfEnhLayer3, parameterOfShrink3, C3Weight = Enlayer(centerTrain, triangle_shape, curvs_e, curvs_s, train_y)
    OutC4, weiOfEnhLayer4, parameterOfShrink4, C4Weight = Enlayer(centerTrain, angle1_shape, angle1_e, angle1_s, train_y)
    OutC5, weiOfEnhLayer5, parameterOfShrink5, C5Weight = Enlayer(centerTrain, angle2_shape, angle2_e, angle2_s, train_y)
    OutC6, weiOfEnhLayer6, parameterOfShrink6, C6Weight = Enlayer(centerTrain, area_shape, area_e, area_s, train_y)

    OutC1_N = softmax(OutC1)
    OutC3_N = softmax(OutC3)
    OutC4_N = softmax(OutC4)
    OutC5_N = softmax(OutC5)
    OutC6_N = softmax(OutC6)

    # 使用 `torch.cat` 来拼接
    InputOfOutputLayer = torch.cat([OutC1_N, OutC3_N, OutC4_N, OutC5_N, OutC6_N], dim=1)

    # 使用 `torch.pinverse` 来代替 `pinv`
    pinvOfInput = pinv(InputOfOutputLayer, reg=1e-5)
    
    # 使用 `torch.matmul` 来代替 `np.dot`
    OutputWeight = torch.matmul(pinvOfInput, train_y)
    time_end = time.time()  # 训练完成
    trainTime = time_end - time_start

    # 训练输出
    OutputOfTrain = torch.matmul(InputOfOutputLayer, OutputWeight)
    trainAcc = show_accuracy(OutputOfTrain, train_y)
    print('Training accurate is', trainAcc * 100, '%')
    print('Training time is ', trainTime, 's')

    # 测试过程
    time_start = time.time()  # 测试计时开始

    # 强化层测试
    OutC1Test = EnlayerTest(centerTest, weiOfEnhLayer1, parameterOfShrink1, C1Weight)
    OutC3Test = EnlayerTest(curvsTest, weiOfEnhLayer3, parameterOfShrink3, C3Weight)
    OutC4Test = EnlayerTest(angle1Test, weiOfEnhLayer4, parameterOfShrink4, C4Weight)
    OutC5Test = EnlayerTest(angle2Test, weiOfEnhLayer5, parameterOfShrink5, C5Weight)
    OutC6Test = EnlayerTest(areaTest, weiOfEnhLayer6, parameterOfShrink6, C6Weight)

    OutC1Test_N = softmax(OutC1Test)
    OutC3Test_N = softmax(OutC3Test)
    OutC4Test_N = softmax(OutC4Test)
    OutC5Test_N = softmax(OutC5Test)
    OutC6Test_N = softmax(OutC6Test)

    print(f"OutC1Test_N shape: {OutC1Test_N.shape}")
    print(f"OutC3Test_N shape: {OutC3Test_N.shape}")
    print(f"OutC4Test_N shape: {OutC4Test_N.shape}")
    print(f"OutC5Test_N shape: {OutC5Test_N.shape}")
    print(f"OutC6Test_N shape: {OutC6Test_N.shape}")

    OutC3Test_N_resized = OutC3Test_N[:120, :]


    # 最终层输入
    InputOfOutputLayerTest = torch.cat([OutC1Test_N, OutC3Test_N_resized, OutC4Test_N, OutC5Test_N, OutC6Test_N], dim=1)

    # 最终测试输出
    OutputOfTest = torch.matmul(InputOfOutputLayerTest, OutputWeight)
    time_end = time.time()  # 训练完成
    testTime = time_end - time_start
    testAcc = show_accuracy(OutputOfTest, test_y)
    print('Testing accurate is', testAcc * 100, '%')
    print('Testing time is ', testTime, 's')
    return trainAcc, testAcc
