from BLS_functions import one_hot_m
from GetShrecdata import get_dataset, getFeatures1, getFeatures2, getFeatures3
from MeshBLS import MeshBLS

# 数据载
dataset_dir = 'shrec_16'  # 读取数据集
train_data_array, train_labels_array, test_data_array, test_labels_array = get_dataset(dataset_dir)
print("train_data_array shape:" ,train_data_array.shape)

centerTrainData = train_data_array[:, :, 0:3]
normalTrainData = train_data_array[:, :, 3:6]
curvsTrainData = train_data_array[:, :, 6:9] 
angle1TrainData = train_data_array[:, :, 9:12]
angle2TrainData = train_data_array[:, :, 12:18]
areaTrainData = train_data_array[:, :, 18:19]


centerTestData = test_data_array[:, :, 0:3]
normalTestData = test_data_array[:, :, 3:6]
curvsTestData = train_data_array[:, :, 6:9] 
angle1TestData = test_data_array[:, :, 9:12]
angle2TestData = test_data_array[:, :, 12:18]
areaTestData = test_data_array[:, :, 18:19]

trainlabel = one_hot_m(train_labels_array, 30)
testlabel = one_hot_m(test_labels_array, 30)

print('================extract the feas =======================')
centerTrain, centerTest = getFeatures1(centerTrainData, centerTestData)

normalTrain, normalTest = getFeatures1(normalTrainData, normalTestData)


print(f"curvsTrainData shape before passing to model: {curvsTrainData.shape}")
print(f"curvsTestData shape before passing to model: {curvsTestData.shape}")
curvsTrain, curvsTest = getFeatures1(curvsTrainData, curvsTestData)


print(f"angle1TrainData shape before passing to model: {angle1TrainData.shape}")
print(f"angle1TestData shape before passing to model: {angle1TestData.shape}")
angle1Train, angle1Test = getFeatures1(angle1TrainData, angle1TestData)


print(f"angle2TrainData shape before passing to model: {angle2TrainData.shape}")
print(f"angle2TestData shape before passing to model: {angle2TestData.shape}")
angle2Train, angle2Test = getFeatures2(angle2TrainData, angle2TestData)

print(f"areaTrainData shape before passing to model: {areaTrainData.shape}")
print(f"areaTestData shape before passing to model: {areaTestData.shape}")
areaTrain, areaTest = getFeatures3(areaTrainData, areaTestData)

print('================run meshbls=======================')
MeshBLS(centerTrain, centerTest,
        normalTrain, normalTest, 
        curvsTrain, curvsTest,
        angle1Train, angle1Test,
        angle2Train, angle2Test,
        areaTrain, areaTest,
        trainlabel, testlabel)