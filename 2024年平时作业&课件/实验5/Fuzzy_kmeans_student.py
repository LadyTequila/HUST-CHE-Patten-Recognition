import math
import copy
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

"""
这段代码实现了基于模糊C均值聚类（Fuzzy C-Means Clustering）算法的人脸图像聚类和识别系统。

主要功能：
1. **数据点定义与加载**：
   - 定义了 `Point` 类，用于存储每个数据点的特征、隶属度和所属簇信息。
   - 提供了从图像加载特征数据的函数 `loadAllImagesWithRealLabel`。

2. **模糊C均值聚类**：
   - 使用 `fuzzyCMeansClustering` 函数执行模糊C均值聚类算法，通过迭代优化更新簇中心及隶属度。
   - 初始簇中心的选择使用了 K-Means++ 方法以提高聚类质量。

3. **簇名分配**：
   - 通过统计每个簇中的真实标签分布，确定簇对应的人名（`assignClusterName`）。

4. **人脸识别**：
   - 使用 `recognizePerson` 函数基于簇中心和特征距离识别新图像中的人名。

5. **结果可视化**：
   - 通过降维（PCA）将高维特征映射到二维平面，并对聚类结果进行可视化展示（`visualizeResults`）。

6. **准确率计算**：
   - 对训练数据进行预测，并计算识别准确率。

代码的主要流程：
1. 加载图片并提取特征。
2. 使用模糊C均值算法对数据进行聚类。
3. 将每个簇与真实人名建立映射关系。
4. 对新图像进行人脸识别。
5. 可视化聚类结果并计算识别准确率。

适用场景：
- 该代码适用于人脸图像数据的聚类和初步分类。
- 可以对未知人脸图像进行预测，输出对应的分类标签或人名。

需要注意：
- 数据目录路径和图像文件名格式需要与代码中的 `loadAllImagesWithRealLabel` 函数保持一致。
- 模糊C均值聚类适合数据分布连续、具有模糊边界的场景。
"""
# 定义一个非常大的浮点数，用于初始化最小距离
FLOAT_MAX = 1e100

class Point:
    """
    定义一个点类，用于表示数据点及其相关属性。
    """
    def __init__(self, clusterCenterNumber, features=None, group=0):
        # 特征向量，如果未提供则初始化为空列表
        self.features = features if features is not None else []
        # 所属的簇编号，默认值为0
        self.group = group
        # 属于各个簇的隶属度，初始化为0.0
        self.membership = [0.0 for _ in range(clusterCenterNumber)]
        # 真实标签，用于记录数据点的实际类别
        self.real_label = None  # 用于记录真实标签

def solveDistanceBetweenPoints(pointA, pointB):
    """
    计算两个点之间的欧氏距离的平方。

    参数:
    - pointA: 第一个点
    - pointB: 第二个点

    返回:
    - 两点之间的距离的平方
    """
    # 将特征向量转换为numpy数组，并计算差值
    diff = np.array(pointA.features) - np.array(pointB.features)
    # 返回差值的平方和
    return np.sum(diff**2)

def getNearestCenter(point, clusterCenterGroup):
    """
    找到距离给定点最近的簇中心。

    参数:
    - point: 要计算的点
    - clusterCenterGroup: 当前所有的簇中心

    返回:
    - 最近簇中心的索引和距离
    """
    minIndex = 0
    minDistance = FLOAT_MAX
    # 遍历所有簇中心，寻找最小距离
    for index, center in enumerate(clusterCenterGroup):
        distance = solveDistanceBetweenPoints(point, center)
        if distance < minDistance:
            minDistance = distance
            minIndex = index
    return minIndex, minDistance

def kMeansPlusPlus(points, clusterCenterGroup):
    """
    初始化簇中心，使用K-Means++算法提高初始簇中心的选择质量。

    参数:
    - points: 所有数据点
    - clusterCenterGroup: 簇中心列表，将被初始化
    """
    import random
    # 随机选择第一个簇中心
    clusterCenterGroup[0] = copy.copy(random.choice(points))
    distanceGroup = [0.0 for _ in range(len(points))]
    total_distance = 0.0

    # 逐个选择剩余的簇中心
    for index in range(1, len(clusterCenterGroup)):
        total_distance = 0.0
        # 计算每个点到最近已选簇中心的距离平方
        for i, point in enumerate(points):
            distanceGroup[i] = getNearestCenter(point, clusterCenterGroup[:index])[1]
            total_distance += distanceGroup[i]

        # 生成一个随机阈值
        total_distance *= random.random()
        # 根据距离概率选择下一个簇中心

        # TODO: 从距离组中逐一减去距离，找到第一个使total_distance小于0的点，并将其设为新的簇中心
        for i, distance in enumerate(distanceGroup):     # 引导注释：
            total_distance -= distance # 1. 减去当前点的距离，减去每个点的距离后，我们相当于按照每个点与簇中心的距离累积来模拟“概率”
            if total_distance <= 0: # 2. 检查total_distance是否小于0
                clusterCenterGroup[index] = copy.copy(points[i]) # 3. 如果是，则将当前点复制为新的簇中心，并退出循环
            #############当 total_distance 小于零时，说明我们已经找到了一个新的簇中心。这是因为在K-Means++中，选择下一个簇中心的概率是基于每个点与簇中心距离的平方反比

def getSingleMembership(point, clusterCenterGroup, weight):
    """
    计算给定点对每个簇中心的隶属度。

    参数:
    - point: 数据点
    - clusterCenterGroup: 当前所有簇中心
    - weight: 模糊权重参数（通常大于1）

    更新:
    - 更新point.membership为新的隶属度列表
    """
    # 计算点到所有簇中心的距离平方
    distances = [solveDistanceBetweenPoints(point, center) for center in clusterCenterGroup]
    for centerIndex, _ in enumerate(point.membership):
        total = 0.0
        if distances[centerIndex] == 0:
            # 如果距离为0，点完全属于该簇
            point.membership = [0.0]*len(distances)
            point.membership[centerIndex] = 1.0
            return
        for distance in distances:
            if distance == 0:
                # 如果某个距离为0，点完全属于对应簇
                point.membership = [0.0]*len(distances)
                point.membership[centerIndex] = 1.0
                return
            # 根据模糊权重计算隶属度
            total += (distances[centerIndex]/distance)**(1.0/(weight-1.0))
        point.membership[centerIndex] = 1.0/total

def fuzzyCMeansClustering(points, clusterCenterNumber, weight):
    """
    执行模糊C均值聚类算法。

    参数:
    - points: 所有数据点
    - clusterCenterNumber: 簇的数量
    - weight: 模糊权重参数

    返回:
    - 最终的簇中心列表
    - 每个簇中心的历史轨迹（用于可视化）
    """
    # 初始化簇中心列表
    clusterCenterGroup = [Point(clusterCenterNumber) for _ in range(clusterCenterNumber)]
    # 使用K-Means++算法初始化簇中心
    kMeansPlusPlus(points, clusterCenterGroup)

    # 记录每个簇中心的变化轨迹
    clusterCenterTrace = [[copy.copy(center)] for center in clusterCenterGroup]
    tolerableError, currentError = 1.0, FLOAT_MAX

    # 迭代直到误差小于容忍误差
    while currentError >= tolerableError:
        # 更新每个点的隶属度
        for point in points:
            getSingleMembership(point, clusterCenterGroup, weight)

        # 计算新的簇中心
        currentCenterGroup = [Point(clusterCenterNumber) for _ in range(clusterCenterNumber)]
        for centerIndex, center in enumerate(currentCenterGroup):
            upperSum = np.zeros(len(points[0].features))  # 分子部分
            lowerSum = 0.0  # 分母部分
            for point in points:
                membershipWeight = (point.membership[centerIndex] ** weight)
                upperSum += membershipWeight * np.array(point.features) # TODO: 根据隶属度权重更新分子
                         # 引导注释：1. 将 `membershipWeight` 与点的特征向量相乘，表示当前点对簇中心的加权特征值。2. 将结果累加到分子变量 `upperSum` 中。
                lowerSum += membershipWeight # TODO: 根据隶属度权重更新分母
                         # 引导注释：1. 将 `membershipWeight` 累加到分母变量 `lowerSum` 中。2. 分母表示所有点对当前簇中心的权重总和。
            # 更新簇中心的特征向量
            center.features = (upperSum / lowerSum).tolist() # TODO: 更新当前簇中心的特征向量
                      # 引导注释：1. 使用分子 `upperSum` 除以分母 `lowerSum`，计算当前簇中心的新特征向量。2. 将结果转换为列表并赋值给 `center.features`。
        # 计算当前迭代的误差
        currentError = 0.0
        for index, singleTrace in enumerate(clusterCenterTrace):
            old_center = singleTrace[-1]
            new_center = currentCenterGroup[index]
            currentError += solveDistanceBetweenPoints(old_center, new_center)
            # 记录新的簇中心
            singleTrace.append(copy.copy(new_center))
            # 更新当前簇中心
            clusterCenterGroup[index] = copy.copy(new_center)

    # 确定每个点最终所属的簇
    for point in points:
        maxIndex, maxMembership = 0, 0.0
        for index, singleMembership in enumerate(point.membership):
            if singleMembership > maxMembership:
                maxMembership = singleMembership
                maxIndex = index
        point.group = maxIndex

    return clusterCenterGroup, clusterCenterTrace

def visualizeResults(points, clusterCenterNumber, title='Clusters visualization'):
    """
    可视化聚类结果，使用PCA将特征降至二维。

    参数:
    - points: 所有数据点
    - clusterCenterNumber: 簇的数量
    - title: 图表标题
    """
    # 提取所有点的特征向量
    X = np.array([p.features for p in points])
    # 提取所有点的簇标签
    labels = np.array([p.group for p in points])

    # 使用PCA将特征降到二维
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # 创建绘图
    plt.figure(figsize=(8,6))
    K = clusterCenterNumber
    # 使用tab10色图为不同簇分配颜色
    colors = plt.cm.get_cmap('tab10', K)
    for k in range(K):
        # 选择属于当前簇的点
        cluster_points = X_pca[labels == k]
        if len(cluster_points) > 0:
            # 绘制当前簇的点
            plt.scatter(cluster_points[:,0], cluster_points[:,1],
                        color=colors(k), label=f'Cluster {k}', alpha=0.7, edgecolors='k')
    # 设置图表标题和轴标签
    plt.title(title)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.grid(True)
    plt.show()

def loadAllImagesWithRealLabel(directory, clusterCenterNumber, name_to_label):
    """
    加载指定目录下的所有图片，创建数据点并设置其真实标签。

    参数:
    - directory: 图片存放的目录
    - clusterCenterNumber: 簇的数量
    - name_to_label: 从人名到标签的映射字典

    返回:
    - 包含所有数据点的列表
    """
    points = []
    # 列出目录中所有以.tif结尾的文件
    image_files = [f for f in os.listdir(directory) if f.lower().endswith('.tif')]
    image_files.sort()

    for img_name in image_files:
        # 分离文件名和扩展名
        base_name, _ = os.path.splitext(img_name)
        # 提取人名部分，例如'TOM'，'JERRY'等
        person_id = base_name.split('-')[0]  # 'TOM', 'JERRY', 'MIKE'
        # 获取对应的标签
        label = name_to_label[person_id]

        # 构建图片的完整路径
        img_path = os.path.join(directory, img_name)
        # 打开图片并转换为灰度图
        img = Image.open(img_path).convert('L')
        # 将图片转换为numpy数组，并转为浮点型
        img_array = np.array(img, dtype=float)
        # 将二维数组展平为一维特征向量
        features = img_array.flatten().tolist()

        # 创建一个数据点
        p = Point(clusterCenterNumber, features=features, group=0)
        # 设置真实标签
        p.real_label = label
        # 添加到点列表中
        points.append(p)
    return points

def assignClusterName(points, clusterCenterNumber, label_to_name):
    """
    为每个簇分配一个人名，基于簇内多数真实标签。

    参数:
    - points: 所有数据点
    - clusterCenterNumber: 簇的数量
    - label_to_name: 从标签到人名的映射字典

    返回:
    - 簇到人名的映射字典
    """
    # 初始化一个二维数组，用于统计每个簇中每个标签的数量
    label_count_per_cluster = np.zeros((clusterCenterNumber, len(label_to_name)), dtype=int)

    for p in points:
        # p.group是聚类后的簇编号
        # p.real_label是原始真实标签
        label_count_per_cluster[p.group, p.real_label] += 1

    cluster_to_name = {}
    for c in range(clusterCenterNumber):
        # 找到每个簇中出现次数最多的标签
        max_label = np.argmax(label_count_per_cluster[c])
        # 将簇映射到对应的人名
        cluster_to_name[c] = label_to_name[max_label]
    return cluster_to_name

def recognizePerson(img_path, clusterCenterGroup, cluster_to_name, clusterCenterNumber):
    """
    识别给定图片中的人名。

    参数:
    - img_path: 待识别图片的路径
    - clusterCenterGroup: 当前所有簇中心
    - cluster_to_name: 簇到人名的映射字典
    - clusterCenterNumber: 簇的数量

    返回:
    - 识别出的人名
    """
    # 打开图片并转换为灰度图
    img = Image.open(img_path).convert('L')
    # 将图片转换为numpy数组，并转为浮点型
    img_array = np.array(img, dtype=float)
    # 将二维数组展平为一维特征向量
    features = img_array.flatten().tolist()

    # 创建一个测试点
    test_point = Point(clusterCenterNumber, features=features)
    # 找到最近的簇中心
    nearestIndex, _ = getNearestCenter(test_point, clusterCenterGroup)
    # 获取对应的人名
    person_name = cluster_to_name[nearestIndex]
    return person_name

def main():
    # 图片存放目录，请根据实际情况修改
    directory = "AR_Gray_50by40"  # 请修改为实际存放图片的目录
    # 簇的数量，这里有3个人
    clusterCenterNumber = 3
    # 模糊权重参数，通常大于1
    weight = 1.3

    # 定义映射关系：从 person_id 到 label
    # 假设数据为 TOM, JERRY, MIKE三个人，根据实际名称或人脸ID修改
    name_to_label = {'TOM':0, 'JERRY':1, 'MIKE':2}
    label_to_name = {0:'TOM', 1:'JERRY', 2:'MIKE'}

    # 加载训练集（全部数据）
    points = loadAllImagesWithRealLabel(directory, clusterCenterNumber, name_to_label)

    # 使用Fuzzy C-Means对训练数据聚类
    clusterCenterGroup, clusterCenterTrace = fuzzyCMeansClustering(points, clusterCenterNumber, weight)

    # 将每个cluster映射到某个人名
    cluster_to_name = assignClusterName(points, clusterCenterNumber, label_to_name)
    print("Cluster to name mapping:", cluster_to_name)

    # 可视化训练集聚类结果
    # visualizeResults(points, clusterCenterNumber, "Train Set Clusters")

    # 测试识别未知图片
    test_img = "test3.tif"  # 您的测试图片路径
    recognized_name = recognizePerson(test_img, clusterCenterGroup, cluster_to_name, clusterCenterNumber)
    print("Recognized person for test image:", recognized_name)

    # -------- 准确率计算部分 --------
    # 使用所有已知数据points，计算识别准确率
    correct = 0
    total = len(points)

    for p in points:
        real_name = label_to_name[p.real_label]        # 真实名称
        predicted_name = cluster_to_name[p.group]      # 聚类预测名称
        if real_name == predicted_name:
            correct += 1

    accuracy = correct / total
    print("Recognition accuracy on training set:", accuracy)
    # -------------------------------------

if __name__ == "__main__":
    main()