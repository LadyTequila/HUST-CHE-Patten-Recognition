import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import euclidean_distances
import copy
"""
这段代码实现了基于 **ISODATA聚类算法** 的动态分类方法。

ISODATA是一种动态聚类算法，通过结合分裂和合并操作，自动调整聚类数，使其尽量接近期望的目标分类数。算法可用于数据的自适应分组，特别适合处理具有复杂分布或未知聚类数的数据集。

主要功能：
1. **初始化**：
   - 输入期望的最终分类数 `designCenterNum` 和其他聚类参数。
   - 使用一组初始数据点作为输入，随机选择若干初始聚类中心。

2. **核心算法**：
   - **分裂操作**：当某些类的样本分布过于分散（标准差超阈值）时，将该类分裂为两个类。
   - **合并操作**：当某些类的中心距离过近（低于最小距离阈值）时，将这两个类合并为一个类。
   - **动态调整**：通过分裂和合并操作，逐步调整聚类数。

3. **辅助功能**：
   - 自动计算样本到类中心的标签分配。
   - 动态更新类中心位置。
   - 可视化聚类结果。

4. **迭代更新**：
   - 算法按照设定的最大迭代次数运行，每次更新当前的标签和类中心，直到分类稳定或达到最大迭代次数。

主要模块及其作用：
- `updateLabel`: 根据当前中心重新分配样本标签，更新类中心位置。
- `divide`: 检测并执行分裂操作，对满足分裂条件的类生成两个新类。
- `combine`: 检测并执行合并操作，对满足合并条件的类进行合并。
- `drawResult`: 绘制当前聚类结果，显示类中心和样本分布。
- `start`: 主流程控制，结合分裂和合并操作完成迭代。

适用场景：
- 当聚类数未知或需要动态调整时，ISODATA提供了比静态聚类算法（如K-Means）更灵活的解决方案。
- 可用于图像分割、文档分类和其他数据聚类任务。

注意事项：
- 示例数据为二维点，实际使用时需提供符合格式的输入数据。
- 参数设置（如分裂阈值、合并距离等）会显著影响结果，应根据数据特性调优。
"""
class ISODATA():
    def __init__(self, designCenterNum, Nc, LeastSampleNum, StdThred, LeastCenterDist, L, iterationNum):
        """
        ISODATA算法初始化函数。

        参数说明：
        :param designCenterNum: K，期望的最终聚类数。ISODATA会通过分裂和合并操作努力使类的数量接近K。
        :param Nc: 初始聚类中心个数。ISODATA开始时会随机从数据中选出Nc个点作为初始类中心。
        :param LeastSampleNum: 每类样本数最小阈值(θN)，用于决定分裂或合并类。
                              若某类样本数过少，不满足该阈值，会在后续决策中考虑合并。
        :param StdThred: 标准差阈值(θS)，若某类的标准差超过此值，则有可能进行分裂操作。
        :param LeastCenterDist: 类中心最小距离阈值(θC)，若两个类中心之间的距离小于该值，则会考虑对这两个类进行合并。
        :param L: 每次允许合并的最大类对数。限制一次迭代中合并类的数量。
        :param iterationNum: 最大迭代次数。算法迭代执行分裂、合并操作，直到迭代次数用完或达到理想状态。

        初始化时：
        1. 创建示例数据集（7个二维数据点）。
        2. 给出初始标签（占位）。
        3. 随机选择Nc个样本作为初始聚类中心。
        """

        # 保存参数
        self.K = designCenterNum  # 期望最终的类数
        self.centerNum = Nc  # 当前类中心的数目
        self.thetaN = LeastSampleNum  # 每类最少样本数阈值
        self.thetaS = StdThred  # 标准差阈值
        self.thetaC = LeastCenterDist  # 类中心最小距离阈值
        self.L = L  # 每次允许合并的最大类对数
        self.iteration = iterationNum  # 最大迭代次数

        # 构造示例数据集：7个二维点
        # 数据点坐标示例： (0,0), (0,1), (4,4), (4,5), (5,4), (5,5), (1,0)
        self.data = np.stack([[0, 0],
                              [0, 1],
                              [4, 4],
                              [4, 5],
                              [5, 4],
                              [5, 5],
                              [1, 0]], dtype=np.float64)

        # 初始标签（全部为0，占位作用，后面会通过updateLabel更新）
        self.label = np.stack([0, 0, 0, 0, 0, 0, 0])

        # 随机选择Nc个初始聚类中心
        # 从self.data中随机挑选Nc个数据点的索引
        center_ind = np.random.choice(np.arange(len(self.data), dtype=np.int32), Nc, replace=False)
        # 复制所选的数据点作为初始聚类中心
        self.center = np.stack([copy.deepcopy(self.data[center_ind[i], :]) for i in range(Nc)])

        # 平均类中心距离，用于度量类分布状态
        self.centerMeanDist = 0

    def updateLabel(self):
        """
        根据当前中心对所有样本重新分配标签，并更新中心位置。
        同时计算所有类到各自中心的平均距离之和的平均值，存于self.centerMeanDist。

        步骤：
        1. 计算每个样本到所有聚类中心的距离，得到距离矩阵distance。
        2. 为每个样本分配距离最近的中心作为其类标签。
        3. 根据新的标签，计算各类中心：对同一类的所有样本求均值得到新的类中心。
        4. 计算所有类的平均距离之和的平均值，以度量当前聚类质量。
        """
        # 计算样本到所有中心的距离矩阵
        # 返回一个shape为 (样本数, 类中心数) 的矩阵
        distance = euclidean_distances(self.data, self.center)

        # 为每个样本分配距离最近的中心作为其类标签
        # np.argmin(..., axis=1)会返回每行最小值的索引值，即距离最近的中心的编号
        self.label = np.argmin(distance, axis=1)

        # 根据新标签，重新计算每个类的中心：对属于该类的所有样本求均值
        for i in range(self.centerNum):
            # 找出属于类i的样本索引
            index = np.argwhere(self.label == i).squeeze()
            sameClassSample = self.data[index, :]
            # 若该类有多个样本，则计算均值作为新中心
            # 如果只有一个样本，sameClassSample是(2, )的数组，也可以直接求mean
            if len(sameClassSample.shape) >= 2:
                self.center[i, :] = np.mean(sameClassSample, axis=0)

        # 计算所有类平均距离之和的平均值
        self.centerMeanDist = 0
        for i in range(self.centerNum):
            index = np.argwhere(self.label == i).squeeze()
            sameClassSample = self.data[index, :]

            # 如果只有一个样本，需要reshape成二维数组以计算距离
            if len(sameClassSample.shape) < 2:
                sameClassSample = sameClassSample.reshape((1, -1))

            # 计算该类的平均距离：类中所有样本点到中心的欧氏距离均值
            dist = np.mean(euclidean_distances(sameClassSample, self.center[i, :].reshape((1, -1))))
            self.centerMeanDist += dist

        # 对所有类的平均距离求平均值
        self.centerMeanDist /= self.centerNum

    def divide(self):
        """
        对类进行分裂操作(divide)。
        分裂条件：
        - 若类的最大标准差超过阈值thetaS，并且类中样本数和类间距离等条件满足要求，则分裂该类为两个类。

        分裂方法：
        在最大标准差的维度上对该类中心进行轻微扰动，形成两个新中心。
        """
        # 暂存新的中心集合，用于更新
        newCenterSet = self.center

        # 遍历每个类，检查是否需要分裂
        for i in range(self.centerNum):
            index = np.argwhere(self.label == i).squeeze()
            sameClassSample = self.data[index, :]

            # 计算该类样本的各维度方差(即标准差的平方)
            stdEachDim = np.mean((sameClassSample - self.center[i, :]) ** 2, axis=0)

            # 若当前类只有一个样本，stdEachDim可能不是数组
            if type(stdEachDim) is not np.ndarray:
                maxStd = stdEachDim
                sameClassSample = sameClassSample.reshape(1, -1)
            else:
                # 找到最大标准差对应的维度
                maxIndex = np.argmax(stdEachDim)
                maxStd = stdEachDim[maxIndex]

            # 计算该类到中心的平均距离
            distance = np.mean(euclidean_distances(sameClassSample, self.center[i, :].reshape((1, -1))))

            # 判断分裂条件：若最大标准差 > θS
            if maxStd > self.thetaS:
                # 附加条件：类数过少(<= K/2) 或者类中样本多且距离较大
                if self.centerNum <= self.K // 2 or \
                        (sameClassSample.shape[0] > 2 * (self.thetaN + 1) and distance >= self.centerMeanDist): # TODO: 补全下面的条件判断:
                        # 引导式注释：1. 第一个条件：当当前类数(self.centerNum)小于等于期望类数的一半(self.K // 2)，说明类数过少，需要增加类数（分裂）。
                        # 2. 第二个条件：当该类样本数大于 2*(θN + 1) 且该类平均距离大于等于全局平均距离(self.centerMeanDist)时，也有分裂的需求。
                    # 创建两个新中心
                    newCenterFirst = self.center[i, :].copy()
                    newCenterSecond = self.center[i, :].copy()

                    # 在最大标准差维度上进行轻微扰动
                    newCenterFirst[maxIndex] += 0.5 * maxStd
                    newCenterSecond[maxIndex] -= 0.5 * maxStd

                    # 删除原中心，加入新的两个中心
                    newCenterSet = np.delete(newCenterSet, i, axis=0)
                    newCenterSet = np.vstack((newCenterSet, newCenterFirst))
                    newCenterSet = np.vstack((newCenterSet, newCenterSecond))
                else:
                    # 不满足分裂条件则不进行操作
                    continue

        # 分裂完成后更新中心集合和中心数量
        self.center = newCenterSet
        self.centerNum = self.center.shape[0]

    def combine(self):
        """
        对类进行合并操作(combine)。

        若两类中心的距离小于θC，则考虑合并。
        按照类样本数加权求平均得到新的合并中心。
        每次合并不超过L对类。

        步骤：
        1. 计算所有类中心之间的距离矩阵。
        2. 找到距离最小的两个中心，若距离 < θC，则合并。
        3. 用两个类的样本数加权平均来计算新中心。
        4. 重复直到合并次数达到L或不存在需合并的类对。
        """
        # 用于存储需要删除的中心索引，因为合并会使两个旧中心消失
        delIndexList = []

        # 计算所有中心之间的距离
        centerDist = euclidean_distances(self.center, self.center)
        # 对角线置为极大值，避免自己与自己比较
        centerDist += (np.eye(self.centerNum)) * 10 ** 10

        for i in range(self.L):
            # 寻找距离最小的两个中心
            minDist = np.min(centerDist)
            if minDist >= self.thetaC:
                # 若最小距离大于阈值，停止合并
                break
            index = np.argmin(centerDist)
            # 根据index推算出行列，对应两个类中心
            row = index // self.centerNum
            col = index % self.centerNum

            # 找出两个类的样本数
            index_row = np.argwhere(self.label == row)
            classNumFirst = len(index_row)
            index_col = np.argwhere(self.label == col)
            classNumSecond = len(index_col)

            # 加权平均计算新中心
            newCenter = self.center[row, :] * (classNumFirst / (classNumFirst + classNumSecond)) + \
                        self.center[col, :] * (classNumSecond / (classNumFirst + classNumSecond))

            # 记录需要删除的旧中心的索引
            delIndexList.append(row)
            delIndexList.append(col)

            # 将新中心添加到中心集合中
            self.center = np.vstack((self.center, newCenter))
            self.centerNum -= 1

            # 将已合并的两个中心距离设置为∞，避免重复合并
            centerDist[row, :] = float("inf")
            centerDist[col, :] = float("inf")
            centerDist[:, col] = float("inf")
            centerDist[:, row] = float("inf")

        # 删除已合并的旧中心
        self.center = np.delete(self.center, delIndexList, axis=0)
        self.centerNum = self.center.shape[0]

    def drawResult(self):
        """
        绘制当前聚类结果。

        每个类用不同的颜色表示，类中心用'x'标记。
        """
        color = ['r', 'b', 'g', 'c', 'm', 'y']
        ax = plt.gca()
        ax.clear()
        for i in range(self.centerNum):
            index = np.argwhere(self.label == i).squeeze()
            # 绘制该类的所有样本点
            ax.scatter(self.data[index, 0], self.data[index, 1], c=color[i], label=f'Cluster {i}')
            # 绘制该类的中心点（x标记）
            ax.scatter(self.center[i, 0], self.center[i, 1], c=color[i], marker='x', label=f'Centroid {i}')
        plt.title('ISODATA Clustering')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()

    def start(self):
        """
        启动ISODATA算法的迭代过程：

        1. 初始化分配标签并绘制结果。
        2. 在每次迭代中，判断类的数量和分布情况，决定是分裂还是合并。
        3. 更新标签和中心，绘制结果，直到迭代结束。
        """
        # 初始更新标签和中心，并绘制初始聚类结果
        self.updateLabel()
        self.drawResult()

        # 迭代iterationNum次
        for i in range(self.iteration):
            # 根据当前类数和迭代步数，决定执行分裂或合并操作
            if self.centerNum < self.K // 2: # TODO: 判断当前类的数量是否过少。 如果当前类中心的数量小于期望分类数的一半，说明类数过少，需要通过分裂操作增加类数。
                # 若当前类数小于期望类数的一半，则优先进行分裂操作以增加类数
                self.divide()
            elif (i > 0 and i % 2 == 0) or self.centerNum > 2 * self.K: # TODO: 判断当前是否需要进行合并操作。1.在偶数次迭代时优先尝试进行合并操作。2.如果类中心数过多，则尝试通过合并操作减少分类数。
                # 若是偶数次迭代或类数已过多，则尝试合并类减少类数
                self.combine()
            else:
                # 否则默认尝试分裂
                self.divide()

            # 分裂或合并后更新标签和中心并绘制结果
            self.updateLabel()
            print("中心数量：{}".format(self.centerNum))
            self.drawResult()


if __name__ == "__main__":
    # 创建ISODATA实例并开始运行
    # 参数说明：designCenterNum=2(期望最终类数为2)，Nc=3(初始类数为3)、
    # LeastSampleNum=1、StdThred=0.1、LeastCenterDist=2、L=3、iterationNum=5
    isoData = ISODATA(designCenterNum=2, 
                      Nc=3, 
                      LeastSampleNum=1, 
                      StdThred=0.1, 
                      LeastCenterDist=2, 
                      L=3, 
                      iterationNum=5)
    isoData.start()