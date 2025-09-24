import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import euclidean_distances
import copy
import pandas as pd

class ISODATA():
    def __init__(self, designCenterNum, Nc, LeastSampleNum, StdThred, LeastCenterDist, L, iterationNum, data):
        # 初始化ISODATA相关参数
        self.K = designCenterNum
        self.centerNum = Nc
        self.thetaN = LeastSampleNum
        self.thetaS = StdThred
        self.thetaC = LeastCenterDist
        self.L = L
        self.iteration = iterationNum

        # 使用传入的数据集
        self.data = data
        # 初始标签随机分配或全为0
        self.label = np.zeros(self.data.shape[0], dtype=int)

        # 随机选择Nc个初始聚类中心
        center_ind = np.random.choice(np.arange(len(self.data), dtype=np.int32), Nc, replace=False)
        self.center = np.stack([copy.deepcopy(self.data[center_ind[i], :]) for i in range(Nc)])
        self.centerMeanDist = 0

    def updateLabel(self):
        distance = euclidean_distances(self.data, self.center.reshape((self.centerNum, -1)))
        self.label = np.argmin(distance, axis=1)

        for i in range(self.centerNum):
            index = np.argwhere(self.label == i).squeeze()
            sameClassSample = self.data[index, :]
            if len(sameClassSample.shape) >= 2 and sameClassSample.shape[0] > 0:
                self.center[i, :] = np.mean(sameClassSample, axis=0)

        self.centerMeanDist = 0
        for i in range(self.centerNum):
            index = np.argwhere(self.label == i).squeeze()
            sameClassSample = self.data[index, :]
            if sameClassSample.ndim == 1:
                sameClassSample = sameClassSample.reshape(1, -1)
            if len(sameClassSample) > 0:
                dist = np.mean(euclidean_distances(sameClassSample, self.center[i, :].reshape((1, -1))))
                self.centerMeanDist += dist
        if self.centerNum > 0:
            self.centerMeanDist /= self.centerNum

    def divide(self):
        newCenterSet = self.center
        for i in range(self.centerNum):
            index = np.argwhere(self.label == i).squeeze()
            sameClassSample = self.data[index, :]
            if sameClassSample.shape[0] == 0:
                continue
            stdEachDim = np.mean((sameClassSample - self.center[i, :]) ** 2, axis=0)
            if stdEachDim.ndim == 0:
                maxStd = stdEachDim
                sameClassSample = sameClassSample.reshape(1, -1)
            else:
                maxIndex = np.argmax(stdEachDim)
                maxStd = stdEachDim[maxIndex]

            distance = np.mean(euclidean_distances(sameClassSample, self.center[i, :].reshape((1, -1))))
            if maxStd > self.thetaS:
                if self.centerNum <= self.K // 2 or (sameClassSample.shape[0] > 2 * (self.thetaN + 1) and distance >= self.centerMeanDist):
                    newCenterFirst = self.center[i, :].copy()
                    newCenterSecond = self.center[i, :].copy()
                    newCenterFirst[maxIndex] += 0.5 * maxStd
                    newCenterSecond[maxIndex] -= 0.5 * maxStd
                    newCenterSet = np.delete(newCenterSet, i, axis=0)
                    newCenterSet = np.vstack((newCenterSet, newCenterFirst))
                    newCenterSet = np.vstack((newCenterSet, newCenterSecond))
                else:
                    continue
            else:
                continue

        self.center = newCenterSet
        self.centerNum = self.center.shape[0]

    def combine(self):
        """
        合并操作：对类进行合并。
        若两个类中心之间的距离小于最小距离阈值 θC，则进行合并。
        """
        delIndexList = []  # 记录被合并的中心索引

        if self.centerNum <= 1:
            # 如果中心数少于等于1，则无需合并
            return

        # 计算所有中心之间的距离
        centerDist = euclidean_distances(self.center, self.center)
        # 对角线设置为极大值，防止中心自己与自己合并
        centerDist += np.eye(self.centerNum) * 1e10

        for _ in range(self.L):  # 每次允许合并的最大中心对数为 L
            if self.centerNum < 2:
                # 如果中心数目少于2，无法再合并
                break

            # 找到最小距离
            minDist = np.min(centerDist)
            if minDist >= self.thetaC:
                # 如果最小距离大于等于最小距离阈值 θC，则停止合并
                break

            # 找到距离最小的中心对的索引
            index = np.argmin(centerDist)
            row = index // self.centerNum
            col = index % self.centerNum

            # 防止重复操作或越界
            if row >= self.centerNum or col >= self.centerNum:
                continue

            # 找到两个类的样本数，用于加权平均计算新中心
            index_row = np.argwhere(self.label == row)
            classNumFirst = len(index_row)
            index_col = np.argwhere(self.label == col)
            classNumSecond = len(index_col)

            total = classNumFirst + classNumSecond
            if total == 0:
                # 若总样本数为0，跳过
                continue

            # 计算新中心（加权平均）
            newCenter = (self.center[row, :] * (classNumFirst / total) +
                         self.center[col, :] * (classNumSecond / total))

            # 将合并后的新中心添加到中心集合
            self.center = np.vstack((self.center, newCenter))

            # 删除旧中心
            delIndexList.append(row)
            delIndexList.append(col)

            # 更新距离矩阵，防止重复操作
            centerDist[row, :] = float("inf")
            centerDist[col, :] = float("inf")
            centerDist[:, row] = float("inf")
            centerDist[:, col] = float("inf")

        # 删除被合并的旧中心
        if delIndexList:
            self.center = np.delete(self.center, delIndexList, axis=0)
            self.centerNum = self.center.shape[0]


    def drawResult(self):
        if self.centerNum == 0:
            print("No centers to draw.")
            return
        color = ['r', 'b', 'g', 'c', 'm', 'y']*10  # 扩展颜色数量
        plt.figure(figsize=(6,6))
        for i in range(self.centerNum):
            index = np.argwhere(self.label == i).squeeze()
            if index.size > 0:
                plt.scatter(self.data[index, 0], self.data[index, 1], c=color[i], s=10, alpha=0.5)
            plt.scatter(self.center[i, 0], self.center[i, 1], c=color[i], marker='x', s=100, linewidths=2)

        plt.title('ISODATA Clustering')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    def start(self):
        self.updateLabel()
        self.drawResult()
        for i in range(self.iteration):
            if self.centerNum < self.K // 2:
                self.divide()
            elif (i > 0 and i % 2 == 0) or self.centerNum > 2 * self.K:
                self.combine()
            else:
                self.divide()

            self.updateLabel()
            print("Iteration: {}, Center count: {}".format(i+1, self.centerNum))
            self.drawResult()


if __name__ == "__main__":
    # 直接从CSV读取数据作为测试集
    data_df = pd.read_csv("data.csv")
    # 只取x,y作为特征
    X_loaded = data_df[["x","y"]].values

    # 使用ISODATA进行聚类
    isoData = ISODATA(designCenterNum = 3, 
                      Nc = 3, 
                      LeastSampleNum = 20, 
                      StdThred = 3, 
                      LeastCenterDist = 5, 
                      L = 2, 
                      iterationNum = 20, 
                      data = X_loaded)
    isoData.start()
