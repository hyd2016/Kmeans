from numpy import *
import matplotlib.pyplot as plt

from sklearn.datasets.samples_generator import make_blobs


def dateset():
    # feature为样本特征，lable为样本簇类别， 共500个样本，每个样本2个特征，共4个簇，簇中心在[-1,-1], [0,0],[1,1], [1,-1]， 簇方差分别为[0.4, 0.2, 0.3, 0.3]
    feature, lable = make_blobs(n_samples=500, n_features=2, centers=[[-1, -1], [0, 0], [1, 1], [1, -1]],
                                cluster_std=[0.4, 0.2, 0.3, 0.3],
                                random_state=5)
    return feature, lable


# 欧式距离计算
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))  # 格式相同的两个向量做运算


# 中心点生成 随机生成最小到最大值之间的值
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))  # 创建中心点，由于需要与数据向量做运算，所以每个中心点与数据得格式应该一致（特征列）
    for j in range(n):  # 循环所有特征列，获得每个中心点该列的随机值
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))  # 获得每列的随机值 一列一列生成
    return centroids


# 返回 中心点矩阵和聚类信息
def kMeans(dataSet, k):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))  # 创建一个矩阵用于记录该样本 （所属中心点 与该点距离）
    centroids = randCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False  # 如果没有点更新则为退出
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):  # 每个样本点需要与 所有 的中心点作比较
                distJI = distEclud(centroids[j, :], dataSet[i, :])  # 距离计算
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:  # 若记录矩阵的i样本的所属中心点更新，则为True，while下次继续循环更新
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2  # 记录该点的两个信息
        for cent in range(k):  # 重新计算中心点
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]  # 得到属于该中心点的所有样本数据
            centroids[cent, :] = mean(ptsInClust, axis=0)  # 求每列的均值替换原来的中心点
    return centroids, clusterAssment


if __name__ == "__main__":
    datMat, target = dateset()

    myCentroids, clustAssing = kMeans(datMat, 4)
    print(myCentroids)

    fig = plt.figure()

    plt.scatter(datMat[:, 0], datMat[:, 1], c=target)
    plt.scatter(myCentroids[:, 0].tolist(), myCentroids[:, 1].tolist(), marker="x", color='r', s=60)
    plt.show()
