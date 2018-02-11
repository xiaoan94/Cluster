#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import math
import pylab
import matplotlib
import matplotlib.pyplot as plt


class KMeans(object):

    def __init__(self, filename, k):
        self.filename = filename  # 包含数据的文件名
        self.k = k  # 聚类的类别总数

        self.dataMat = []  # 原始数据的矩阵
        self.centroids = np.mat  # K个随机质心

    # 下载聚类所用的原始数据
    def loadDataSet(self):
        """
        fr = open(self.filename)  # 打开文件
        for line in fr.readlines():  # 逐行读取文件
            curLine = line.strip().split('\t')  #  除去每一行的前后空格及换行符，然后以空格分隔每一行的数据
            fltLine = map(float, curLine)  将curLine中的每一个元素转化成float型
            self.dataMat.append(fltLine)
        """
        self.dataMat = np.loadtxt(self.filename)  # 直接下载文件，并将数据转化为浮点数，并用矩阵形式输出，默认是用空格分隔数据
        #print type(self.dataMat), self.dataMat.dtype
        print np.shape(self.dataMat)

    # 训练数据的样本展示图
    def show(self):
        self.loadDataSet()
        fig = plt.figure(figsize=(12, 6))  # 创建一幅图，指定图的尺寸。
        pylab.mpl.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体
        plt.suptitle(u'训练样本散点图')  # 整个图的大标题
        p1 = fig.add_subplot(1, 1, 1)  # 将图分割成1行1列，图像画在从左到右从上到下的第1块
        p1.set_title(u'图1')  # 子图标题
        p1.set_xlabel("x")  # 子图x横坐标标注
        p1.set_ylabel("y")  # 子图y纵坐标标注
        T = np.arctan2(self.dataMat[:, 0], self.dataMat[:, 1])
        p1.scatter(self.dataMat[:, 0], self.dataMat[:, 1], c=T, s=25, alpha=0.4, marker='o')
        plt.show()  # 显示

    # 计算两个向量的欧氏距离
    def distEclud(self, vecA, vecB):
        return math.sqrt(sum(np.power(vecA - vecB, 2)))

    # 计算K个族的随机质心，初始化质心
    def randCent(self):
        self.loadDataSet()
        n = np.shape(self.dataMat)[1]  # 数据矩阵的列数
        self.centroids = np.mat(np.zeros((self.k, n)))  # 生成一个维度为k*n的零矩阵，行数为聚类的类别总数k，列数为数据矩阵的列数n
        for j in range(n):
            minJ = min(self.dataMat[:, j])  # 数据矩阵每一列的最小值
            rangeJ = float(max(self.dataMat[:, j]) - minJ)  # 数据矩阵的每一列的取值范围，即数据的每一个特征对应的特征值的取值范围，数据矩阵的每一列的最大值减最小值
            self.centroids[:, j] = minJ + rangeJ * np.random.rand(self.k, 1)  # 生成k*n的随机质心矩阵，每个随机值均在每一列的特征值的范围内
        self.centroids = np.mat(self.centroids)
        #print self.centroids

    # kmeans算法的迭代过程
    def KMeans_main(self):
        self.randCent()
        m = np.shape(self.dataMat)[0]  # 数据矩阵的行数
        clusterAssment = np.mat(np.zeros((m, 2)))   # 初始化簇分配结果矩阵。两列：一列记录簇索引值；一列存储误差（误差指当前点到簇质心的距离）
        clusterChanged = True  # 聚类迭代的标志变量
        while clusterChanged:
            clusterChanged = False
            for i in range(m):
                minDist = np.inf  # 初始化
                minIndex = -1
                for j in range(self.k):
                    distJI = self.distEclud(self.centroids[j, :], self.dataMat[i, :])
                    if distJI < minDist:
                        minDist = distJI
                        minIndex = j
                if clusterAssment[i, 0] != minIndex:
                    clusterChanged = True
                clusterAssment[i, :] = minIndex
                minDist = minDist**2
            print self.centroids
            for cent in range(self.k):
                ptsInClust = self.dataMat[np.nonzero(clusterAssment[:, 0].A==cent)[0]]
                self.centroids[cent, :] = np.mean(ptsInClust, axis=0)
        return centrorids, clusterAssment











if __name__ == "__main__":
    tit = KMeans("testSet.txt", 10)
    #tit.loadDataSet()
    #print tit.dataMat
    tit.randCent()
    #tit.show()














