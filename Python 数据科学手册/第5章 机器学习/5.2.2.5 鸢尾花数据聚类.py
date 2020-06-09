# !/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
Created on Fri Oct 11 20:14:02 2019

@author: SHI YI
"""

# 聚类：对没有任何标签的数据集进行分组
# 高斯混合模型：试图将数据构造成若干服从高斯分布的概率密度函数簇


import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA


iris = sns.load_dataset("iris")
X_iris = iris.drop("species", axis=1)

model = GaussianMixture(n_components=3, covariance_type='full')
model.fit(X_iris)
y_gmm = model.predict(X_iris)
iris['cluster'] = y_gmm

model = PCA(n_components=2)     # 2.设置超参数，初始化模型
model.fit(X_iris)               # 3.拟合数据，注意这里不用 y 变量
X_2D = model.transform(X_iris)  # 4.将数据转换为二维

iris['PCA1'] = X_2D[ : , 0]
iris['PCA2'] = X_2D[ : , 1]

sns.lmplot('PCA1', 'PCA2', data=iris, hue='species', col='cluster', fit_reg=False)
