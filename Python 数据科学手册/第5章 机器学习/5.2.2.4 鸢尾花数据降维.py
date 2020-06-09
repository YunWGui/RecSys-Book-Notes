# !/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
Created on Fri Oct 11 19:59:58 2019

@author: SHI YI
"""

# 降维的任务：找到一个可以保留数据本质特征的低维矩阵来表示高维数据。
# 降维通常用于辅助数据可视化的工作
# PCA：主成分分析
# 此处将用模型返回两个主成分，也就是用二维数据表示鸢尾花的四维数据


import seaborn as sns
from sklearn.decomposition import PCA  # 1.选择模型类


iris = sns.load_dataset("iris")
X_iris = iris.drop('species', axis=1)
y_iris = iris['species']

model = PCA(n_components=2)     # 2.设置超参数，初始化模型
model.fit(X_iris)               # 3.拟合数据，注意这里不用 y 变量
X_2D = model.transform(X_iris)  # 4.将数据转换为二维

iris['PCA1'] = X_2D[ : , 0]
iris['PCA2'] = X_2D[ : , 1]
sns.lmplot('PCA1', 'PCA2', hue='species', data=iris, fit_reg=False)